"""On-demand NHL archive evaluation. Only finished games, no credential access."""
from datetime import datetime, timezone
from numbers import Integral
import time
import numpy as np
import pandas as pd
from archive_store import clean, KEY

LINES=(4.5,5.5,6.5)
MAX_DAYS=30

def parse_day(payload, day):
    if not isinstance(payload,dict) or not isinstance(payload.get('games'),list):
        raise ValueError('Neočekávaný formát NHL API.')
    if payload.get('currentDate',day)!=day:
        raise ValueError('NHL API vrátilo jiné datum.')
    out={}
    for g in payload['games']:
        if g.get('gameDate')!=day or g.get('gameType')!=2 or g.get('gameState') not in ('FINAL','OFF'): continue
        if g.get('gameScheduleState','OK') not in ('OK',): continue
        try:
            h,a=g['homeTeam'],g['awayTeam']; x,y=h['score'],a['score']
            typ=g.get('gameOutcome',{}).get('lastPeriodType')
            if typ not in ('REG','OT','SO'): continue
            if any(isinstance(v,bool) or not isinstance(v,Integral) or v<0 for v in (x,y)): continue
            if typ=='REG' and x==y: continue
            if typ!='REG':
                if abs(x-y)!=1: continue
                if x>y: x-=1
                else: y-=1
            start=pd.Timestamp(g['startTimeUTC'])
            if start.tzinfo is None or pd.isna(start): continue
            start=start.tz_convert('UTC')
            k=(day,h['abbrev'],a['abbrev'])
            row=(int(x),int(y),start,int(g['id']))
        except (KeyError,ValueError,TypeError,OverflowError): continue
        if k in out and out[k]!=row: raise ValueError('Rozporné zápasy pro stejné týmy a datum.')
        out[k]=row
    return out

def refresh(records, known, fetch, today=None, max_days=MAX_DAYS, pause=.35, progress=None, now=None, after=None):
    r=clean(records); saved=dict(known); errors=[]
    now=pd.Timestamp(now or datetime.now(timezone.utc)); today=today or now.date().isoformat()
    days=sorted({g.date for g in r.itertuples() if g.date<=today and (g.date,g.home,g.away) not in saved})
    ordered=([d for d in days if d>after]+[d for d in days if d<=after]) if after else days
    selected=ordered[:max_days]; checked=0
    for i,day in enumerate(selected):
        if i and pause: time.sleep(pause)
        if progress: progress(i,len(selected),day)
        try:
            fresh=parse_day(fetch('/score/'+day),day)
            # Store only requested games, with a start in the past.
            wanted={(g.date,g.home,g.away) for g in r.itertuples() if g.date==day}
            saved.update({k:v for k,v in fresh.items() if k in wanted and v[2]<now})
            checked+=1
        except Exception as e:
            status=getattr(getattr(e,'response',None),'status_code',None)
            errors.append(f'{day}: '+(f'HTTP {status}' if status else 'výsledky nelze ověřit'))
            if status==429: break
    return saved,errors,checked,len(days)

def evaluate(records, known):
    r=clean(records).drop_duplicates(KEY,keep='first').copy()
    r=r.assign(_time=pd.to_datetime(r.created_utc,utc=True,format='mixed')).sort_values('_time',kind='stable').drop(columns='_time')
    matches=[]; rows=[]
    for _,g in r.iterrows():
        key=(g.date,g.home,g.away); actual=known.get(key)
        desc={'Datum':g.date,'Domácí':g.home,'Hosté':g.away,'Model 1X2':g.model_1x2,'Model góly':g.model_goals,'Původ':g.origin,'Vytvořeno UTC':g.created_utc}
        if actual is None:
            matches.append({**desc,'Stav':'Čeká na ověřený výsledek','Skóre 60 min':''}); continue
        x,y,start,gid=actual
        if pd.Timestamp(g.created_utc)>=start:
            matches.append({**desc,'Stav':'Vyloučeno – vytvořeno po začátku','Skóre 60 min':f'{x}:{y}'}); continue
        matches.append({**desc,'Stav':'Vyhodnoceno','Skóre 60 min':f'{x}:{y}'})
        p=g[['p1','px','p2']].to_numpy(float); z=0 if x>y else 1 if x==y else 2
        row={**desc,'Skóre 60 min':f'{x}:{y}','ID NHL':gid}
        rows.append({**row,'Trh':'1X2','Model':g.model_1x2,'Predikce':['1','X','2'][int(p.argmax())],'Skutečnost':['1','X','2'][z],'P skutečnosti':p[z],'Brier':float(np.sum((p-np.eye(3)[z])**2)),'Log Loss':float(-np.log(max(p[z],1e-12))),'Správně':int(p.argmax()==z)})
        for line in LINES:
            q=float(g[f'over_{line}']); event=x+y>line; prob=q if event else 1-q
            rows.append({**row,'Trh':f'Over {line}','Model':g.model_goals,'Predikce':'Over' if q>=.5 else 'Under','Skutečnost':'Over' if event else 'Under','P skutečnosti':prob,'Brier':(q-int(event))**2,'Log Loss':float(-np.log(max(prob,1e-12))),'Správně':int((q>=.5)==event)})
    games=pd.DataFrame(matches,columns=['Datum','Domácí','Hosté','Model 1X2','Model góly','Původ','Vytvořeno UTC','Stav','Skóre 60 min'])
    detail=pd.DataFrame(rows,columns=['Datum','Domácí','Hosté','Model 1X2','Model góly','Původ','Vytvořeno UTC','Skóre 60 min','ID NHL','Trh','Model','Predikce','Skutečnost','P skutečnosti','Brier','Log Loss','Správně'])
    detail=detail.drop_duplicates(['Datum','Domácí','Hosté','Trh','Model'],keep='first')
    summary=detail.groupby(['Trh','Model','Původ']).agg(Zápasů=('Brier','size'),Brier=('Brier','mean'),**{'Log Loss':('Log Loss','mean'),'Přesnost':('Správně','mean')}).reset_index()
    return summary,detail,games
