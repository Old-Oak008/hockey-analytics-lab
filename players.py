"""Individual NHL skater estimates, conditional on appearing in a regular-season game.
No team model or sportsbook data; historical dates strictly precede target date.
"""
import math
import pandas as pd

TEAMS='ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR UTA VAN VGK WPG WSH'.split()

def season_start(day):
    day=pd.Timestamp(day)
    return day.year-int(day.month<7)

def roster(payload):
    def name(v): return v.get('default','') if isinstance(v,dict) else str(v or '')
    rows={}
    for group in ('forwards','defensemen'):
        for p in payload.get(group,[]):
            pid=int(p['id']); label=(name(p.get('firstName'))+' '+name(p.get('lastName'))).strip()
            if label: rows[pid]=f'{label} ({p.get("positionCode", "?")})'
    if not rows: raise ValueError('API nevrátilo soupisku hráčů v poli.')
    return dict(sorted(rows.items(),key=lambda x:x[1].casefold()))

def seconds(v):
    try:
        m,s=map(int,str(v).split(':'))
        return m*60+s if m>=0 and 0<=s<60 else float('nan')
    except (ValueError,TypeError): return float('nan')

def logs(payloads):
    rows=[]
    for payload in payloads:
        if not isinstance(payload.get('gameLog'),list): raise ValueError('API nevrátilo očekávané zápasové statistiky.')
        for g in payload['gameLog']:
            gid=str(g['gameId'])
            if len(gid)!=10 or gid[4:6]!='02': continue
            points,goals=int(g['points']),int(g['goals'])
            if not 0<=goals<=points: raise ValueError('Rozporné statistiky hráče.')
            rows.append(dict(game=gid,date=pd.Timestamp(g['gameDate']).normalize(),team=g.get('teamAbbrev',''),points=points,goals=goals,toi=seconds(g.get('toi')),shots=g.get('shots')))
    if not rows: raise ValueError('Žádná dostupná utkání základní části.')
    d=pd.DataFrame(rows).drop_duplicates()
    if d.game.duplicated().any(): raise ValueError('Rozporné duplicity v zápasových datech.')
    return d.sort_values(['date','game']).reset_index(drop=True)

def before(d,day): return d[d.date<pd.Timestamp(day).normalize()].sort_values(['date','game']).tail(100)

def probabilities(d):
    if len(d)<10: raise ValueError('Nejméně 10 odehraných zápasů před datem analýzy je nutných.')
    result={}
    for target in ('points','goals'):
        y=d[target].gt(0).astype(int)
        base=(y.sum()+.5)/(len(y)+1)
        recent=y.tail(20); older=y.iloc[:-20]
        if len(older):
            prior=(older.sum()+.5)/(len(older)+1)
            weight=min(20,len(older))
            adaptive=(recent.sum()+weight*prior)/(len(recent)+weight)
        else: adaptive=base
        result[target]=(float(base),float(adaptive))
    return result

def profile(d,day):
    past=before(d,day); p=probabilities(past)
    rows=[]
    for n in (5,10,20,100):
        m=past.tail(n)
        rows.append({'Okno':f'Posledních {n}','Zápasy':len(m),'Body / zápas':m.points.mean(),'Zápasy s bodem':m.points.gt(0).mean(),'Góly / zápas':m.goals.mean(),'Prům. čas (min)':m.toi.mean()/60})
    return p,pd.DataFrame(rows),past

def backtest(d,day):
    past=d[d.date<pd.Timestamp(day).normalize()].sort_values(['date','game']); rows=[]
    for date, games in past.groupby('date',sort=True):
        hist=before(past,date)
        if len(hist)<20: continue
        p=probabilities(hist)
        for g in games.itertuples():
            for target in ('points','goals'):
                y=int(getattr(g,target)>0)
                for model,q in zip(('Dlouhodobý','Forma (experiment)'),p[target]):
                    rows.append(dict(date=date.date(),game=g.game,target=target,model=model,p=q,actual=y,brier=(q-y)**2,logloss=-math.log(max(1e-12,q if y else 1-q))))
    if not rows: raise ValueError('Není dostatek zápasů pro historický test (minimum 21).')
    detail=pd.DataFrame(rows)
    summary=detail.groupby(['target','model'],as_index=False).agg(Zápasy=('actual','size'),Brier=('brier','mean'),LogLoss=('logloss','mean'),Predikce=('p','mean'),Skutečnost=('actual','mean'))
    return summary,detail
