"""Optional API-Hockey import; no credentials or responses on disk/global cache.
Use only with the user's own authorized API account. No historical data bundled.
"""
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import requests
import pandas as pd

BASE='https://v1.hockey.api-sports.io/'
COUNTRIES={'Czech-Republic','Finland','Sweden','Germany','Switzerland','United-Kingdom'}
COLS=['date','home','away','hg','ag']
TZ=ZoneInfo('Europe/Prague')
FINAL={'FT','AOT','AP'}


def request(endpoint, params, key):
    if endpoint not in {'leagues','games'} or not isinstance(key,str) or not key.strip():
        raise ValueError('Chybí vlastní API klíč.')
    try:
        r=requests.get(BASE+endpoint,params=params,headers={'x-apisports-key':key.strip()},timeout=35,allow_redirects=False)
    except requests.RequestException:
        raise ValueError('API není dostupné. Zkus to později.') from None
    if r.status_code==429: raise ValueError('API limit požadavků. Zkus to po obnovení kvóty.')
    if r.status_code in (401,403): raise ValueError('API klíč nebo tarif nemá přístup.')
    if r.status_code!=200: raise ValueError(f'API odpovědělo HTTP {r.status_code}.')
    try: p=r.json()
    except ValueError: raise ValueError('API nevrátilo JSON.') from None
    if not isinstance(p,dict): raise ValueError('Neočekávaný formát API.')
    if p.get('errors'):
        # Never return arbitrary upstream text that might contain credentials.
        msg=str(p['errors']).lower()
        if 'season' in msg or 'plan' in msg or 'subscription' in msg:
            raise ValueError('Sezona nebo soutěž není dostupná v tomto tarifu (nebo ji API nezná).')
        if 'limit' in msg or 'quota' in msg or 'request' in msg: raise ValueError('API odmítlo požadavek; zkontroluj kvótu a přístup.')
        raise ValueError('API odmítlo požadavek. Zkontroluj klíč, tarif a parametry.')
    rows=p.get('response')
    if not isinstance(rows,list) or len(rows)>10000: raise ValueError('Neočekávaný rozsah odpovědi API.')
    if isinstance(p.get('paging'),dict) and p['paging'].get('total',1)>1:
        raise ValueError('Odpověď je stránkovaná; neúplnou historii nepřijímáme.')
    if isinstance(p.get('results'),int) and p['results']!=len(rows): raise ValueError('Nesouhlasí počet vrácených zápasů.')
    return rows


def leagues(country,key):
    if country not in COUNTRIES: raise ValueError('Neplatná země.')
    out={}
    for x in request('leagues',{'country':country},key):
        lg=x.get('league',x); c=x.get('country',{})
        if not isinstance(lg,dict) or not isinstance(c,dict): continue
        if c.get('name') and c['name'].lower().replace(' ','-')!=country.lower():
            # UK may be represented as Great Britain; do not silently import another country.
            continue
        if lg.get('type')!='League' or not isinstance(lg.get('id'),int) or not isinstance(lg.get('name'),str): continue
        years=[]
        for s in x.get('seasons',lg.get('seasons',[])):
            y=s.get('season',s.get('year')) if isinstance(s,dict) else s
            if str(y).isdigit() and 2000<=int(y)<=2100: years.append(int(y))
        out[lg['id']]={'id':lg['id'],'name':lg['name'],'years':sorted(set(years),reverse=True)}
    return sorted(out.values(),key=lambda x:x['name'])


def games(league,season,key):
    return request('games',{'league':int(league),'season':int(season),'timezone':'Europe/Prague'},key)


def label(game):
    stage=game.get('stage')
    if isinstance(stage,str) and stage.strip(): return 'Fáze: '+stage.strip()
    # A missing stage is never silently declared a regular season.
    week=game.get('week')
    if week is None or week=='': return 'Bez označení fáze / kola'
    if isinstance(week,(str,int)):
        w=str(week).strip()
        if re.fullmatch(r'\d+|(?:round|week|kolo)\s*\d+',w,re.I): return 'Číslované kolo (fáze neověřena)'
        return 'Kolo: '+w
    return 'Bez označení fáze / kola'


def groups(seasons):
    return Counter(label(g) for records in seasons.values() for g in records if isinstance(g,dict))


def score(v):
    if isinstance(v,str) and re.fullmatch(r'\d{1,2}',v.strip()): v=int(v)
    if type(v) is not int or not 0<=v<=30: raise ValueError('Neplatné dílčí skóre.')
    return v


def period_pair(value):
    # Missing values are distinct from zero and from malformed values.
    if value is None or value=='': return (None,None)
    if isinstance(value,dict):
        return tuple(None if value.get(k) in (None,'') else score(value[k]) for k in ('home','away'))
    if isinstance(value,str):
        m=re.fullmatch(r'\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*',value)
        if m: return tuple(score(v) for v in m.groups())
    raise ValueError('Neznámý formát skóre třetiny.')


def regulation_with_source(g):
    status=g['status']['short']
    f=(score(g['scores']['home']),score(g['scores']['away']))
    if status=='FT': expected=f
    elif status in ('AOT','AP') and abs(f[0]-f[1])==1:
        # Standard final score credits the winner with exactly one extra goal.
        expected=(min(f),min(f))
    else: raise ValueError('Nejednoznačný konečný výsledek.')
    periods=g.get('periods')
    if periods is None: periods={}
    if not isinstance(periods,dict): raise ValueError('Neznámý formát třetin.')
    pairs=[period_pair(periods.get(p)) for p in ('first','second','third')]
    complete=all(v is not None for pair in pairs for v in pair)
    for i in range(2):
        known=[pair[i] for pair in pairs if pair[i] is not None]
        if sum(known)>expected[i] or (len(known)==3 and sum(known)!=expected[i]):
            raise ValueError('Nesouhlasí skóre třetin a výsledek.')
    if complete: return (*expected,'Třetiny')
    # If periods are absent, use explicit finished status and a valid final score.
    # Never turn unknown status, missing totals or inconsistent periods into results.
    return (*expected,'Konečné skóre + stav')


def regulation(g):
    x,y,_=regulation_with_source(g)
    return x,y


def convert(seasons,league,selected,today=None):
    today=today or datetime.now(TZ).date(); selected=set(selected)
    if not selected: raise ValueError('Vyber fázi základní části.')
    counts=Counter(); names={}; records={}
    for year,items in sorted(seasons.items()):
        for g in items:
            if not isinstance(g,dict): counts['Neplatná struktura']+=1; continue
            if label(g) not in selected: counts['Jiná fáze']+=1; continue
            try:
                if int(g['league']['id'])!=int(league) or int(g['league']['season'])!=int(year): raise ValueError('Jiná liga nebo sezona.')
                gid=g['id']; assert type(gid) is int
                stamp=pd.Timestamp(g['date'])
                if pd.isna(stamp): raise ValueError('Chybí datum.')
                if stamp.tzinfo is None:
                    zone=g.get('timezone')
                    if zone not in ('UTC','Europe/Prague'): raise ValueError('Chybí časové pásmo.')
                    stamp=stamp.tz_localize(zone)
                day=stamp.tz_convert(TZ).date()
                if not datetime(year,7,1).date()<=day<datetime(year+1,7,1).date(): raise ValueError('Datum mimo sezonu.')
                h,a=g['teams']['home'],g['teams']['away']; hi,ai=h['id'],a['id']
                assert type(hi) is int and type(ai) is int and hi!=ai
                hn,an=h['name'].strip(),a['name'].strip(); assert hn and an
                status=g['status']['short']
                if status in FINAL:
                    if day>=today: counts['Dnešní/budoucí výsledek vynechán']+=1; continue
                    try: x,y,method=regulation_with_source(g)
                    except (KeyError,TypeError,ValueError,AttributeError) as e:
                        counts['Neplatný nebo chybějící údaj']+=1
                        safe={'Neplatné dílčí skóre.','Neznámý formát skóre třetiny.','Nejednoznačný konečný výsledek.','Neznámý formát třetin.','Nesouhlasí skóre třetin a výsledek.'}
                        reason=str(e) if isinstance(e,ValueError) and str(e) in safe else 'Chybí nebo je neplatná struktura skóre'
                        counts['Důvod: '+reason]+=1
                        continue
                    kind='Výsledek'
                elif status=='NS' and day>=today:
                    x,y=None,None; kind='Rozpis'
                else:
                    counts['Nedokončeno / zrušeno']+=1; continue
            except (KeyError,TypeError,ValueError,AttributeError,AssertionError,OverflowError) as e:
                counts['Neplatný nebo chybějící údaj']+=1
                # Only expose controlled diagnostics, never raw upstream payload or key.
                safe={'Jiná liga nebo sezona.','Chybí datum.','Chybí časové pásmo.','Datum mimo sezonu.'}
                reason=str(e) if isinstance(e,ValueError) and str(e) in safe else 'Neplatná identita, datum nebo struktura zápasu'
                counts['Důvod: '+reason]+=1
                continue
            row=(day.isoformat(),hi,ai,x,y)
            if gid in records:
                if records[gid]!=row: raise ValueError('Rozporný záznam se stejným ID.')
                counts['Duplicita']+=1; continue
            records[gid]=row; counts[kind]+=1
            if kind=='Výsledek': counts['Metoda: '+method]+=1
            for tid,n in ((hi,hn),(ai,an)):
                if tid not in names or (day,n)>names[tid]: names[tid]=(day,n)
    names={i:v[1] for i,v in names.items()}; duplicates=Counter(names.values())
    names={i:(f'{n} [{i}]' if duplicates[n]>1 else n) for i,n in names.items()}
    d=pd.DataFrame([(day,names[h],names[a],x,y) for day,h,a,x,y in records.values()],columns=COLS)
    if d.duplicated(['date','home','away']).any(): raise ValueError('Více záznamů se stejnými týmy a datem; import odmítnut.')
    if not d.empty:
        for c in ['hg','ag']: d[c]=d[c].astype('Int64')
        d=d.sort_values(['date','home','away']).reset_index(drop=True)
    return d,counts
