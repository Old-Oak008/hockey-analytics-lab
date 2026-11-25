"""European league-local history and open Czech standings. No scraped history."""
from io import BytesIO
import re
from datetime import date
import pandas as pd
import requests
from model import validate

LEAGUES={'Tipsport extraliga (API historie)':'elhapi','Maxa liga (API historie)':'maxaapi','Maxa liga':'maxa','SHL (Švédsko)':'shl','Liiga (Finsko)':'liiga','EIHL (Velká Británie)':'eihl','National League (Švýcarsko)':'nl','DEL (Německo)':'del'}
SLOTS=tuple('eu18_'+v for v in LEAGUES.values())
TABLE_URLS={'Maxa liga':'https://sazkyweb.cz/data/maxa-liga-tabulka.json','Tipsport extraliga':'https://sazkyweb.cz/data/extraliga-tabulka.json'}
COLS=['date','home','away','hg','ag']

def read_history(files, today=None):
    """CSV: exact ISO local dates, regulation scores; both empty for fixtures."""
    today=pd.Timestamp(today or date.today()).normalize()
    tables=[]
    for file in files:
        f=pd.read_csv(file, dtype=str, encoding='utf-8-sig',keep_default_na=False)
        if not set(COLS).issubset(f.columns): raise ValueError('CSV musí mít sloupce date,home,away,hg,ag.')
        f=f[COLS].copy()
        for c in COLS: f[c]=f[c].str.strip()
        if f.empty: raise ValueError('Prázdný CSV soubor.')
        if not f.date.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all(): raise ValueError('Datum musí být YYYY-MM-DD.')
        f['date']=pd.to_datetime(f.date,format='%Y-%m-%d',errors='raise')
        if (f.home.eq('')|f.away.eq('')|f.home.eq(f.away)).any(): raise ValueError('Chybí tým nebo jsou oba týmy stejné.')
        if f.hg.eq('').ne(f.ag.eq('')).any(): raise ValueError('Obě skóre musí být vyplněna, nebo obě prázdná.')
        done=f.hg.ne('')
        for c in ['hg','ag']:
            if not f.loc[done,c].str.fullmatch(r'\d{1,2}').all(): raise ValueError('Skóre musí být celé nezáporné číslo po 60 minutách.')
            f.loc[done,c]=f.loc[done,c].astype(int).astype(str)
        if (done & f.date.gt(today)).any(): raise ValueError('Budoucí zápas nesmí mít výsledek.')
        tables.append(f)
    if not tables: return pd.DataFrame(columns=COLS),[]
    allrows=pd.concat(tables,ignore_index=True).drop_duplicates()
    done=allrows[allrows.hg.ne('')]
    if done.duplicated(['date','home','away']).any(): raise ValueError('Rozporná skóre stejného zápasu; oprav zdroj.')
    hist=validate(done) if not done.empty else pd.DataFrame(columns=COLS)
    fixtures=sorted({(r.date.date().isoformat(),r.home,r.away) for r in allrows.itertuples()})
    return hist,fixtures

def parse_table(payload):
    if not isinstance(payload,dict) or payload.get('source')!='SazkyWeb.cz' or 'CC BY 4.0' not in str(payload.get('license','')): raise ValueError('Neočekávaný zdroj nebo licence.')
    updated=pd.Timestamp(payload['updated'])
    if pd.isna(updated) or updated.tzinfo is None: raise ValueError('Chybí čas aktualizace.')
    cols=['poradi','tym','zapasy','vyhry','vyhry_prodl','prohry_prodl','prohry','skore','body']
    d=pd.DataFrame(payload['tabulka'])[cols].copy()
    if len(d)!=14 or d.tym.duplicated().any() or not d.tym.map(lambda x:isinstance(x,str) and bool(x.strip())).all(): raise ValueError('Neúplná tabulka nebo duplicitní tým.')
    for c in ['poradi','zapasy','vyhry','vyhry_prodl','prohry_prodl','prohry','body']:
        v=pd.to_numeric(d[c],errors='raise')
        if v.isna().any() or not ((v>=0)&(v%1==0)).all(): raise ValueError('Neplatná tabulka.')
        d[c]=v.astype(int)
    if set(d.poradi)!=set(range(1,15)) or not d.zapasy.eq(d[['vyhry','vyhry_prodl','prohry_prodl','prohry']].sum(axis=1)).all() or not d.skore.astype(str).str.fullmatch(r'\d+:\d+').all(): raise ValueError('Nesouhlasí součty v tabulce.')
    return d.sort_values('poradi').reset_index(drop=True),updated.isoformat()

def fetch_table(league):
    r=requests.get(TABLE_URLS[league],timeout=15,headers={'User-Agent':'HockeyAnalyticsLab/1.8 (personal analytics)'})
    r.raise_for_status()
    return parse_table(r.json())

