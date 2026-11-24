import math
import pandas as pd
from scipy.stats import poisson, skellam
COLS=['date','home','away','hg','ag']
def validate(df):
    d=df[COLS].copy(); d['date']=pd.to_datetime(d.date,errors='raise');
    for c in ['hg','ag']:
        d[c]=pd.to_numeric(d[c],errors='raise')
        if d[c].isna().any() or not ((d[c]>=0)&(d[c]<=50)&(d[c]%1==0)).all(): raise ValueError('Neplatné skóre')
    for c in ['home','away']:
        if d[c].isna().any(): raise ValueError('Chybí tým')
        d[c]=d[c].astype(str).str.strip()
    if (d.home==d.away).any() or (d.home=='').any() or (d.away=='').any() or d.date.isna().any(): raise ValueError('Neplatný zápas')
    return d.drop_duplicates().sort_values('date')
def rates(d,h,a):
    if h==a or len(d)<10: raise ValueError('Potřebuji dva různé týmy a alespoň 10 zápasů.')
    H=max(.1,d.hg.mean()); A=max(.1,d.ag.mean()); prior=8
    x=d[d.home==h]; y=d[d.away==a]
    if len(x)<3 or len(y)<3: raise ValueError('Potřebuji alespoň 3 domácí a 3 venkovní zápasy příslušných týmů.')
    avg=lambda s,b:(s.sum()+prior*b)/(len(s)+prior)
    return avg(x.hg,H)*avg(y.hg,H)/H,avg(y.ag,A)*avg(x.ag,A)/A

def predict(h,a):
    if not all(math.isfinite(x) and x>0 for x in (h,a)): raise ValueError('Neplatné očekávané góly')
    p=[float(skellam.sf(0,h,a)),float(skellam.pmf(0,h,a)),float(skellam.cdf(-1,h,a))]
    totals=[(k,float(poisson.sf(k,h+a))) for k in (4.5,5.5,6.5)]
    scores=sorted([(i,j,float(poisson.pmf(i,h)*poisson.pmf(j,a))) for i in range(16) for j in range(16)],key=lambda x:x[2],reverse=True)[:5]
    return p,totals,scores

def backtest(d):
    rows=[]
    for date,games in d.groupby('date'):
        past=d[d.date<date]
        for g in games.itertuples():
            try: p,_,_=predict(*rates(past,g.home,g.away))
            except ValueError: continue
            actual=0 if g.hg>g.ag else 2 if g.hg<g.ag else 1
            rows.append((sum((v-(i==actual))**2 for i,v in enumerate(p)),-math.log(max(p[actual],1e-15)),int(max(range(3),key=lambda i:p[i])==actual)))
    if not rows: raise ValueError('Nedostatek historie pro test.')
    return len(rows),*[sum(r[i] for r in rows)/len(rows) for i in range(3)]
