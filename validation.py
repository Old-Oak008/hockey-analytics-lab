import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from model import rates

LINES=np.array([4.5,5.5,6.5])
# Predeclared small grid; fit only on earlier evaluation dates.
GRID=[(e,f,t) for e in (0.,.5,1.) for f in (0.,.5,1.) for t in (.9,1.,1.1)]
def evaluate(d, elo, recent_shift):
    records=[]
    for day,games in d.groupby('date',sort=True):
        past=d[d.date<day]; rating=elo(past)
        for g in games.itertuples():
            try: x,y=rates(past,g.home,g.away)
            except ValueError: continue
            e=(rating.get(g.home,1500)-rating.get(g.away,1500))/1200
            f=recent_shift(past,g.home,g.away)
            records.append((day,x,y,e,f,0 if g.hg>g.ag else 2 if g.hg<g.ag else 1,g.hg+g.ag, float(past.hg.mean()+past.ag.mean()),[(past.hg>past.ag).mean(),(past.hg==past.ag).mean(),(past.hg<past.ag).mean()]))
    if len(records)<60: raise ValueError('Pro oddělené ladění a test je třeba nejméně 60 vyhodnotitelných zápasů.')
    days=sorted(set(r[0] for r in records)); boundary=days[int(len(days)*.7)]
    train=[r for r in records if r[0]<boundary]; test=[r for r in records if r[0]>=boundary]
    if min(len(train),len(test))<20: raise ValueError('Příliš málo zápasů v oddělených blocích.')
    def arrays(rows,params):
        x,y,e,f=np.array([r[1:5] for r in rows]).T
        if params is None:
            p=np.array([r[8] for r in rows]); p=(p+0.001)/1.003
            q=poisson.sf(LINES[None,:],np.array([r[7] for r in rows])[:,None])
        else:
            ew,fw,t=params; shift=np.clip(ew*e+fw*f,-.35,.35)
            u,v=x*np.exp(shift),y*np.exp(-shift); total=(x+y)*t
            x,y=u*total/(u+v),v*total/(u+v)
            p=np.stack([skellam.sf(0,x,y),skellam.pmf(0,x,y),skellam.cdf(-1,x,y)],axis=1)
            q=poisson.sf(LINES[None,:],total[:,None])
        a=np.array([r[5] for r in rows]); b=np.array([r[6] for r in rows])[:,None]>LINES
        return p,q,a,b
    def losses(rows,params):
        p,q,a,b=arrays(rows,params); q=np.clip(q,1e-12,1-1e-12)
        ll=-np.log(np.clip(p[np.arange(len(a)),a],1e-12,1)).mean()
        ou=-(b*np.log(q)+(1-b)*np.log1p(-q)).mean()
        return ll,ou
    best=min(GRID,key=lambda z:sum(losses(train,z)))
    summary=[]; totals=[]; calibration=[]
    for name,params in [('Ligový průměr',None),('Poisson',(0,0,1)),('Elo + forma',(1,1,1)),('Laděný model',best)]:
        p,q,a,b=arrays(test,params); ll,_=losses(test,params)
        summary.append([name,len(a),np.mean(np.sum((p-np.eye(3)[a])**2,axis=1)),ll,np.mean(p.argmax(axis=1)==a)])
        for j,line in enumerate(LINES):
            z=np.clip(q[:,j],1e-12,1-1e-12); actual=b[:,j]
            totals.append([name,line,len(a),np.mean((z-actual)**2),-np.mean(actual*np.log(z)+(1-actual)*np.log1p(-z)),np.mean((z>=.5)==actual),z.mean(),actual.mean()])
            if name=='Laděný model':
                for low in np.arange(0,1,.2):
                    mask=(z>=low)&(z<low+.2+ (1e-9 if low>.79 else 0))
                    if mask.any(): calibration.append([line,f'{low:.0%}–{low+.2:.0%}',int(mask.sum()),z[mask].mean(),actual[mask].mean()])
    return best,boundary,len(train),pd.DataFrame(summary,columns=['Model','Zápasů','Brier','Log Loss','Přesnost']),pd.DataFrame(totals,columns=['Model','Over','Zápasů','Brier','Log Loss','Přesnost','Průměr predikce','Skutečný podíl']),pd.DataFrame(calibration,columns=['Over','Interval','Zápasů','Predikce','Skutečnost'])
