"""Historical season validation. No future observations used in fitting."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson, skellam, nbinom
from model import validate

LINES=np.array([4.5,5.5,6.5])
def season_year(d):
    return d.dt.year-(d.dt.month<7).astype(int)

def merge_history(frames):
    d=validate(pd.concat(frames,ignore_index=True))
    if d.duplicated(['date','home','away']).any(): raise ValueError('Rozporné výsledky stejného zápasu.')
    return d

def fit(train, half_life=365):
    teams=sorted(set(train.home)|set(train.away)); idx={t:i for i,t in enumerate(teams)}; n=len(teams)
    h=train.home.map(idx).to_numpy(); a=train.away.map(idx).to_numpy()
    x=train.hg.to_numpy(); y=train.ag.to_numpy()
    age=(train.date.max()-train.date).dt.days.to_numpy(); w=2**(-age/half_life)
    def objective(b):
        u=b[0]+b[2+h]+b[2+n+a]; v=b[1]+b[2+a]+b[2+n+h]
        m=np.exp(u); k=np.exp(v); r=w*(m-x); s=w*(k-y)
        loss=np.sum(w*(m-x*u+k-y*v))+4*np.sum(b[2:]**2)
        grad=np.r_[r.sum(),s.sum(),np.bincount(h,r,n)+np.bincount(a,s,n),np.bincount(a,r,n)+np.bincount(h,s,n)]
        grad[2:]+=8*b[2:]; return loss,grad
    initial=np.r_[np.log(max(.1,x.mean())),np.log(max(.1,y.mean())),np.zeros(2*n)]
    res=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(-2,3)]*2+[(-1.5,1.5)]*(2*n))
    if not res.success: raise ValueError('Regrese nekonvergovala: '+str(res.message))
    b=res.x
    def rates(h,a):
        def effect(t,offset): return b[offset+idx[t]] if t in idx else 0.
        return np.exp(b[0]+effect(h,2)+effect(a,2+n)),np.exp(b[1]+effect(a,2)+effect(h,2+n))
    mu=np.exp(b[0]+b[2+h]+b[2+n+a])+np.exp(b[1]+b[2+a]+b[2+n+h])
    alpha=max(0.,float(np.sum(w*((x+y-mu)**2-mu))/np.sum(w*mu**2)))
    return rates,alpha

def dynamic_test(data):
    """Refit every 14 days; select parameters using only earlier forecast blocks."""
    d=merge_history([data]).sort_values('date'); days=(d.date-d.date.min()).dt.days//14
    halves=(30,60,90,180,365); blends=(0.,.5,1.)
    keys=[(h,b) for h in halves for b in blends]
    losses={k:np.zeros(4) for k in keys}; seen=0; output=[]
    for block,test in d.groupby(days):
        train=d[days<block]
        if len(train)<200: continue
        n=len(test); actual=np.where(test.hg>test.ag,0,np.where(test.hg==test.ag,1,2)); goals=(test.hg+test.ag).to_numpy()
        hot=np.eye(3)[actual]; obs=goals[:,None]>LINES
        base=(np.bincount(np.where(train.hg>train.ag,0,np.where(train.hg==train.ag,1,2)),minlength=3)+1)/(len(train)+3)
        base_ou=((train.hg.to_numpy()[:,None]+train.ag.to_numpy()[:,None]>LINES).sum(axis=0)+1)/(len(train)+2)
        def pack(p,q):
            return np.column_stack([np.sum((p-hot)**2,axis=1),(q-obs)**2])
        def logloss(p,q):
            q=np.clip(q,1e-12,1-1e-12)
            return np.column_stack([-np.log(np.clip(p[np.arange(n),actual],1e-12,1)),-np.where(obs,np.log(q),np.log1p(-q))])
        predictions={}
        for half in halves:
            rate,_=fit(train,half)
            x,y=np.array([rate(g.home,g.away) for g in test.itertuples()]).T
            p=np.column_stack([skellam.sf(0,x,y),skellam.pmf(0,x,y),skellam.cdf(-1,x,y)])
            q=poisson.sf(LINES[None,:],(x+y)[:,None])
            for b in blends: predictions[(half,b)]=((1-b)*base+b*p,(1-b)*base_ou+b*q)
        # Select separately for 1X2 and entire coherent goal distribution using past Brier.
        if seen>=200:
            k1=min(keys,key=lambda k:losses[k][0]); kg=min(keys,key=lambda k:losses[k][1:].mean())
            for name,p,q in [('Ligový průměr',np.tile(base,(n,1)),np.tile(base_ou,(n,1))),('Dynamická regrese 365',*predictions[(365,1.)]),('Adaptivní kombinace',predictions[k1][0],predictions[kg][1])]:
                br=pack(p,q); ll=logloss(p,q)
                for i,g in enumerate(test.itertuples()):
                    for j,market in enumerate(['1X2','Over 4.5','Over 5.5','Over 6.5']):
                        probs=p[i] if j==0 else [q[i,j-1]]
                        output.append([g.date,g.home,g.away,g.hg,g.ag,market,name,br[i,j],ll[i,j],*([*probs] if j==0 else [probs[0],np.nan,np.nan]),str(k1 if j==0 else kg) if name=='Adaptivní kombinace' else ''])
        for k,(p,q) in predictions.items(): losses[k]+=pack(p,q).sum(axis=0)
        seen+=n
    if not output: raise ValueError('Potřebuji nejméně 400 zápasů a další testovací období.')
    detail=pd.DataFrame(output,columns=['Datum','Domácí','Hosté','HG','AG','Trh','Model','Brier','Log Loss','P1 / Over','PX','P2','Parametry'])
    summary=detail.groupby(['Trh','Model']).agg(Zápasů=('Brier','size'),Brier=('Brier','mean'),**{'Log Loss':('Log Loss','mean')}).reset_index()
    return summary,detail
