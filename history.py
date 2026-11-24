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

def fit(train):
    teams=sorted(set(train.home)|set(train.away)); idx={t:i for i,t in enumerate(teams)}; n=len(teams)
    h=train.home.map(idx).to_numpy(); a=train.away.map(idx).to_numpy()
    x=train.hg.to_numpy(); y=train.ag.to_numpy()
    age=(train.date.max()-train.date).dt.days.to_numpy(); w=2**(-age/365.)
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

def season_test(d):
    d=merge_history([d]); years=season_year(d.date); rows=[]
    for yr in sorted(years.unique())[1:]:
        train=d[years<yr]; test=d[years==yr]
        if len(train)<100 or len(test)<30: continue
        # Frozen parameters before season starts; same test rows for every model.
        rate,alpha=fit(train)
        x,y=np.array([rate(g.home,g.away) for g in test.itertuples()]).T
        total=x+y; actual=np.where(test.hg>test.ag,0,np.where(test.hg==test.ag,1,2)); goals=(test.hg+test.ag).to_numpy()
        base=np.array([(train.hg>train.ag).mean(),(train.hg==train.ag).mean(),(train.hg<train.ag).mean()]); base=(base+.001)/1.003
        p=np.stack([skellam.sf(0,x,y),skellam.pmf(0,x,y),skellam.cdf(-1,x,y)],axis=1)
        for name,probs in [('Ligový průměr',np.tile(base,(len(test),1))),('Poisson regrese',p)]:
            rows.append([yr,'1X2',name,len(test),np.mean(np.sum((probs-np.eye(3)[actual])**2,axis=1)),-np.log(np.clip(probs[np.arange(len(test)),actual],1e-12,1)).mean()])
        for k in LINES:
            obs=goals>k; q=poisson.sf(k,total)
            nb=nbinom.sf(k,1/alpha,1/(1+alpha*total)) if alpha>1e-8 else q
            for name,z in [('Ligový průměr',np.full(len(test),((train.hg+train.ag>k).sum()+1)/(len(train)+2))),('Poisson regrese',q),('Neg. binomické',nb)]:
                z=np.clip(z,1e-12,1-1e-12); rows.append([yr,f'Over {k}',name,len(test),np.mean((z-obs)**2),-np.mean(obs*np.log(z)+(1-obs)*np.log1p(-z))])
    if not rows: raise ValueError('Potřebuji alespoň dvě sezony, 100 starších a 30 novějších zápasů.')
    return pd.DataFrame(rows,columns=['Sezona od','Trh','Model','Zápasů','Brier','Log Loss'])
