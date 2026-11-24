"""Live NHL forecasts using the v0.7 and v0.8 development protocols.
The model families and hyperparameter search are fixed; only past data update.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from dynamic import fit, merge_history, LINES
from advanced import features, fit_softmax

HALVES=(30,60,90,180,365)
BLENDS=(0.,.5,1.)
KEYS=[(h,b) for h in HALVES for b in BLENDS]

def _base(train):
    y=np.where(train.hg>train.ag,0,np.where(train.hg==train.ag,1,2))
    return ((np.bincount(y,minlength=3)+1)/(len(train)+3),
            (((train.hg+train.ag).to_numpy()[:,None]>LINES).sum(0)+1)/(len(train)+2))

def _probs(train, test, half):
    rate,_=fit(train,half)
    x,y=np.array([rate(g.home,g.away) for g in test.itertuples()]).T
    return (np.column_stack([skellam.sf(0,x,y),skellam.pmf(0,x,y),skellam.cdf(-1,x,y)]),
            poisson.sf(LINES[None,:],(x+y)[:,None]))

def _targets(day,pairs):
    return pd.DataFrame([(day,h,a,0,0) for h,a in pairs],columns=['date','home','away','hg','ag'])

def _history(games,day):
    d=merge_history([games]); d=d[d.date<day].sort_values('date')
    if len(d)<400: raise ValueError('Pro model PRO potřebuji historii více sezon (nejméně 400 zápasů).')
    if (day-d.date.max()).days>14: raise ValueError('Výsledky jsou starší než 14 dní; aktualizuj sezonu nebo datum.')
    return d

def dynamic_forecast(games,day,pairs):
    day=pd.Timestamp(day).normalize(); d=_history(games,day)
    origin=d.date.min(); blocks=(d.date-origin).dt.days//14; target_block=(day-origin).days//14
    losses={k:np.zeros(4) for k in KEYS}; seen=0
    # Same block start and scoring rule as dynamic.dynamic_test. Current partial block is excluded.
    for block,test in d.groupby(blocks):
        if block>=target_block: break
        train=d[blocks<block]
        if len(train)<200: continue
        base,bq=_base(train)
        actual=np.where(test.hg>test.ag,0,np.where(test.hg==test.ag,1,2)); hot=np.eye(3)[actual]
        obs=(test.hg+test.ag).to_numpy()[:,None]>LINES
        for half in HALVES:
            p,q=_probs(train,test,half)
            for blend in BLENDS:
                pp=(1-blend)*base+blend*p; qq=(1-blend)*bq+blend*q
                losses[half,blend]+=np.r_[np.sum((pp-hot)**2),np.sum((qq-obs)**2,axis=0)]
        seen+=len(test)
    train=d[blocks<target_block]
    if seen<200 or len(train)<200: raise ValueError('Nedostatek uzavřených validačních bloků pro v0.7.')
    k1=min(KEYS,key=lambda k:losses[k][0]); kg=min(KEYS,key=lambda k:losses[k][1:].mean())
    base,bq=_base(train); target=_targets(day,pairs)
    pp,_=_probs(train,target,k1[0]); _,qq=_probs(train,target,kg[0]);
    pp=(1-k1[1])*base+k1[1]*pp; qq=(1-kg[1])*bq+kg[1]*qq
    return {'p':pp,'q':qq,'k1':k1,'kg':kg,'n':len(train),'seen':seen,'train_end':str(train.date.max().date())}

def advanced_forecast(games,stats,day,pairs):
    day=pd.Timestamp(day).normalize(); d=_history(games,day)
    # Mirror the v0.8 feature construction and 28-day block anchor.
    hist=features(d,stats[stats.date<day]); target=features(_targets(day,pairs),stats[stats.date<day]);
    if hist.empty: raise ValueError('MoneyPuck nemá historii odpovídajících týmů.')
    origin=hist.date.min(); block=(day-origin).days//28
    train=hist[(hist.date-origin).dt.days//28<block]
    if len(train)<400: raise ValueError('Pro v0.8 potřebuji alespoň 400 zápasů s historickými xG.')
    if target.empty: raise ValueError('MoneyPuck nemá dostatečně aktuální statistiky vybraných týmů.')
    x=train.iloc[:,5:].to_numpy(float); v=target.iloc[:,5:].to_numpy(float)
    y=np.where(train.hg>train.ag,0,np.where(train.hg==train.ag,1,2)); p=fit_softmax(x,y,3)(v)
    return {'p':{(g.home,g.away):p[i] for i,g in enumerate(target.itertuples())},'n':len(train),'train_end':str(train.date.max().date()),'stats_end':str(stats[stats.date<day].date.max().date())}

def score_archive(records,games,today):
    """Evaluate externally saved pregame snapshots; keep first per game/model.
    The CSV is user-controlled, not an auditable/tamper-proof log.
    """
    cols=['created_utc','date','home','away','status','model_1x2','model_goals','p1','px','p2','over_4.5','over_5.5','over_6.5']
    if not set(cols)<=set(records): raise ValueError('CSV nemá formát exportu v1.0.')
    r=records[cols].copy(); r=r[r.status=='předzápasová (API ověřeno)'].copy()
    r['created_utc']=pd.to_datetime(r.created_utc,utc=True,errors='raise'); r['date']=pd.to_datetime(r.date,errors='raise').dt.normalize()
    probcols=cols[7:]; r[probcols]=r[probcols].apply(pd.to_numeric,errors='raise'); v=r[probcols].to_numpy(float)
    if not np.isfinite(v).all() or ((v<0)|(v>1)).any() or not np.allclose(v[:,:3].sum(1),1,atol=1e-6) or (np.diff(v[:,3:],axis=1)>1e-9).any(): raise ValueError('Neplatné pravděpodobnosti v archivu.')
    r=r.sort_values('created_utc').drop_duplicates(['date','home','away','model_1x2','model_goals'])
    d=merge_history([games]); d=d[d.date<pd.Timestamp(today).normalize()]
    result=r.merge(d,on=['date','home','away'],how='inner',validate='many_to_one'); out=[]
    for _,g in result.iterrows():
        p=g[['p1','px','p2']].to_numpy(float); y=0 if g.hg>g.ag else 1 if g.hg==g.ag else 2
        out.append([g.date,g.home,g.away,'1X2',g.model_1x2,float(np.sum((p-np.eye(3)[y])**2)),float(-np.log(max(p[y],1e-12)))])
        for line in LINES:
            q=float(g[f'over_{line}']); event=g.hg+g.ag>line
            out.append([g.date,g.home,g.away,f'Over {line}',g.model_goals,float((q-event)**2),float(-np.log(max(q if event else 1-q,1e-12)))])
    detail=pd.DataFrame(out,columns=['Datum','Domácí','Hosté','Trh','Model','Brier','Log Loss'])
    # One goal prediction per game and goal model, even if the 1X2 model changed.
    detail=detail.drop_duplicates(['Datum','Domácí','Hosté','Trh','Model'])
    summary=detail.groupby(['Trh','Model']).agg(Zápasů=('Brier','size'),Brier=('Brier','mean'),**{'Log Loss':('Log Loss','mean')}).reset_index()
    return summary,detail,len(result),len(r)-len(result)
