"""League-local, chronological evaluation using regulation scores only."""
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from model import validate
from dynamic import fit

LINES=np.array([4.5,5.5,6.5])
WEIGHTS=(0.,.25,.5,.75,1.)
MARKETS=('1X2','Over 4.5','Over 5.5','Over 6.5')

def clean(data,day):
    d=validate(data)
    d['date']=d.date.dt.normalize()
    d=d[d.date<pd.Timestamp(day).normalize()].drop_duplicates()
    if d.duplicated(['date','home','away']).any():
        raise ValueError('Rozporná skóre stejného zápasu.')
    return d.sort_values('date').reset_index(drop=True)

def components(train,pairs):
    if len(train)<50: raise ValueError('Potřebuji nejméně 50 starších zápasů.')
    rate,_=fit(train,half_life=180)
    x,y=np.array([rate(h,a) for h,a in pairs]).T
    p=np.column_stack([skellam.sf(0,x,y),skellam.pmf(0,x,y),skellam.cdf(-1,x,y)])
    q=poisson.sf(LINES[None,:],(x+y)[:,None])
    actual=np.where(train.hg>train.ag,0,np.where(train.hg==train.ag,1,2))
    bp=(np.bincount(actual,minlength=3)+1)/(len(train)+3)
    bq=((train.hg.to_numpy()[:,None]+train.ag.to_numpy()[:,None]>LINES).sum(axis=0)+1)/(len(train)+2)
    return bp,bq,p,q

def mixed(bp,bq,p,q,w1,wg):
    return (1-w1)*bp+w1*p,(1-wg)*bq+wg*q

def loss(p,q,actual,obs):
    p=np.clip(p,1e-12,1); q=np.clip(q,1e-12,1-1e-12)
    br=np.column_stack([((p-np.eye(3)[actual])**2).sum(axis=1),(q-obs)**2])
    ll=np.column_stack([-np.log(p[np.arange(len(actual)),actual]),-np.where(obs,np.log(q),np.log1p(-q))])
    return br,ll

def evaluate(data,day):
    d=clean(data,day)
    if len(d)<110: raise ValueError('Pro oddělené ladění a test potřebuji alespoň 110 zápasů před datem analýzy (50 pro učení a nejméně 30 + 30 pro hodnocení).')
    # Each block is forecast using only earlier days, including within a season.
    blocks=(d.date-d.date.min()).dt.days//14
    games=[]; preds={w:[] for w in WEIGHTS}
    for block,test in d.groupby(blocks,sort=True):
        train=d[blocks<block]
        if len(train)<50: continue
        known=set(train.home)|set(train.away)
        test=test[test.home.isin(known)&test.away.isin(known)]
        if test.empty: continue
        c=components(train,list(zip(test.home,test.away)))
        games.append(test)
        for w in WEIGHTS: preds[w].append(mixed(*c,w,w))
    if not games: raise ValueError('Nedostatek chronologické historie.')
    games=pd.concat(games,ignore_index=True)
    dates=games.date.drop_duplicates().tolist()
    candidates=[day for day in dates if 30<=int((games.date<day).sum())<=len(games)-30]
    if not candidates: raise ValueError('Potřebuji alespoň 30 ladicích a 30 testovacích zápasů na různých dnech. Přidej starší historii.')
    split=min(candidates,key=lambda dt:abs((games.date<dt).sum()-len(games)*.6))
    tuning=(games.date<split).to_numpy(); test=~tuning
    actual=np.where(games.hg>games.ag,0,np.where(games.hg==games.ag,1,2))
    obs=(games.hg.to_numpy()+games.ag.to_numpy())[:,None]>LINES
    pred={w:(np.concatenate([v[0] for v in chunks]),np.concatenate([v[1] for v in chunks])) for w,chunks in preds.items()}
    scores={w:loss(*v,actual,obs)[0][tuning].mean(axis=0) for w,v in pred.items()}
    w1=min(WEIGHTS,key=lambda w:scores[w][0]); wg=min(WEIGHTS,key=lambda w:scores[w][1:].mean())
    models={'Ligový průměr':pred[0.],'Regrese 180 dní':pred[1.],'Laděná kombinace':(pred[w1][0],pred[wg][1])}
    rows=[]; details=[]
    for name,(p,q) in models.items():
        br,ll=loss(p,q,actual,obs)
        for j,market in enumerate(MARKETS):
            rows.append([market,name,int(test.sum()),br[test,j].mean(),ll[test,j].mean()])
            frame=games.loc[test,['date','home','away','hg','ag']].copy()
            frame['trh']=market; frame['model']=name
            frame['p1_nebo_over']=p[test,0] if j==0 else q[test,j-1]
            frame['px']=p[test,1] if j==0 else np.nan; frame['p2']=p[test,2] if j==0 else np.nan
            frame['brier']=br[test,j]; frame['log_loss']=ll[test,j]
            details.append(frame)
    return pd.DataFrame(rows,columns=['Trh','Model','Zápasů','Brier','Log Loss']),pd.concat(details,ignore_index=True),{'w1':w1,'wg':wg,'tuning':int(tuning.sum()),'test':int(test.sum()),'split':str(split.date()),'n':len(d),'end':str(d.date.max().date())}

def forecast(data,day,home,away,config):
    d=clean(data,day)
    if home==away or any(((d.home==t)|(d.away==t)).sum()<5 for t in (home,away)):
        raise ValueError('Potřebuji alespoň pět dřívějších zápasů každého týmu.')
    if config['n']!=len(d) or config['end']!=str(d.date.max().date()): raise ValueError('Historie se změnila; spusť znovu test.')
    p,q=mixed(*components(d,[(home,away)]),config['w1'],config['wg'])
    return p[0],q[0]
