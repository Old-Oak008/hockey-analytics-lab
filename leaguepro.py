"""League-specific walk-forward comparison; regulation scores only."""
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import skellam, poisson
from model import rates
from dynamic import fit
from multileague import clean, loss, LINES, MARKETS

HALVES=(90,180,365)
WEIGHTS=(.25,.5,.75,1.)
BASE='Ligový průměr'
OLD='Základní Poisson'
TUNED='Laděný model ligy'
KEYS=[(0,0.)]+[(h,w) for h in HALVES for w in WEIGHTS]

def fingerprint(d):
    return hashlib.sha256(d.sort_values(['date','home','away']).to_csv(index=False).encode()).hexdigest()

def probs(x,y):
    x,y=np.asarray(x),np.asarray(y)
    return (np.column_stack([skellam.sf(0,x,y),skellam.pmf(0,x,y),skellam.cdf(-1,x,y)]),poisson.sf(LINES[None,:],(x+y)[:,None]))

def components(train,pairs):
    n=len(train)
    actual=np.where(train.hg>train.ag,0,np.where(train.hg==train.ag,1,2))
    bp=np.tile((np.bincount(actual,minlength=3)+1)/(n+3),(len(pairs),1))
    bq=np.tile(((train.hg.to_numpy()[:,None]+train.ag.to_numpy()[:,None]>LINES).sum(axis=0)+1)/(n+2),(len(pairs),1))
    result={(0,0.):(bp,bq)}
    for half in HALVES:
        rate,_=fit(train,half)
        x,y=np.asarray([rate(h,a) for h,a in pairs]).T
        p,q=probs(x,y)
        for w in WEIGHTS: result[half,w]=((1-w)*bp+w*p,(1-w)*bq+w*q)
    return result

def evaluate(data,day):
    d=clean(data,day)
    if len(d)<110: raise ValueError('Nejméně 110 zápasů; podle rozložení může být potřeba více.')
    blocks=(d.date-d.date.min()).dt.days//14
    frames=[]; old=[]; chunks={k:[] for k in KEYS}
    for block,games in d.groupby(blocks,sort=True):
        train=d[blocks<block]
        if len(train)<50: continue
        known=set(train.home)|set(train.away)
        selected=[]; xy=[]
        for g in games.itertuples():
            if g.home not in known or g.away not in known: continue
            try:
                # Match deployed basic model: all completed dates before game day.
                x,y=rates(d[d.date<g.date],g.home,g.away)
            except ValueError: continue
            selected.append(g.Index); xy.append((x,y))
        if not selected: continue
        test=d.loc[selected]
        frames.append(test); old.append(probs(*np.asarray(xy).T))
        c=components(train,list(zip(test.home,test.away)))
        for k in KEYS: chunks[k].append(c[k])
    if not frames: raise ValueError('Není dost zápasů se známou historií týmů.')
    games=pd.concat(frames,ignore_index=True)
    possible=[t for t in games.date.unique() if 30<=(games.date<t).sum()<=len(games)-30]
    if not possible: raise ValueError('Potřebuji nejméně 30 ladicích a 30 testovacích zápasů na různých dnech.')
    split=pd.Timestamp(min(possible,key=lambda t:abs((games.date<t).sum()-.6*len(games))))
    tune=(games.date<split).to_numpy(); test=~tune
    actual=np.where(games.hg>games.ag,0,np.where(games.hg==games.ag,1,2))
    obs=(games.hg.to_numpy()+games.ag.to_numpy())[:,None]>LINES
    def join(v): return tuple(np.concatenate([x[j] for x in v]) for j in (0,1))
    pred={k:join(v) for k,v in chunks.items()}
    scores={k:loss(*v,actual,obs)[0][tune].mean(axis=0) for k,v in pred.items()}
    k1=min(KEYS,key=lambda k:scores[k][0]); kg=min(KEYS,key=lambda k:scores[k][1:].mean())
    models={BASE:pred[(0,0.)],OLD:join(old),TUNED:(pred[k1][0],pred[kg][1])}
    rows=[]; details=[]
    for name,(p,q) in models.items():
        br,ll=loss(p,q,actual,obs)
        for j,market in enumerate(MARKETS):
            rows.append([market,name,int(test.sum()),br[test,j].mean(),ll[test,j].mean()])
            f=games.loc[test,['date','home','away','hg','ag']].copy()
            f['trh']=market; f['model']=name
            f['p1_nebo_over']=p[test,0] if j==0 else q[test,j-1]
            f['px']=p[test,1] if j==0 else np.nan; f['p2']=p[test,2] if j==0 else np.nan
            f['brier']=br[test,j]; f['log_loss']=ll[test,j]; details.append(f)
    config={'half1':k1[0],'w1':k1[1],'halfg':kg[0],'wg':kg[1],'tuning':int(tune.sum()),'test':int(test.sum()),'split':str(split.date()),'n':len(d),'end':str(d.date.max().date()),'fingerprint':fingerprint(d)}
    return pd.DataFrame(rows,columns=['Trh','Model','Zápasů','Brier','Log Loss']),pd.concat(details,ignore_index=True),config

def forecast(data,day,pairs,config):
    d=clean(data,day)
    if fingerprint(d)!=config['fingerprint']: raise ValueError('Historie se změnila; spusť znovu test.')
    if not pairs: raise ValueError('Chybí zápasy.')
    counts=pd.concat([d.home,d.away]).value_counts()
    if any(h==a or min(counts.get(h,0),counts.get(a,0))<5 for h,a in pairs): raise ValueError('Potřebuji pět dřívějších zápasů každého týmu.')
    c=components(d,pairs)
    return c[config['half1'],config['w1']][0],c[config['halfg'],config['wg']][1]
