"""Speed-only alternative to hybrid.py; same training protocols and probabilities."""
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from hybrid import _history, _base, _probs, _targets, HALVES, BLENDS, KEYS, LINES
from advanced import COLS, fit_softmax
from dynamic import merge_history


def closed_history(games, day):
    day=pd.Timestamp(day).normalize()
    d=_history(games,day)
    origin=d.date.min()
    boundary=origin+pd.Timedelta(days=14*((day-origin).days//14))
    return d[d.date<boundary].sort_values(['date','home','away']).reset_index(drop=True), boundary


def dynamic_closed(train, boundary, teams):
    """No observations from the active block. Cache by closed history, not matchup."""
    train=merge_history([train]).sort_values('date')
    if len(train)<200: raise ValueError('Nedostatek uzavřených validačních bloků pro v0.7.')
    blocks=(train.date-train.date.min()).dt.days//14
    losses={k:np.zeros(4) for k in KEYS}; seen=0
    for block,test in train.groupby(blocks):
        prev=train[blocks<block]
        if len(prev)<200: continue
        base,bq=_base(prev)
        actual=np.where(test.hg>test.ag,0,np.where(test.hg==test.ag,1,2)); hot=np.eye(3)[actual]
        obs=(test.hg+test.ag).to_numpy()[:,None]>LINES
        for half in HALVES:
            p,q=_probs(prev,test,half)
            for blend in BLENDS:
                pp=(1-blend)*base+blend*p; qq=(1-blend)*bq+blend*q
                losses[half,blend]+=np.r_[np.sum((pp-hot)**2),np.sum((qq-obs)**2,axis=0)]
        seen+=len(test)
    if seen<200: raise ValueError('Nedostatek uzavřených validačních bloků pro v0.7.')
    k1=min(KEYS,key=lambda k:losses[k][0]); kg=min(KEYS,key=lambda k:losses[k][1:].mean())
    base,bq=_base(train)
    pairs=tuple((h,a) for h in teams for a in teams if h!=a)
    target=_targets(pd.Timestamp(boundary),pairs)
    pp,_=_probs(train,target,k1[0]); _,qq=_probs(train,target,kg[0])
    pp=(1-k1[1])*base+k1[1]*pp; qq=(1-kg[1])*bq+kg[1]*qq
    return {'p':pp,'q':qq,'pairs':pairs,'k1':k1,'kg':kg,'n':len(train),'seen':seen,'train_end':str(train.date.max().date())}


def fast_features(games,stats):
    """Same trailing 15 observations (min 10), strict prior dates, max age 14."""
    ordered=games.sort_values('date')
    dates=ordered.date.to_numpy(dtype='datetime64[D]')
    out=np.full((len(ordered),10),np.nan)
    # Input stats are sorted by date by read_stats, as in the original model.
    for t,g in stats.groupby('team'):
        gd=g.date.to_numpy(dtype='datetime64[D]')
        if len(gd)==0: continue
        if (gd[1:]<gd[:-1]).any(): raise ValueError('Statistiky nejsou chronologicky seřazeny.')
        rolling=g[COLS].rolling(15,min_periods=10).mean().to_numpy(float)
        for side,offset in [('home',0),('away',5)]:
            loc=np.flatnonzero(ordered[side].to_numpy()==t)
            if not len(loc): continue
            idx=np.searchsorted(gd,dates[loc],side='left')-1
            has=idx>=9
            loc,idx=loc[has],idx[has]
            age=(dates[loc]-gd[idx]).astype('timedelta64[D]').astype(int)
            good=(age<=14)&(age>0)
            loc,idx,age=loc[good],idx[good],age[good]
            out[loc,offset:offset+4]=rolling[idx]; out[loc,offset+4]=np.minimum(7,age)
    ok=np.isfinite(out).all(axis=1)
    result=ordered.loc[:,['date','home','away','hg','ag']].iloc[np.flatnonzero(ok)].reset_index(drop=True)
    for i,c in enumerate([f'{side}_{c}' for side in ['h','a'] for c in COLS+['rest']]):
        result[c]=out[ok,i]
    return result


def advanced_fast(games,stats,day,pairs):
    day=pd.Timestamp(day).normalize(); d=_history(games,day)
    earlier=stats[stats.date<day]
    hist=fast_features(d,earlier); target=fast_features(_targets(day,pairs),earlier)
    if hist.empty: raise ValueError('MoneyPuck nemá historii odpovídajících týmů.')
    origin=hist.date.min(); block=(day-origin).days//28
    train=hist[(hist.date-origin).dt.days//28<block]
    if len(train)<400: raise ValueError('Pro v0.8 potřebuji alespoň 400 zápasů s historickými xG.')
    if target.empty: raise ValueError('MoneyPuck nemá dostatečně aktuální statistiky vybraných týmů.')
    x=train.iloc[:,5:].to_numpy(float); v=target.iloc[:,5:].to_numpy(float)
    y=np.where(train.hg>train.ag,0,np.where(train.hg==train.ag,1,2)); p=fit_softmax(x,y,3)(v)
    return {'p':{(g.home,g.away):p[i] for i,g in enumerate(target.itertuples())},'n':len(train),'train_end':str(train.date.max().date()),'stats_end':str(earlier.date.max().date())}
