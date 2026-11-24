"""MoneyPuck team-game data: past-only features and chronological validation."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

COLS=['xGoalsFor','xGoalsAgainst','shotsOnGoalFor','shotsOnGoalAgainst']

def read_stats(file):
    d=pd.read_csv(file,compression='zip' if getattr(file,'name','').lower().endswith('.zip') else None, usecols=lambda c:c in ['gameId','gameDate','team','situation']+COLS)
    if not set(['gameId','gameDate','team','situation']+COLS)<=set(d): raise ValueError('Chybí sloupce týmových dat MoneyPuck.')
    d=d[d.situation.eq('all')].copy()
    # NHL regular season identifier YYYY02xxxx
    d=d[d.gameId.astype(str).str.fullmatch(r'\d{4}02\d{4}')].copy()
    d['date']=pd.to_datetime(d.gameDate.astype(str),format='%Y%m%d',errors='coerce')
    d['team']=d.team.replace({'LA':'LAK','NJ':'NJD','SJ':'SJS','TB':'TBL','MON':'MTL'})
    for c in COLS: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['date','team']+COLS)
    if d.duplicated(['date','team']).any(): raise ValueError('Duplicitní týmové záznamy.')
    if d.empty: raise ValueError('Žádná použitelná data základní části.')
    return d.sort_values('date')

def features(games, stats):
    # Every feature uses strictly earlier calendar days, never the game's own statistics.
    by={t:g for t,g in stats.groupby('team')}; rows=[]
    for g in games.sort_values('date').itertuples():
        vals=[]
        for t in (g.home,g.away):
            if t not in by: break
            past=by[t]; past=past[past.date<g.date].tail(15)
            if len(past)<10 or (g.date-past.date.iloc[-1]).days>14: break
            vals.extend(past[COLS].mean().tolist()); vals.append(min(7,(g.date-past.date.iloc[-1]).days))
        if len(vals)==10: rows.append([g.date,g.home,g.away,g.hg,g.ag,*vals])
    return pd.DataFrame(rows,columns=['date','home','away','hg','ag']+[f'{side}_{c}' for side in ['h','a'] for c in COLS+['rest']])

def fit_softmax(x,y,k):
    mean=x.mean(0); std=x.std(0); std[std<1e-8]=1
    x=np.column_stack([np.ones(len(x)),(x-mean)/std]); hot=np.eye(k)[y]; p=x.shape[1]
    def loss(b):
        b=b.reshape(p,k); z=x@b; lp=z-logsumexp(z,axis=1,keepdims=True); grad=x.T@(np.exp(lp)-hot); grad[1:]+=20*b[1:]
        return -np.sum(hot*lp)+10*np.sum(b[1:]**2),grad.ravel()
    r=minimize(loss,np.zeros(p*k),jac=True,method='L-BFGS-B');
    if not r.success: raise ValueError('Model nekonverguje.')
    def predict(v):
        z=np.column_stack([np.ones(len(v)),(v-mean)/std])@r.x.reshape(p,k); return np.exp(z-logsumexp(z,axis=1,keepdims=True))
    return predict

def evaluate(games,stats):
    d=features(games,stats); out=[]
    if len(d)<500: raise ValueError('Potřebuji alespoň 500 zápasů s dostupnou historií statistik.')
    blocks=(d.date-d.date.min()).dt.days//28
    for block,test in d.groupby(blocks):
        train=d[blocks<block]
        if len(train)<400: continue
        x=train.iloc[:,5:].to_numpy(float); v=test.iloc[:,5:].to_numpy(float)
        y=np.where(train.hg>train.ag,0,np.where(train.hg==train.ag,1,2)); actual=np.where(test.hg>test.ag,0,np.where(test.hg==test.ag,1,2))
        bins=np.digitize(train.hg+train.ag,[5,6,7]); obs=np.digitize(test.hg+test.ag,[5,6,7]);
        p=fit_softmax(x,y,3)(v); q=fit_softmax(x,bins,4)(v)
        base=(np.bincount(y,minlength=3)+1)/(len(y)+3); bq=(np.bincount(bins,minlength=4)+1)/(len(bins)+4)
        for name,pp,qq in [('Ligový průměr',np.tile(base,(len(test),1)),np.tile(bq,(len(test),1))),('xG + střely + odpočinek',p,q)]:
            for i,g in enumerate(test.itertuples()):
                out.append([g.date,g.home,g.away,'1X2',name,np.sum((pp[i]-np.eye(3)[actual[i]])**2),-np.log(max(pp[i,actual[i]],1e-12))])
                for j,line in enumerate([4.5,5.5,6.5]):
                    prob=qq[i,j+1:].sum(); event=obs[i]>j
                    out.append([g.date,g.home,g.away,f'Over {line}',name,(prob-event)**2,-np.log(max(prob if event else 1-prob,1e-12))])
    detail=pd.DataFrame(out,columns=['Datum','Domácí','Hosté','Trh','Model','Brier','Log Loss'])
    if detail.empty: raise ValueError('Nedostatek testovacích zápasů.')
    return detail.groupby(['Trh','Model']).agg(Zápasů=('Brier','size'),Brier=('Brier','mean'),**{'Log Loss':('Log Loss','mean')}).reset_index(),detail,len(d)
