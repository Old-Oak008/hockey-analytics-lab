"""Past-only team special teams and netminding proxy; paired validation."""
import numpy as np
import pandas as pd
from advanced import read_stats, features, fit_softmax

def read_extra(file):
    file.seek(0)
    cols=['gameId','gameDate','team','situation','iceTime','goalsAgainst','xGoalsAgainst','xGoalsFor','shotsOnGoalAgainst']
    d=pd.read_csv(file,compression='zip' if file.name.lower().endswith('.zip') else None,usecols=lambda c:c in cols)
    if not set(cols)<=set(d): raise ValueError('Chybí sloupce: '+', '.join(set(cols)-set(d)))
    d=d[d.gameId.astype(str).str.fullmatch(r'\d{4}02\d{4}') & d.situation.isin(['all','5on4','4on5'])].copy()
    d['date']=pd.to_datetime(d.gameDate.astype(str),format='%Y%m%d',errors='coerce')
    d['team']=d.team.replace({'LA':'LAK','NJ':'NJD','SJ':'SJS','TB':'TBL','MON':'MTL'})
    for c in cols[4:]: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['date','team']+cols[4:]);
    if d.duplicated(['date','team','situation']).any(): raise ValueError('Duplicitní statistiky.')
    return d.sort_values('date')

def build(games,stats,extra):
    base=features(games,stats); rows=[]
    by={t:g for t,g in extra.groupby('team')}
    for g in base.itertuples(index=False,name=None):
        vals=[]
        for t in g[1:3]:
            if t not in by: break
            p=by[t]; p=p[p.date<g[0]]; all_=p[p.situation=='all'].tail(20)
            if len(all_)<10 or (g[0]-all_.date.iloc[-1]).days>14: break
            pp=p[(p.situation=='5on4') & p.date.isin(all_.date)]; pk=p[(p.situation=='4on5') & p.date.isin(all_.date)]
            if len(pp)<8 or len(pk)<8 or min(pp.iceTime.sum(),pk.iceTime.sum(),all_.shotsOnGoalAgainst.sum())<=0: break
            vals.extend([3600*pp.xGoalsFor.sum()/pp.iceTime.sum(),3600*pk.xGoalsAgainst.sum()/pk.iceTime.sum(),(all_.xGoalsAgainst-all_.goalsAgainst).mean(),1-all_.goalsAgainst.sum()/all_.shotsOnGoalAgainst.sum()])
        if len(vals)==8: rows.append([*g,*vals])
    return pd.DataFrame(rows,columns=list(base.columns)+[f'{s}_{c}' for s in ['h','a'] for c in ['pp_xg60','pk_xga60','team_xga_minus_ga','team_sv_proxy']])

def evaluate(games,stats,extra):
    d=build(games,stats,extra); out=[]
    if len(d)<500: raise ValueError('Potřebuji nejméně 500 společných zápasů.')
    blocks=(d.date-d.date.min()).dt.days//28
    for b,test in d.groupby(blocks):
        train=d[blocks<b]
        if len(train)<400: continue
        y=np.where(train.hg>train.ag,0,np.where(train.hg==train.ag,1,2)); actual=np.where(test.hg>test.ag,0,np.where(test.hg==test.ag,1,2)); bins=np.digitize(train.hg+train.ag,[5,6,7]); obs=np.digitize(test.hg+test.ag,[5,6,7])
        for name,end in [('v0.8 – společný vzorek',15),('v0.9 – speciální formace',23),('v0.9 – pouze speciální formace',None)]:
            if end is None:
                ix=list(range(5,15))+[15,16,19,20]; x=train.iloc[:,ix].to_numpy(float); v=test.iloc[:,ix].to_numpy(float)
            else: x=train.iloc[:,5:end].to_numpy(float); v=test.iloc[:,5:end].to_numpy(float)
            p=fit_softmax(x,y,3)(v); q=fit_softmax(x,bins,4)(v)
            for i,g in enumerate(test.itertuples()):
                out.append([g.date,g.home,g.away,'1X2',name,float(np.sum((p[i]-np.eye(3)[actual[i]])**2)),float(-np.log(max(p[i,actual[i]],1e-12))),p[i,0],p[i,1],p[i,2],actual[i]])
                for j,line in enumerate([4.5,5.5,6.5]):
                    prob=q[i,j+1:].sum(); event=int(obs[i]>j)
                    out.append([g.date,g.home,g.away,f'Over {line}',name,(prob-event)**2,-np.log(max(prob if event else 1-prob,1e-12)),prob,np.nan,np.nan,event])
    detail=pd.DataFrame(out,columns=['Datum','Domácí','Hosté','Trh','Model','Brier','Log Loss','P1 / Over','PX','P2','Skutečnost'])
    if detail.empty: raise ValueError('Nedostatek testovacích dat.')
    return detail.groupby(['Trh','Model']).agg(Zápasů=('Brier','size'),Brier=('Brier','mean'),**{'Log Loss':('Log Loss','mean')}).reset_index(),detail,len(d)
