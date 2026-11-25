"""Experimental player opponent adjustment; no additional API calls.
The baseline is v1.2 smoothed appearance-level frequency. An offset logistic
model uses prior opponent goals allowed relative to prior league average.
Historical rows are matched by date and the player's actual team, not roster.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from players import before, probabilities

MARKETS={'points':'Bod ANO','goals':'Gól ANO'}
BETAS=(0.,.25,.5,1.,1.5,2.)

def team_context(history):
    d=history[['date','home','away','hg','ag']].copy()
    d['date']=pd.to_datetime(d.date).dt.normalize()
    if d[['date','home','away','hg','ag']].isna().any().any(): raise ValueError('Neúplné týmové výsledky.')
    if d.duplicated(['date','home','away']).any(): raise ValueError('Duplicitní týmové výsledky.')
    long=pd.concat([d.rename(columns={'home':'team','away':'opp','ag':'ga'})[['date','team','opp','ga']],d.rename(columns={'away':'team','home':'opp','hg':'ga'})[['date','team','opp','ga']]],ignore_index=True)
    if long.duplicated(['date','team']).any(): raise ValueError('Více zápasů týmu ve stejný den; nelze bezpečně spojit historii.')
    long=long.sort_values('date').reset_index(drop=True)
    return long

class Context:
    def __init__(self,history):
        self.long=team_context(history)
        self.by_team={t:g for t,g in self.long.groupby('team')}
        self.match={(r.date,r.team):r.opp for r in self.long.itertuples()}
        self.cache={}
    def index(self,day,opp):
        day=pd.Timestamp(day).normalize(); key=(day,opp)
        if key in self.cache: return self.cache[key]
        team=self.by_team.get(opp)
        result=None
        if team is not None:
            start=day-pd.Timedelta(days=365)
            hist=team[(team.date<day)&(team.date>=start)].tail(30)
            league=self.long[(self.long.date<day)&(self.long.date>=day-pd.Timedelta(days=365))]
            if len(hist)>=10 and len(league)>=200:
                age=(day-hist.date.max()).days
                if age<=200:
                    mean=float(league.ga.mean())
                    if mean>0:
                        rate=(float(hist.ga.sum())+10*mean)/(len(hist)+10)
                        result=(float(np.log(rate/mean)),rate,mean,len(hist),age)
        self.cache[key]=result
        return result

def adjusted(p,x,beta):
    return float(expit(logit(np.clip(p,1e-9,1-1e-9))+beta*x))

def historical(ctx, logs_by_id, day):
    cutoff=pd.Timestamp(day).normalize(); rows=[]
    for pid,item in logs_by_id.items():
        data=item[0] if isinstance(item,tuple) else item
        past=data[data.date<cutoff].sort_values(['date','game'])
        for g in past.itertuples():
            hist=before(past,g.date)
            if len(hist)<20: continue
            opp=ctx.match.get((g.date,g.team))
            if not opp: continue
            info=ctx.index(g.date,opp)
            if info is None: continue
            p=probabilities(hist)
            for target in MARKETS:
                rows.append(dict(date=g.date,game=g.game,player_id=pid,team=g.team,opponent=opp,target=target,p_base=p[target][0],x=info[0],actual=int(getattr(g,target)>0)))
    return pd.DataFrame(rows)

def score(p,y):
    p=np.clip(np.asarray(p),1e-9,1-1e-9); y=np.asarray(y)
    return float(np.mean((p-y)**2)),float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))

def calibrate(records):
    if records.empty: raise ValueError('Chybí spárované historické zápasy hráčů a týmů.')
    days=sorted(records.date.unique()); split=int(.7*len(days))
    if split<30 or len(days)-split<15: raise ValueError('Pro test je potřeba alespoň 30 trénovacích a 15 testovacích hracích dnů.')
    cutoff=days[split]; train=records[records.date<cutoff]; test=records[records.date>=cutoff]
    if len(train)<400 or len(test)<200: raise ValueError('Málo spárovaných hráčských záznamů pro nezávislý test.')
    summary=[]; details=[]; params={}
    for target in MARKETS:
        t=train[train.target==target]; v=test[test.target==target].copy()
        if t.actual.nunique()!=2 or v.empty: raise ValueError('Nedostatek různých výsledků pro test.')
        def pred(frame,b): return expit(logit(frame.p_base.clip(1e-9,1-1e-9))+b*frame.x)
        beta=min(BETAS,key=lambda b:score(pred(t,b),t.actual)[1])
        params[target]=beta
        v['p_opponent']=pred(v,beta); v['beta']=beta
        details.append(v)
        for label,col in [('v1.5 základ','p_base'),('v1.6 soupeř','p_opponent')]:
            brier,ll=score(v[col],v.actual)
            summary.append({'Predikce':MARKETS[target],'Model':label,'Trénink':len(t),'Test':len(v),'Brier':brier,'Log Loss':ll,'Beta':beta if col=='p_opponent' else 0.,'Test od':pd.Timestamp(cutoff).date().isoformat()})
    return params,pd.DataFrame(summary),pd.concat(details,ignore_index=True)

def analyze(history,logs_by_id,rows,day,home,away):
    ctx=Context(history); rec=historical(ctx,logs_by_id,day)
    params,summary,detail=calibrate(rec)
    out=rows.copy(); info_rows=[]
    for team,opp in [(home,away),(away,home)]:
        info=ctx.index(day,opp)
        if info is None: raise ValueError(f'Nedostatečná historie obrany: {opp}.')
        x,ga,league,n,age=info
        if age>21: raise ValueError(f'Týmová historie {opp} je stará {age} dní. Aktualizuj ji v Datovém centru.')
        info_rows.append({'Tým hráče':team,'Soupeř':opp,'GA/60 vyhlazené':ga,'Ligový průměr':league,'Zápasy soupeře':n,'Stáří (dny)':age})
        mask=out['Tým']==team
        for target,col in MARKETS.items():
            out.loc[mask,col+' v1.6']=out.loc[mask,col].map(lambda p:adjusted(p,x,params[target]))
    out['Bod NE v1.6']=1-out['Bod ANO v1.6']
    return out,summary,detail,pd.DataFrame(info_rows)
