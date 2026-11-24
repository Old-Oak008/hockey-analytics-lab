"""MoneyPuck individual goalie features. Historical starter projection is strictly past-only."""
import io
import zipfile
import numpy as np
import pandas as pd
from advanced import features, fit_softmax

ALIASES = {'id':('playerId','playerID','goalieId'), 'name':('name','playerName'),
 'game':('gameId','gameID'), 'date':('gameDate','date'), 'team':('playerTeam','team'),
 'situation':('situation',), 'seconds':('icetime','iceTime'), 'shots':('ongoal','onGoal','shotsAgainst','shotsOnGoalAgainst'),
 'ga':('goals','goalsAgainst'), 'xga':('xGoals','xGoalsAgainst')}
EXTRA=['gsax60','sv','gsax_recent','rest','starts14']

def read_goalies(files):
    frames=[]
    def one(f):
        keys={a for vals in ALIASES.values() for a in vals}
        d=pd.read_csv(f,usecols=lambda c:c in keys,low_memory=False)
        mapping={}
        for dest,opts in ALIASES.items():
            present=next((c for c in opts if c in d),None)
            if present is None: raise ValueError(f'V brankářském CSV chybí {dest} ({" / ".join(opts)}).')
            mapping[present]=dest
        d=d.rename(columns=mapping)[list(ALIASES)]
        d=d[d.situation.eq('all') & d.game.astype(str).str.fullmatch(r'\d{4}02\d{4}')].copy()
        d['date']=pd.to_datetime(d.date.astype(str),format='%Y%m%d',errors='coerce').fillna(pd.to_datetime(d.date.astype(str),format='%Y-%m-%d',errors='coerce'))
        d['team']=d.team.replace({'LA':'LAK','NJ':'NJD','SJ':'SJS','TB':'TBL','MON':'MTL'})
        for c in ['seconds','shots','ga','xga','id']: d[c]=pd.to_numeric(d[c],errors='coerce')
        d=d.dropna(subset=['id','name','date','team','seconds','shots','ga','xga'])
        d=d[(d.seconds>0)&(d.seconds<=4200)&(d.shots>=d.ga)&(d.ga>=0)&(d.xga>=0)]
        d['id']=d.id.astype('int64'); frames.append(d)
    for f in files:
        f.seek(0)
        if f.name.lower().endswith('.zip'):
            with zipfile.ZipFile(f) as z:
                entries=[i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith('.csv') and not i.filename.startswith('__MACOSX/')]
                if not entries or sum(i.file_size for i in entries)>350_000_000: raise ValueError('ZIP bez CSV nebo příliš velký (max. 350 MB po rozbalení).')
                for entry in entries:
                    with z.open(entry) as stream: one(stream)
        else: one(f)
    if not frames: raise ValueError('Nahraj alespoň jeden brankářský soubor.')
    d=pd.concat(frames,ignore_index=True).drop_duplicates()
    if d.empty: raise ValueError('Žádné brankářské záznamy základní části.')
    if d.duplicated(['id','game']).any(): raise ValueError('Konfliktní duplicity brankářů a zápasů. Zkontroluj překryv souborů.')
    return d.sort_values(['date','game','id']).reset_index(drop=True)

class GoalieBook:
    def __init__(self,d):
        self.players={int(i):g.reset_index(drop=True) for i,g in d.groupby('id')}
        self.teams={t:g.reset_index(drop=True) for t,g in d.groupby('team')}
    @staticmethod
    def before(g,day):
        return g.iloc[:g.date.searchsorted(day,side='left')]
    def candidates(self,team,day):
        day=pd.Timestamp(day).normalize(); g=self.teams.get(team)
        if g is None: return []
        p=self.before(g,day); p=p[p.date>=day-pd.Timedelta(days=45)]
        if p.empty or (day-p.date.iloc[-1]).days>14: return []
        last_games=p.game.drop_duplicates().tail(5); r=p[p.game.isin(last_games)]
        ranked=r.groupby('id').seconds.sum().sort_values(ascending=False,kind='stable').index
        # Include candidates with recent appearances but never use roster data from future dates.
        ids=list(ranked)+[i for i in p.id.unique() if i not in ranked]
        out=[]
        for i in ids:
            hist=self.before(self.players[int(i)],day)
            if hist.team.iloc[-1]==team and self.profile(int(i),day) is not None: out.append(int(i))
        return out
    def profile(self,i,day):
        day=pd.Timestamp(day).normalize(); p=self.before(self.players[int(i)],day)
        p=p[p.date>=day-pd.Timedelta(days=365)].tail(20)
        if len(p)<5 or p.shots.sum()<100 or p.seconds.sum()<10800: return None
        def gsax(g): return float((g.xga-g.ga).sum())
        def gs60(g,prior=0): return 3600*gsax(g)/(g.seconds.sum()+prior)
        def sv(g): return 1-g.ga.sum()/g.shots.sum() if g.shots.sum()>0 else np.nan
        q=p.tail(5); r=p.tail(10)
        # Shrink noisy small samples; fixed development choice, not optimized on test data.
        v=[gs60(p,18000), (p.shots.sum()-p.ga.sum()+90)/(p.shots.sum()+100), gs60(q,18000),min(30,(day-p.date.iloc[-1]).days),int((p.date>=day-pd.Timedelta(days=14)).sum())]
        return {'name':str(p.name.iloc[-1]),'id':int(i),'team':str(p.team.iloc[-1]),'last':str(p.date.iloc[-1].date()),'n':len(p),'sv':sv(p),'gsax':gsax(p),'gsax60':gs60(p),'sv5':sv(q),'sv10':sv(r),'gsax5':gsax(q),'gsax10':gsax(r),'rest':int((day-p.date.iloc[-1]).days),'games14':v[-1],'vector':v}
    def project(self,team,day):
        ids=self.candidates(team,day)
        return self.profile(ids[0],day) if ids else None

def build(games,stats,goalies):
    base=features(games,stats); book=GoalieBook(goalies); rows=[]
    for g in base.itertuples(index=False,name=None):
        h=book.project(g[1],g[0]); a=book.project(g[2],g[0])
        if h is not None and a is not None: rows.append([*g,*h['vector'],*a['vector']])
    return pd.DataFrame(rows,columns=list(base.columns)+[f'{s}_{c}' for s in ['h','a'] for c in EXTRA])

def _fit(train,values,extra):
    x=train.iloc[:,5:25 if extra else 15].to_numpy(float)
    y=np.where(train.hg>train.ag,0,np.where(train.hg==train.ag,1,2)); bins=np.digitize(train.hg+train.ag,[5,6,7])
    return fit_softmax(x,y,3)(values),fit_softmax(x,bins,4)(values)

def evaluate(games,stats,goalies):
    d=build(games,stats,goalies); rows=[]
    if len(d)<500: raise ValueError(f'Použitelných společných zápasů pouze {len(d)}; potřebuji alespoň 500. Přidej starší brankářské sezony.')
    blocks=(d.date-d.date.min()).dt.days//28
    for b,test in d.groupby(blocks):
        train=d[blocks<b]
        if len(train)<400: continue
        y=np.where(test.hg>test.ag,0,np.where(test.hg==test.ag,1,2)); bins=np.digitize(test.hg+test.ag,[5,6,7])
        for name,extra in [('v0.8 – společný vzorek',False),('v1.1 – odhad brankářů',True)]:
            p,q=_fit(train,test.iloc[:,5:25 if extra else 15].to_numpy(float),extra)
            for i,g in enumerate(test.itertuples()):
                rows.append([g.date,g.home,g.away,'1X2',name,float(np.sum((p[i]-np.eye(3)[y[i]])**2)),float(-np.log(max(p[i,y[i]],1e-12))),*p[i],y[i]])
                for j,line in enumerate([4.5,5.5,6.5]):
                    prob=float(q[i,j+1:].sum()); event=int(bins[i]>j)
                    rows.append([g.date,g.home,g.away,f'Over {line}',name,(prob-event)**2,-np.log(max(prob if event else 1-prob,1e-12)),prob,np.nan,np.nan,event])
    detail=pd.DataFrame(rows,columns=['Datum','Domácí','Hosté','Trh','Model','Brier','Log Loss','P1 / Over','PX','P2','Skutečnost'])
    if detail.empty: raise ValueError('Nedostatek testovacích bloků.')
    summary=detail.groupby(['Trh','Model']).agg(Zápasů=('Brier','size'),Brier=('Brier','mean'),**{'Log Loss':('Log Loss','mean')}).reset_index()
    return summary,detail,len(d)

def scenario(games,stats,goalies,day,home,away,hid,aid):
    day=pd.Timestamp(day).normalize(); d=build(games[games.date<day],stats[stats.date<day],goalies[goalies.date<day]); book=GoalieBook(goalies)
    if d.empty: raise ValueError('Chybí společná historie.')
    blocks=(d.date-d.date.min()).dt.days//28; train=d[blocks<(day-d.date.min()).days//28]
    if len(train)<400: raise ValueError('Potřebuji alespoň 400 zápasů v uzavřených trénovacích blocích.')
    target=features(pd.DataFrame([(day,home,away,0,0)],columns=['date','home','away','hg','ag']),stats[stats.date<day])
    if target.empty: raise ValueError('Chybí aktuální týmové statistiky.')
    if hid not in book.candidates(home,day) or aid not in book.candidates(away,day): raise ValueError('Brankář není v dostupných předchozích záznamech týmu.')
    projected=[book.project(home,day),book.project(away,day)]; chosen=[book.profile(hid,day),book.profile(aid,day)]
    x=target.iloc[:,5:].to_numpy(float); vals=np.array([np.r_[x[0],u['vector'],v['vector']] for u,v in [projected,chosen]])
    p,q=_fit(train,vals,True)
    return pd.DataFrame({'Scénář':['Odhad z historie','Ruční výběr'],'1':p[:,0],'X':p[:,1],'2':p[:,2],**{f'Over {line}':q[:,j+1:].sum(1) for j,line in enumerate([4.5,5.5,6.5])}}),len(train)
