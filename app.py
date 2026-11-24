import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date
from model import validate,rates,predict,backtest
from scipy.stats import skellam
import math

def elo(d):
    rating={}
    for _, games in d.sort_values("date").groupby("date"):
        delta={}
        for g in games.itertuples():
            h,a=g.home,g.away
            e=1/(1+10**((rating.get(a,1500)-rating.get(h,1500)-35)/400))
            actual=1 if g.hg>g.ag else 0 if g.hg<g.ag else .5
            change=20*(actual-e)
            delta[h]=delta.get(h,0)+change; delta[a]=delta.get(a,0)-change
        for t,x in delta.items(): rating[t]=rating.get(t,1500)+x
    return rating

def recent_shift(d,h,a):
    def recent(t):
        m=d[(d.home==t)|(d.away==t)].sort_values("date").tail(5)
        return (m.hg.where(m.home==t,m.ag)-m.ag.where(m.home==t,m.hg)).sum()/(len(m)+5)
    return .06*(recent(h)-recent(a))

def advanced(d,h,a,ratings):
    x,y=rates(d,h,a)
    def recent(t):
        m=d[(d.home==t)|(d.away==t)].sort_values("date").tail(5)
        gf=m.hg.where(m.home==t,m.ag); ga=m.ag.where(m.home==t,m.hg)
        return (gf-ga).sum()/(len(m)+5)
    # Experimental coefficients, not fitted. Preserve total expected goals.
    shift=max(-.35,min(.35,(ratings.get(h,1500)-ratings.get(a,1500))/1200 + .06*(recent(h)-recent(a))))
    u,v=x*math.exp(shift),y*math.exp(-shift)
    return u*(x+y)/(u+v),v*(x+y)/(u+v)

def compare(d):
    results=[[],[]]
    for day,games in d.groupby("date"):
        past=d[d.date<day]; r=elo(past)
        for g in games.itertuples():
            try: pairs=[rates(past,g.home,g.away),advanced(past,g.home,g.away,r)]
            except ValueError: continue
            actual=0 if g.hg>g.ag else 2 if g.hg<g.ag else 1
            for out,(x,y) in zip(results,pairs):
                p=[skellam.sf(0,x,y),skellam.pmf(0,x,y),skellam.cdf(-1,x,y)]
                out.append((sum((v-(i==actual))**2 for i,v in enumerate(p)),-math.log(max(p[actual],1e-15)),int(max(range(3),key=lambda i:p[i])==actual)))
    if not results[0]: raise ValueError("Nedostatek historie.")
    return pd.DataFrame([[name,len(r),*[sum(x[i] for x in r)/len(r) for i in range(3)]] for name,r in zip(["Základní Poisson","Experimentální Elo + forma"],results)],columns=["Model","Zápasů","Brier","Log Loss","Přesnost"])
from nhl import load, get
st.set_page_config(page_title='Hockey Analytics Lab',page_icon='🏒',layout='centered')
st.markdown('## 🏒 Hockey Analytics Lab')
st.caption('v0.9 • Advanced Statistics Beta • Poisson • výsledky po 60 minutách')
mode=st.radio('Zdroj dat',['NHL API','Tipsport extraliga','Vlastní CSV','Ruční scénář'],horizontal=True)
d=None
elh_fixtures=[]
if mode=='NHL API':
    year=date.today().year-(date.today().month<7)
    season=st.selectbox('Sezóna',[f'{y}{y+1}' for y in range(year,2020,-1)])
    path=Path('data')/f'nhl_{season}.csv'
    if st.button('Stáhnout / aktualizovat NHL'):
        try:
            bar=st.progress(0); data,errors=load(season,bar.progress); bar.empty()
            if errors: st.error('Neúplný import: '+ '; '.join(errors)); st.info('Původní soubor nebyl přepsán.')
            else: path.parent.mkdir(exist_ok=True); data.to_csv(path,index=False); st.success(f'Uloženo {len(data)} zápasů.')
        except Exception as e: st.error(str(e))
    if path.exists(): d=validate(pd.read_csv(path)); st.caption(f'Lokální data: {len(d)} zápasů, poslední datum {d.date.max().date()}')
    else: st.info('Klikni na stažení dat nebo vyzkoušej ruční scénář.')
elif mode=='Tipsport extraliga':
    from elh import load_elh, URL
    st.caption('Experimentální import: základní část 2026/27. Zdroj: iDNES.cz – los soutěže.')
    st.markdown(f'[Zdroj výsledků]({URL})')
    st.caption('Skóre P/N se převádí na remízu po 60 minutách. Dnešní výsledky se do historie nezahrnují.')
    if st.button('Stáhnout / aktualizovat extraligu'):
        try:
            with st.spinner('Načítám rozpis…'):
                data, fixtures, pending = load_elh()
                data=validate(data)
                if data.empty: raise ValueError('Žádné dokončené zápasy.')
            st.session_state['elh_data']=(data,fixtures,pending)
        except Exception as e: st.error(f'Import se nezdařil: {e}. Původní data zůstala zachována.')
    if 'elh_data' in st.session_state:
        d,elh_fixtures,pending=st.session_state['elh_data']
        st.caption(f'Načteno {len(d)} zápasů, poslední datum {d.date.max().date()}.')
        if pending: st.warning(f'{pending} minulých termínů bez rozpoznaného konečného skóre; historie může být neúplná.')
    else: st.info('Klikni na stažení dat. Při chybě lze využít vlastní CSV.')
elif mode=='Vlastní CSV':
    st.caption('Sloupce date,home,away,hg,ag; skóre po 60 minutách, datum YYYY-MM-DD.')
    st.download_button('Stáhnout prázdnou šablonu CSV','date,home,away,hg,ag\n',file_name='sablona.csv')
    f=st.file_uploader('Historické zápasy',type='csv')
    if f:
        try: d=validate(pd.read_csv(f))
        except Exception as e: st.error(str(e))
if mode=='Ruční scénář':
    st.warning('Ruční předpoklady, nikoliv predikce z ověřených dat.')
    h=st.text_input('Domácí','Hradec'); a=st.text_input('Hosté','Třinec')
    lh=st.number_input('Očekávané góly domácích',.1,12.,2.5,.1); la=st.number_input('Očekávané góly hostů',.1,12.,3.2,.1)
elif d is not None and not d.empty:
    with st.expander('Rozšíření historických dat (volitelné)'):
        extra=st.file_uploader('Starší sezony stejné soutěže, skóre po 60 minutách',type='csv',accept_multiple_files=True)
        if extra:
            try:
                combined=validate(pd.concat([d]+[pd.read_csv(f) for f in extra],ignore_index=True))
                if combined.duplicated(['date','home','away']).any(): raise ValueError('Rozporné skóre stejného zápasu.')
                d=combined
                st.caption(f'Rozšířená historie: {len(d)} zápasů.')
            except Exception as e: st.error(str(e))
    with st.expander('Historická databáze – více sezon'):
        from history import merge_history, season_year
        if mode=='NHL API':
            start=int(season[:4])
            years=st.multiselect('Starší sezony NHL',list(range(start-1,start-5,-1)),default=list(range(start-1,start-5,-1)))
            if st.button('Stáhnout vybrané starší sezony NHL'):
                for yr in years:
                    ss=f'{yr}{yr+1}'; dest=Path('data')/f'nhl_{ss}.csv'
                    try:
                        with st.spinner(f'Načítám {ss}…'):
                            old,errors=load(ss)
                            if errors or len(old)!=1312 or old.duplicated(['date','home','away']).any(): raise ValueError(f'Neúplná sezona: {len(old)} / 1312 zápasů. '+ '; '.join(errors))
                            dest.parent.mkdir(exist_ok=True); old.to_csv(dest,index=False)
                        st.success(f'{ss}: {len(old)} zápasů')
                    except Exception as e: st.error(f'{ss}: {e}')
            frames=[d]
            for yr in years:
                dest=Path('data')/f'nhl_{yr}{yr+1}.csv'
                if dest.exists():
                    old=validate(pd.read_csv(dest))
                    if len(old)==1312: frames.append(old)
            history=merge_history(frames)
        else:
            history=merge_history([d])
            st.caption('Starší sezony extraligy přidej výše pomocí CSV z oprávněného zdroje. Automatický archiv zatím není dostupný.')
        st.dataframe(history.assign(Sezona=season_year(history.date)).groupby('Sezona').size().rename('Zápasů').reset_index(),hide_index=True)
        st.download_button('Export historických dat CSV',history.to_csv(index=False),file_name='historie.csv')
        st.caption('Historie pro sezonní test. Živý model NHL nadále používá vybranou sezonu.')
    teams=sorted(set(d.home)|set(d.away))
    cutoff=st.date_input('Datum predikce (datum zápasu dle zdroje)',date.today())
    past=d[d.date<pd.Timestamp(cutoff)]
    game=None
    fixtures=[]
    ratings=elo(past)
    experimental=st.checkbox("Experimentální Elo + forma",value=False)
    st.caption("Volitelný model s testovacími vahami; zatím nekalibrován. Celkový očekávaný počet gólů nemění.")
    def selected_rates(h,a):
        return advanced(past,h,a,ratings) if experimental else rates(past,h,a)
    if mode=='NHL API':
        @st.cache_data(ttl=900,show_spinner=False)
        def schedule(day):
            payload=get('/schedule/'+day.isoformat())
            return [(g['homeTeam']['abbrev'],g['awayTeam']['abbrev']) for week in payload['gameWeek'] if week['date']==day.isoformat() for g in week['games'] if g.get('gameType')==2]
        try:
            fixtures=[(x,y) for x,y in schedule(cutoff) if x in teams and y in teams]
            if fixtures: game=st.selectbox('Zápasy NHL (domácí – hosté)',[None]+fixtures,format_func=lambda x:'Vlastní výběr týmů' if x is None else ' – '.join(x))
            else: st.caption('Pro toto datum nebyly nalezeny zápasy s dostupnou historií.')
        except Exception: st.caption('Rozpis se nepodařilo načíst; ruční výběr je dostupný.')
    if mode=='Tipsport extraliga':
        fixtures=[(h,a) for day,h,a in elh_fixtures if day==cutoff.isoformat() and h in teams and a in teams]
        if fixtures: game=st.selectbox('Zápasy extraligy (domácí – hosté)',[None]+fixtures,format_func=lambda x:'Vlastní výběr týmů' if x is None else ' – '.join(x))
        else: st.caption('Pro vybraný den nejsou v rozpisu dostupné zápasy.')
    if fixtures and st.checkbox("Přehled všech zápasů dne"):
        rows=[]
        for ht,at in fixtures:
            try:
                x,y=selected_rates(ht,at); pp,tt,_=predict(x,y)
                rows.append([ht+" – "+at,*[f"{z:.1%}" for z in pp],f"{x+y:.2f}"])
            except ValueError: rows.append([ht+" – "+at,"—","—","—","—"])
        st.dataframe(pd.DataFrame(rows,columns=["Zápas","1","X","2","Góly"]),hide_index=True,use_container_width=True)
    st.caption(f"Historie před datem predikce: {len(past)} zápasů, {len(set(past.home)|set(past.away))} týmů.")
    if game: h,a=game
    else:
        h=st.selectbox('Domácí',teams); a=st.selectbox('Hosté',teams,index=min(1,len(teams)-1))
    try: lh,la=selected_rates(h,a)
    except ValueError as e: st.info(str(e)); st.stop()
else: st.stop()
if h==a: st.error('Vyber dva různé týmy.'); st.stop()
p,totals,scores=predict(lh,la)
st.subheader(f'{h} – {a}')
st.dataframe(pd.DataFrame([[n,f'{v:.1%}',f'{1/v:.2f}' if v else '—'] for n,v in zip(['Domácí','Remíza','Hosté'],p)],columns=['Výsledek','Šance','Férový kurz']),hide_index=True,use_container_width=True)
if d is not None:
    st.caption(f'Elo po 60 minutách: {h} {ratings.get(h,1500):.0f} | {a} {ratings.get(a,1500):.0f}')
with st.expander('Vítěz včetně prodloužení – orientační scénář'):
    st.write(f'{h}: {p[0]+p[1]/2:.1%} | {a}: {p[2]+p[1]/2:.1%}')
    st.caption('Remízy rozděleny 50/50; nejde o naučený model prodloužení ani nájezdů.')
st.write(f'Očekávané góly: {lh:.2f} + {la:.2f} = {lh+la:.2f}')
rows=[]; candidates=[]
for k,v in totals:
    for name,q in [('Over',v),('Under',1-v)]:
        rows.append([f'{name} {k}',f'{q:.1%}',f'{1/q:.2f}' if q else '—']); candidates.append((q,f'{name} {k}'))
q,label=max(candidates)
st.subheader('Nejpravděpodobnější gólová hranice')
st.write(f'**{label} · {q:.1%} · férový kurz {1/q:.2f}**')
st.caption('Nejvyšší modelová pravděpodobnost ze zobrazených hranic, nikoli vyhodnocení výhodnosti sázky.')
with st.expander('Všechny gólové hranice po 60 minutách'):
    st.dataframe(pd.DataFrame(rows,columns=['Hranice','Pravděpodobnost','Férový kurz']),hide_index=True,use_container_width=True)
if d is not None:
    with st.expander('Vstupní statistiky a forma'):
        for team,venue in [(h,'home'),(a,'away')]:
            matches=past[(past.home==team)|(past.away==team)].sort_values('date').tail(5)
            gf=matches.hg.where(matches.home==team,matches.ag); ga=matches.ag.where(matches.home==team,matches.hg)
            st.write(f'**{team}** — posledních {len(matches)} zápasů: {sum(gf>ga)} V / {sum(gf==ga)} R / {sum(gf<ga)} P, skóre {int(gf.sum())}:{int(ga.sum())} po 60 min.')
            st.caption(f'V modelu: {len(past[past[venue]==team])} zápasů na příslušném hřišti.')
        st.caption('Forma mění poměr očekávaných gólů jen při zapnutí experimentálního modelu.')
with st.expander('Potvrzení brankáři (poznámka)'):
    st.text_input('Brankář domácích',key='goalie_home'); st.text_input('Brankář hostů',key='goalie_away')
    st.caption('Jména jsou pouze poznámka; bez ověřených statistik nemění pravděpodobnosti.')
st.subheader('Nejpravděpodobnější skóre'); st.dataframe(pd.DataFrame([(f'{i}:{j}',f'{v:.1%}') for i,j,v in scores],columns=['Skóre','Pravděpodobnost']),hide_index=True,use_container_width=True)
st.caption('Nezávislý Poisson; domácí/venkovní síla s vyhlazením 8 ligovými zápasy. Volitelné experimentální Elo + forma. Bez kalibrace brankářů, sestav a prodloužení. Férový kurz není doporučení k sázce.')
if d is not None and st.button('Porovnat modely na historii (walk-forward)'):
    try:
        with st.spinner('Probíhá test...'): result=compare(past)
        st.dataframe(result,hide_index=True,use_container_width=True)
        st.caption('Oba modely na stejných zápasech, vždy jen předchozí dny. Nižší Brier a Log Loss jsou lepší. Výsledek není nezávislá validace po výběru modelu.')
    except ValueError as e: st.info(str(e))

if d is not None:
    st.subheader('Validační laboratoř')
    st.caption('Chronologické ladění na prvních 70 % vyhodnotitelných dnů, test na zbývajících dnech. Váhy se automaticky nepřenášejí do živé predikce.')
    if st.button('Spustit validaci 1X2 a Over/Under'):
        try:
            from validation import evaluate
            with st.spinner('Testuji modely…'):
                params,boundary,n,summary,ou,cal=evaluate(past,elo,recent_shift)
            st.session_state['validation_result']=(params,boundary,n,summary,ou,cal)
            st.session_state['validation_key']=(mode,str(cutoff),len(past),int(pd.util.hash_pandas_object(past,index=False).sum()))
        except Exception as e: st.error(str(e))
    if st.session_state.get('validation_key')==(mode,str(cutoff),len(past),int(pd.util.hash_pandas_object(past,index=False).sum())):
        params,boundary,n,summary,ou,cal=st.session_state['validation_result']
        st.write(f'Ladění: {n} zápasů. Test od {boundary.date()}. Váhy Elo / forma / celkové góly: {params}.')
        st.dataframe(summary,hide_index=True)
        st.dataframe(ou,hide_index=True)
        st.write('Kontrola kalibrace laděného modelu (Over)')
        st.dataframe(cal,hide_index=True)
        st.caption('Nižší Brier a Log Loss jsou lepší. Malé skupiny jsou nespolehlivé. Opakovaný výběr podle tohoto testu vyžaduje nový nezávislý test.')
    with st.expander('Monte Carlo – kontrola numerické nejistoty'):
        if st.button('100 000 simulací'):
            import numpy as np
            rng=np.random.default_rng(42); x=rng.poisson(lh,100000); y=rng.poisson(la,100000)
            st.write(f'Domácí {np.mean(x>y):.2%} | remíza {np.mean(x==y):.2%} | hosté {np.mean(x<y):.2%}')
            st.write(pd.DataFrame({'Over':[4.5,5.5,6.5],'Simulace':[float(np.mean(x+y>k)) for k in (4.5,5.5,6.5)]}))
            st.caption('Simulace stejného Poissonova modelu; nezahrnuje nejistotu vstupních parametrů a nezvyšuje přesnost modelu.')

if d is not None:
    st.subheader('Sezonní test regrese')
    st.caption('Model se učí pouze ze starších sezon; parametry jsou v testované sezoně zmrazené. Poločas váhy historie 365 dní. Negativně binomické rozdělení pouze pro celkové góly.')
    if st.button('Spustit sezonní test'):
        try:
            from history import season_test
            with st.spinner('Počítám regresi…'):
                report=season_test(history[history.date<pd.Timestamp(cutoff)])
            st.dataframe(report,hide_index=True)
            st.download_button('Stáhnout výsledky testu CSV',report.to_csv(index=False),file_name='sezonni_test.csv')
        except Exception as e: st.error(str(e))

if d is not None:
    st.subheader('Dynamická validační laboratoř')
    st.caption('Přeučení každých 14 dní. Poločasy 30, 60, 90, 180 a 365 dní; kombinace s ligovým průměrem 0/50/100 %. Parametry pro 1X2 a góly vybíráme odděleně jen podle dřívějších predikcí. Živé predikce se nemění.')
    dynamic_data=history[history.date<pd.Timestamp(cutoff)]
    dynamic_key=(mode,str(cutoff),len(dynamic_data),int(pd.util.hash_pandas_object(dynamic_data,index=False).sum()))
    if st.button('Spustit dynamický test'):
        try:
            from dynamic import dynamic_test
            with st.spinner('Přeučuji modely a testuji historické bloky…'):
                st.session_state['dynamic_report']=dynamic_test(dynamic_data)
                st.session_state['dynamic_key']=dynamic_key
        except Exception as e: st.error(str(e))
    if st.session_state.get('dynamic_key')==dynamic_key:
        summary,detail=st.session_state['dynamic_report']
        st.dataframe(summary,hide_index=True)
        st.download_button('Stáhnout dynamický souhrn CSV',summary.to_csv(index=False),file_name='dynamicky_test.csv')
        st.download_button('Stáhnout jednotlivé predikce CSV',detail.to_csv(index=False),file_name='dynamicke_predikce.csv')
        st.caption('Retrospektivní vývojový test, nikoli nový nezávislý důkaz přesnosti. Nižší Brier a Log Loss jsou lepší.')

if d is not None and mode=='NHL API':
    st.subheader('Pokročilé statistiky NHL – v0.8 beta')
    st.markdown('Zdroj: [MoneyPuck.com](https://www.moneypuck.com/data.htm). Stáhni soubor **all game level data for all teams for all seasons** a nahraj jej níže. Pouze nekomerční použití.')
    st.caption('Model využívá průměry xG, střel pro a proti z posledních 15 utkání a dny odpočinku. Žádná statistika právě predikovaného dne se nepoužívá. Živé predikce zůstávají beze změny.')
    mp=st.file_uploader('MoneyPuck týmové statistiky (CSV nebo ZIP)',type=['csv','zip'],key='mp08')
    if mp is not None:
        import hashlib
        mp_key=(hashlib.sha256(mp.getbuffer()).hexdigest(),str(cutoff),int(pd.util.hash_pandas_object(history,index=False).sum()))
    if mp is not None and st.button('Spustit test pokročilých statistik'):
        try:
            from advanced import read_stats,evaluate
            with st.spinner('Testuji pokročilé statistiky…'):
                stats=read_stats(mp)
                report,detail,n=evaluate(history[history.date<pd.Timestamp(cutoff)],stats)
            st.session_state['mp08_result']=(report,detail,n,str(stats.date.max().date()))
            st.session_state['mp08_key']=mp_key
        except Exception as e: st.session_state.pop('mp08_result',None); st.error(str(e))
    if mp is not None and 'mp08_result' in st.session_state and st.session_state.get('mp08_key')==mp_key:
        report,detail,n,last=st.session_state['mp08_result']
        st.write(f'Použitelné zápasy: {n}; poslední datum MoneyPuck: {last}.')
        st.dataframe(report,hide_index=True)
        st.download_button('Stáhnout pokročilý souhrn CSV',report.to_csv(index=False),file_name='advanced_summary.csv')
        st.download_button('Stáhnout pokročilý detail CSV',detail.to_csv(index=False),file_name='advanced_detail.csv')
        st.caption('Test každých 28 dní, trénink jen na předchozích blocích. Výsledky porovnávej s v0.7 jen na totožných zápasech.')

    st.subheader('Speciální formace a týmové chytání – v0.9 beta')
    st.caption('MoneyPuck.com: stejný týmový soubor jako výše. Testujeme xG za 60 minut při 5 na 4 a 4 na 5, a týmový rozdíl xGA minus GA a aproximaci úspěšnosti. Nejde o individuální statistiky potvrzeného brankáře. Vše pouze z předchozích dnů.')
    if mp is not None and st.button('Spustit test speciálních formací'):
        try:
            from advanced import read_stats
            from special import read_extra,evaluate
            with st.spinner('Porovnávám modely na shodných zápasech…'):
                mp.seek(0); stats=read_stats(mp); extra=read_extra(mp)
                st.session_state['special_result']=evaluate(history[history.date<pd.Timestamp(cutoff)],stats,extra)
                st.session_state['special_key']=mp_key
        except Exception as e: st.session_state.pop('special_result',None); st.error(str(e))
    if mp is not None and 'special_result' in st.session_state and st.session_state.get('special_key')==mp_key:
        summary,detail,n=st.session_state['special_result']
        st.write(f'Společný vzorek: {n} zápasů před vyřazením trénovacího období.')
        st.dataframe(summary,hide_index=True)
        st.download_button('Stáhnout souhrn v0.9 CSV',summary.to_csv(index=False),file_name='special_summary.csv')
        st.download_button('Stáhnout detail v0.9 CSV',detail.to_csv(index=False),file_name='special_detail.csv')
        st.caption('Všechny tři modely používají shodné zápasy a časové bloky. Živé predikce se nemění.')
