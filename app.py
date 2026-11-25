import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date
from model import validate,rates,predict,backtest
from scipy.stats import skellam
import math
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED


def result_zip(files):
    """Pack already computed tables; no model training or disk writes."""
    out = BytesIO()
    with ZipFile(out, 'w', compression=ZIP_DEFLATED) as archive:
        for name, table in files:
            archive.writestr(name, table.to_csv(index=False).encode('utf-8'))
    return out.getvalue()


def download_file(label, data, file_name, mime=None):
    # Streamlit >=1.43 supports browser-only downloads. Older supported
    # releases (>=1.40) can isolate the download in a fragment.
    import re
    version = tuple(map(int, re.match(r'(\d+)\.(\d+)', st.__version__).groups()))
    kwargs = dict(file_name=file_name, key='download_' + file_name)
    if mime is not None:
        kwargs['mime'] = mime
    if version >= (1, 43):
        st.download_button(label, data, on_click='ignore', **kwargs)
    else:
        @st.fragment
        def isolated_download():
            st.download_button(label, data, **kwargs)
        isolated_download()


def download_results(label, zip_name, *files):
    download_file(label, result_zip(files), zip_name, mime='application/zip')

def session_upload(label, slot, multiple=False, types=None):
    """Keep bytes in this browser session even when widget is not rendered."""
    generation=st.session_state.get(slot+'_generation',0)
    widget=slot+'_upload_'+str(generation)
    def remember():
        value=st.session_state.get(widget)
        files=(value or []) if multiple else ([value] if value is not None else [])
        st.session_state[slot]=tuple((f.name,f.getvalue()) for f in files)
    def forget():
        st.session_state.pop(slot,None)
        st.session_state[slot+'_generation']=generation+1
    st.file_uploader(label,type=types or ['csv','zip'],accept_multiple_files=multiple,key=widget,on_change=remember)
    payload=st.session_state.get(slot,())
    if payload:
        st.caption('Uchováno pro tuto návštěvu: '+', '.join(name for name,_ in payload))
        st.button('Odebrat uložená data',key=slot+'_clear',on_click=forget)
    streams=[]
    for name,raw in payload:
        f=BytesIO(raw); f.name=name; streams.append(f)
    return streams if multiple else (streams[0] if streams else None)

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
st.caption('v1.9 • Lineup Intelligence Beta • týmové výsledky po 60 minutách')
page=st.radio('Režim',['Analýza zápasu','Hráči NHL','Datové centrum','Validační laboratoř / brankáři'],horizontal=True,key='page12')
from datahub import SLOTS, season_data, complete, build_backup, restore_backup
from europe import LEAGUES, TABLE_URLS, read_history as europe_history, fetch_table
import os
import tempfile

@st.cache_data(max_entries=20,show_spinner=False)
def parse_nhl_bytes(season, raw):
    return season_data(raw,season)

def nhl_raw(season):
    restored=st.session_state.get('nhl14_restored',{})
    if season in restored: return restored[season]
    path=Path('data')/f'nhl_{season}.csv'
    return path.read_bytes() if path.exists() else None

def nhl_data(season):
    raw=nhl_raw(season)
    return parse_nhl_bytes(season,raw) if raw is not None else None

def refresh_nhl(season, require_complete=False, force=False):
    # Only source-fetched data are written to the shared ephemeral filesystem.
    if require_complete and not force:
        try:
            prior=nhl_data(season)
            if prior is not None and complete(prior,season): return len(prior),True
        except (ValueError,OSError): pass
    data,errors=load(season)
    if errors: raise ValueError('Neúplný import: '+ '; '.join(errors))
    raw=data.to_csv(index=False).encode('utf-8'); data=season_data(raw,season)
    if require_complete and not complete(data,season): raise ValueError(f'Neúplná sezona: {len(data)} / 1312 zápasů.')
    path=Path('data')/f'nhl_{season}.csv'; path.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
        tmp.write(raw); temp=tmp.name
    try: os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    st.session_state.get('nhl14_restored',{}).pop(season,None)
    return len(data),False

def european_upload(league):
    return session_upload('Historie a případný rozpis – '+league+' (CSV)', 'eu18_'+LEAGUES[league], multiple=True, types=['csv'])

@st.cache_data(ttl=1800, max_entries=2, show_spinner=False)
def open_standings(league):
    return fetch_table(league)

def standings_panel(league):
    if league not in TABLE_URLS: return
    with st.expander('Tabulka soutěže – otevřená data', expanded=False):
        st.markdown('Zdroj: [SazkyWeb.cz – otevřená data](https://sazkyweb.cz/otevrena-data.html), CC BY 4.0. Tabulka není historie jednotlivých zápasů a nepoužívá se k predikci.')
        slot='standings18_'+league
        if st.button('Načíst tabulku '+league,key='load_'+slot):
            try: st.session_state[slot]=open_standings(league)
            except Exception as e: st.warning('Tabulku nelze aktualizovat: '+str(e))
        if slot in st.session_state:
            table,stamp=st.session_state[slot]
            st.caption('Aktualizace zdroje: '+stamp)
            try:
                age=(pd.Timestamp.now(tz='UTC')-pd.Timestamp(stamp)).total_seconds()/3600
                if age>48 or age<0: st.warning('Čas aktualizace je mimo očekávané období; zkontroluj aktuálnost zdroje.')
            except Exception: pass
            st.dataframe(table.rename(columns={'poradi':'#','tym':'Tým','zapasy':'Z','vyhry':'V','vyhry_prodl':'VP','prohry_prodl':'PP','prohry':'P','skore':'Skóre','body':'Body'}),hide_index=True)

if page=='Datové centrum':
    st.subheader('Datové centrum')
    st.caption('Aktualizuj NHL jediným tlačítkem. MoneyPuck můžeš nahrát nebo obnovit z vlastní zálohy. Cloudové soubory nejsou trvalé; zálohu si ulož do Souborů nebo na Disk.')
    year=date.today().year-(date.today().month<7)
    seasons=[f'{y}{y+1}' for y in range(year,year-5,-1)]
    chosen=st.multiselect('Sezony NHL',seasons,default=seasons,key='hub_seasons14')
    force=st.checkbox('Znovu stáhnout i kompletní starší sezony',key='hub_force14')
    if st.button('Aktualizovat vybraná data NHL',key='hub_update14'):
        for ss in chosen:
            try:
                with st.spinner(f'Kontroluji / načítám {ss}…'):
                    n,skipped=refresh_nhl(ss,require_complete=ss!=seasons[0],force=force)
                st.success(f'{ss}: {n} zápasů — '+('již uloženo' if skipped else 'aktualizováno'))
            except Exception as e: st.error(f'{ss}: {e}. Původní data ponechána.')
    loaded={}; rows=[]
    all_seasons=sorted(set(seasons)|set(st.session_state.get('nhl14_restored',{}))|{p.stem[4:] for p in Path('data').glob('nhl_*.csv')},reverse=True)
    for ss in all_seasons:
        try:
            raw=nhl_raw(ss)
            if raw is None: rows.append([ss,0,'—','Chybí']); continue
            d0=parse_nhl_bytes(ss,raw); loaded[ss]=raw
            rows.append([ss,len(d0),str(d0.date.max().date()),'Kompletní' if complete(d0,ss) else 'Průběžná / neúplná'])
        except Exception as e: st.warning(f'{ss}: {e}')
    st.dataframe(pd.DataFrame(rows,columns=['Sezona','Zápasy','Poslední výsledek','Stav']),hide_index=True)
    st.markdown('**Doplňková data MoneyPuck**')
    st.markdown('[MoneyPuck.com – datové soubory](https://www.moneypuck.com/data.htm) · pouze nekomerční použití.')
    session_upload('Týmové statistiky (CSV / ZIP)','mp12_session')
    session_upload('Brankáři (CSV / ZIP)','goalie12_session',multiple=True)
    st.markdown('**Vlastní ligová data**')
    session_upload('Vlastní soutěž (CSV)','own14_session',types=['csv'])
    session_upload('Starší sezony vlastní soutěže','extra_own14_session',multiple=True,types=['csv'])
    session_upload('Starší sezony extraligy z oprávněného zdroje','extra_elh14_session',multiple=True,types=['csv'])
    st.caption('Vlastní CSV: date,home,away,hg,ag; skóre po 60 minutách. V rámci jednoho pole nemíchej soutěže.')
    st.subheader('Evropské soutěže')
    st.caption('Každá liga má samostatnou historii; v záloze se uchovají všechny nahrané ligy. Používej pouze oprávněně získané výsledky základní části po 60 minutách. Prázdné hg a ag označují rozpis, nikoliv 0:0.')
    eu_choice=st.selectbox('Evropská soutěž',list(LEAGUES),key='eu_hub18')
    eu_files=european_upload(eu_choice)
    if eu_files:
        try:
            ed,ef=europe_history(eu_files)
            st.success(f'{eu_choice}: {len(ed)} výsledků, {len(ef)} termínů včetně dokončených zápasů.')
        except Exception as e: st.error(str(e))
    standings_panel(eu_choice)
    st.dataframe(pd.DataFrame([[name,len(st.session_state.get('eu18_'+code,()))] for name,code in LEAGUES.items()],columns=['Soutěž','Uložené CSV']),hide_index=True)
    download_file('Šablona historie a rozpisu CSV','date,home,away,hg,ag\n',file_name='europe_template.csv')
    st.subheader('Jedna přenosná záloha')
    st.caption('Obsahuje načtené NHL sezony a původní nahrané soubory. Neobsahuje predikce, výsledky testů ani automaticky stažený rozpis extraligy. Nezveřejňuj soubory MoneyPuck na GitHubu.')
    if st.button('Připravit jednu zálohu dat (ZIP)',key='hub_pack14'):
        try:
            with st.spinner('Baluji data…'):
                st.session_state['hub_zip14']=build_backup(loaded,{s:st.session_state.get(s,()) for s in SLOTS})
        except Exception as e: st.session_state.pop('hub_zip14',None); st.error(str(e))
    if 'hub_zip14' in st.session_state:
        st.caption('Záloha obsahuje data z okamžiku stisku Připravit. Po aktualizaci připrav novou.')
        download_file('Stáhnout kompletní zálohu dat',st.session_state['hub_zip14'],file_name='hockey_data_backup.zip',mime='application/zip')
        if st.button('Uvolnit připravenou zálohu z paměti',key='hub_free14'):
            st.session_state.pop('hub_zip14',None); st.rerun()
    restore=st.file_uploader('Obnovit vlastní zálohu ZIP',type='zip',key='hub_restore_file14')
    if restore is not None and st.button('Obnovit data ze zálohy',key='hub_restore14'):
        try:
            restored,uploads=restore_backup(restore.getvalue())
            st.session_state.setdefault('nhl14_restored',{}).update(restored)
            for slot,payload in uploads.items():
                st.session_state[slot]=payload
                st.session_state[slot+'_generation']=st.session_state.get(slot+'_generation',0)+1
            st.session_state.pop('hub_zip14',None)
            st.success(f'Obnoveno {len(restored)} sezon NHL a {sum(map(len,uploads.values()))} nahraných souborů. Data jsou připravena v této návštěvě; starší data mimo zálohu zůstala zachována.')
            st.info('Přejdi do režimu Analýza zápasu. Obnovení stránky v Safari může ukončit relaci.')
        except Exception as e: st.error('Zálohu nelze obnovit: '+str(e))
    st.stop()

if page=='Hráči NHL':
    player_mode=st.radio('Typ analýzy hráčů',['Celý zápas','Jednotlivý hráč'],horizontal=True,key='player_mode15')
    if player_mode=='Celý zápas':
        from playerbatch import fixtures_for_day, batch, PacedGet, candidates
        from lineups import match_key, validate_selection, annotate
        from players import TEAMS, season_start
        import time
        st.subheader('Hráči obou týmů – hromadná analýza')
        st.caption('Pouze NHL, základní část. Zdroj: NHL API. Odhady platí, pouze pokud hráč nastoupí. Sezonní soupiska není potvrzenou sestavou. Výsledek početně zahrnuje prodloužení, nikoli nájezdy. Základní model v1.2 zůstává výchozí. Vliv soupeře lze samostatně otestovat z uložených výsledků NHL.')
        day=st.date_input('Datum zápasu (dle rozpisu NHL)',date.today(),key='batch_day15')
        yy=season_start(day)
        @st.cache_data(ttl=900,show_spinner=False,max_entries=24)
        def batch_schedule(day): return fixtures_for_day(get('/schedule/'+day.isoformat()),day)
        try:
            games=batch_schedule(day)
        except Exception as e:
            games=[]; st.warning('Rozpis není dostupný: '+str(e))
        game=st.selectbox('Zápas (domácí – hosté)',games+[None],format_func=lambda x:'Vlastní výběr týmů' if x is None else ' – '.join(x),key='batch_game15_'+str(day))
        if game is None:
            ts=TEAMS if yy>=2024 else sorted(set(TEAMS)-{'UTA'}|{'ARI'})
            h=st.selectbox('Domácí',ts,key='batch_home15')
            a=st.selectbox('Hosté',[t for t in ts if t!=h],key='batch_away15')
        else: h,a=game
        api_state=st.session_state.setdefault('player_api161',{})
        remaining=max(0,math.ceil(api_state.get('blocked_until',0)-time.time()))
        if remaining:
            st.warning(f'NHL API omezilo stahování. Další pokus nejdříve za {remaining} s. Úspěšně načtená data zůstanou v paměti; po pauze stiskni tlačítko znovu.')
        # Load only two roster endpoints before selecting which player logs to fetch.
        match_id=match_key('NHL',day,h,a)
        roster_key='lineup19_roster_'+match_id
        selection_mode=st.radio('Rozsah hráčů',['Celá sezonní soupiska','Vlastní výběr ze sestavy'],horizontal=True,key='lineup19_mode')
        selected_ids=None; roles={}; selection_ok=True
        if selection_mode=='Vlastní výběr ze sestavy':
            st.caption('Vyber skutečně nastupující hráče podle svého zdroje. Aplikace jejich účast sama nepotvrzuje; výběr nemění modelové pravděpodobnosti.')
            if st.button('Načíst seznam hráčů obou týmů',key='lineup19_load'):
                try:
                    paced=PacedGet(get,api_state)
                    people,duplicates=candidates(h,a,paced(f'/roster/{h}/{yy}{yy+1}'),paced(f'/roster/{a}/{yy}{yy+1}'))
                    st.session_state[roster_key]=(people,duplicates)
                except Exception as e: st.warning('Soupisky nejsou dostupné: '+str(e))
            roster_report=st.session_state.get(roster_key)
            selection_ok=False
            if roster_report:
                people,duplicates=roster_report
                if duplicates: st.warning('Hráči uvedení v obou soupiskách byli vyřazeni.')
                available={t:{pid:name for team,pid,name in people if team==t} for t in (h,a)}
                chosen={}
                for team in (h,a):
                    names=available[team]
                    with st.expander(f'{team} – výběr hráčů',expanded=True):
                        ids=st.multiselect('Hráči, kteří mají nastoupit',list(names),format_func=lambda p:names[p],key=roster_key+'_'+team)
                        chosen[team]=ids
                        if ids:
                            st.caption('Volitelné označení formací. Neovlivňuje výpočet.')
                            for pid in ids: roles[int(pid)]={}
                            used=set()
                            for line in ('Útok 1','Útok 2','Útok 3','Útok 4','Obrana 1','Obrana 2','Obrana 3'):
                                line_ids=st.multiselect(line,[p for p in ids if p not in used],format_func=lambda p:names[p],key=roster_key+'_'+team+'_'+line,max_selections=3 if line.startswith('Útok') else 2)
                                for pid in line_ids: roles[int(pid)]['line']=line
                                used.update(line_ids)
                            pp_used=set()
                            for pp in ('PP1','PP2'):
                                pp_ids=st.multiselect(pp,[p for p in ids if p not in pp_used],format_func=lambda p:names[p],key=roster_key+'_'+team+'_'+pp,max_selections=5)
                                for pid in pp_ids: roles[int(pid)]['pp']=pp
                                pp_used.update(pp_ids)
                        st.caption(f'Vybráno {len(ids)} hráčů; ostatní se nebudou stahovat.')
                try:
                    selected_ids=validate_selection(available,chosen)
                    selection_ok=True
                except ValueError as e: st.info(str(e))
        key=(str(day),h,a,None if selection_mode=='Celá sezonní soupiska' else tuple(sorted(selected_ids or ())))
        if st.button('Analyzovat / doplnit chybějící hráče',key='batch_run15',disabled=not selection_ok):
            if time.time()<api_state.get('blocked_until',0):
                st.info('Pauza API ještě neskončila. Zkus tlačítko později.')
                st.stop()
            st.session_state.pop('opponent_report16',None)
            now=time.time()
            cache=st.session_state.setdefault('batch_logs15',{})
            cache={k:v for k,v in cache.items() if now-v[0]<1800}
            existing={pid:v[1] for (year,pid),v in cache.items() if year==yy}
            bar=st.progress(0.0,text='Načítám soupisky…')
            try:
                def progress(i,n): bar.progress(i/n,text=f'Zpracováno {i}/{n} hráčů')
                rows,errors,fetched=batch(PacedGet(get,api_state),h,a,day,yy,existing,progress,selected_ids=selected_ids)
                cache.update({(yy,pid):(time.time(),v) for pid,v in fetched.items()})
                st.session_state['batch_logs15']=dict(sorted(cache.items(),key=lambda kv:kv[1][0])[-160:])
                st.session_state['batch_report15']=(key,rows,errors)
            except Exception as e: st.error('Hromadná analýza: '+str(e))
            finally: bar.empty()
        report=st.session_state.get('batch_report15')
        if report and report[0]==key:
            _,rows,errors=report
            rows=annotate(rows,roles) if selected_ids is not None else rows
            st.caption('Rozsah: vlastní výběr uživatele.' if selected_ids is not None else 'Rozsah: sezonní soupiska.')
            st.caption(f'Analyzováno: {len(rows)} | Vynecháno: {len(errors)} záznamů soupisky. Úspěšně načtené historie se v této návštěvě uchovávají až 30 minut.')
            incomplete=not errors.empty and errors['Důvod'].str.contains('429',regex=False).any()
            if incomplete: st.warning('Neúplná analýza: NHL API omezilo požadavky. Po uplynutí pauzy znovu stiskni Analyzovat / doplnit chybějící hráče. Výsledky nereprezentují kompletní soupisky.')
            if rows.empty: st.warning('Není dostatek dostupných dat pro výpočet.')
            else:
                market=st.radio('Seřadit podle pravděpodobnosti',['Bod ANO','Bod NE','Gól ANO'],horizontal=True,key='batch_market15')
                scope=st.radio('Zobrazit tým',['Oba týmy',h,a],horizontal=True,key='batch_scope15')
                visible=rows if scope=='Oba týmy' else rows[rows['Tým']==scope]
                visible=visible.sort_values([market,'Hráč'],ascending=[False,True])
                st.dataframe(visible[['Tým','Hráč']+(['Formace','Přesilovka'] if selected_ids is not None else [])+['Bod ANO','Bod NE','Gól ANO','Zápasy','Stav']].style.format({m:'{:.1%}' for m in ['Bod ANO','Bod NE','Gól ANO']}),hide_index=True)
                with st.expander('Produktivita a čas na ledě'):
                    st.dataframe(visible[['Tým','Hráč','Body/zápas L5','Body/zápas L10','Body/zápas L20','Čas na ledě L20 (min)','Poslední zápas']].style.format({m:'{:.2f}' for m in ['Body/zápas L5','Body/zápas L10','Body/zápas L20','Čas na ledě L20 (min)']}),hide_index=True)
            if not errors.empty:
                with st.expander('Vynechaní hráči a důvody'):
                    st.dataframe(errors,hide_index=True)
            if not rows.empty:
                st.markdown('### Vliv soupeře – v1.6 experiment')
                st.caption('Používá uloženou historii NHL, nikoli další soubory. Zohledňuje inkasované góly soupeře po 60 minutách; hráčské body a góly zahrnují i prodloužení. Základní predikce zůstávají nezměněné.')
                if st.button('Spočítat a otestovat vliv soupeře',key='opp_run16',disabled=incomplete):
                    st.session_state.pop('opponent_report16',None)
                    try:
                        from opponent import analyze as opponent_analyze
                        histories=[]
                        for y in range(yy-2,yy+1):
                            frame=nhl_data(f'{y}{y+1}')
                            if frame is not None: histories.append(frame)
                        if not histories: raise ValueError('Nejdříve obnov zálohu nebo načti NHL v Datovém centru.')
                        history=pd.concat(histories,ignore_index=True)
                        cache=st.session_state.get('batch_logs15',{})
                        ids=set(rows['ID']); raw={pid:v[1] for (year,pid),v in cache.items() if year==yy and pid in ids and not v[1][1]}
                        if not raw: raise ValueError('Historie hráčů není v paměti. Znovu spusť hromadnou analýzu.')
                        with st.spinner('Testuji vliv soupeře na dřívějších zápasech hráčů…'):
                            adjusted,summary,detail,context=opponent_analyze(history,raw,rows,day,h,a)
                        st.session_state['opponent_report16']=(key,adjusted,summary,detail,context,len(raw))
                    except Exception as e: st.warning('Experiment není dostupný: '+str(e))
                opp_report=st.session_state.get('opponent_report16')
                if opp_report and opp_report[0]==key:
                    _,adjusted,summary,detail,context,n_players=opp_report
                    st.caption(f'Test vychází z historií {n_players} hráčů vybraných sezonních soupisek, kteří mají načtené obě dostupné sezony. Není to nezávislá validace všech hráčů NHL. Výběr hráčů může zkreslit výsledek.')
                    st.dataframe(context.style.format({'GA/60 vyhlazené':'{:.2f}','Ligový průměr':'{:.2f}'}),hide_index=True)
                    view=adjusted if scope=='Oba týmy' else adjusted[adjusted['Tým']==scope]
                    col=market+' v1.6'; view=view.assign(Změna=view[col]-view[market]).sort_values(col,ascending=False)
                    st.dataframe(view[['Tým','Hráč',market,col,'Změna']].style.format({market:'{:.1%}',col:'{:.1%}','Změna':lambda x:f'{100*x:+.1f} p. b.'}),hide_index=True)
                    st.markdown('**Historický test na stejných hráčských zápasech**')
                    st.caption('Starších 70 % dostupných hracích dnů slouží k výběru váhy, novějších 30 % k testu. Nižší Brier a Log Loss jsou lepší. Váha 0 znamená žádnou korekci.')
                    st.dataframe(summary.style.format({'Brier':'{:.5f}','Log Loss':'{:.5f}','Beta':'{:.2f}'}),hide_index=True)
                    download_results('Stáhnout test i predikce soupeře (ZIP)',f'opponent_{day}_{h}_{a}.zip',('opponent_summary.csv',summary),('opponent_detail.csv',detail),('opponent_predictions.csv',adjusted),('opponent_context.csv',context))
            metadata=pd.DataFrame([{'datum':str(day),'domaci':h,'hoste':a,'model':'v1.2 vyhlazena cetnost','podmineno_ucasti':True,'rozsah':selection_mode,'vybrani_id':','.join(map(str,sorted(selected_ids or ())))}])
            download_results('Stáhnout celou analýzu (ZIP)',f'players_{day}_{h}_{a}.zip',('hraci.csv',rows),('vynechani.csv',errors),('informace.csv',metadata))
        st.stop()
    from players import TEAMS,season_start,roster,logs,profile,backtest as player_test
    st.subheader('Hráči NHL – body a góly')
    st.caption('Zdroj: veřejné webové rozhraní NHL. Žádný soubor MoneyPuck ani týmová historie nejsou nutné. Predikce platí pouze za předpokladu, že hráč nastoupí. Body a góly včetně prodloužení, bez nájezdů.')
    player_day=st.date_input('Datum hráčské analýzy',date.today(),key='player_day12')
    yy=season_start(player_day)
    team=st.selectbox('Tým hráče',TEAMS if yy>=2024 else sorted(set(TEAMS)-{'UTA'}|{'ARI'}),key='player_team12')
    @st.cache_data(ttl=1800,max_entries=64,show_spinner=False)
    def player_roster(team,yy):
        return roster(get(f'/roster/{team}/{yy}{yy+1}'))
    @st.cache_data(ttl=1800,max_entries=64,show_spinner=False)
    def player_logs(pid,yy):
        from requests import HTTPError
        payloads=[]; warnings=[]
        for y in (yy-1,yy):
            try: payloads.append(get(f'/player/{pid}/game-log/{y}{y+1}/2'))
            except HTTPError as e:
                if e.response is not None and e.response.status_code==404: warnings.append(f'{y}/{y+1}: záznamy nedostupné (404).')
                else: raise
        return logs(payloads),warnings
    try:
        names=player_roster(team,yy)
        pid=st.selectbox('Hráč',list(names),format_func=lambda x:names[x],key=f'player_id12_{team}_{yy}')
        st.caption('Sezonní soupiska API není potvrzenou sestavou na daný den; u starších dat nemusí odpovídat tehdejší příslušnosti hráče.')
        key=(pid,str(player_day))
        if st.button('Analyzovat hráče',key='player_run12'):
            try:
                with st.spinner('Načítám zápasy hráče…'):
                    data,warnings=player_logs(pid,yy)
                    p,table,past=profile(data,player_day)
                st.session_state['player_report12']=(key,p,table,past,data,warnings)
            except Exception as e: st.session_state.pop('player_report12',None); st.error(str(e))
        report=st.session_state.get('player_report12')
        if report and report[0]==key:
            _,p,table,past,data,warnings=report
            for warning in warnings: st.warning(warning)
            st.markdown(f'### {names[pid]}')
            a,b=st.columns(2)
            a.metric('Získá bod – ANO',f'{p["points"][0]:.1%}')
            b.metric('Získá bod – NE',f'{1-p["points"][0]:.1%}')
            st.metric('Vstřelí gól',f'{p["goals"][0]:.1%}')
            st.caption(f'Dlouhodobý vyhlazený odhad z {len(past)} zápasů. Poslední záznam {past.date.max().date()}.')
            if len(past)<30 or (pd.Timestamp(player_day)-past.date.max()).days>21:
                st.warning('Malý nebo zastaralý vzorek; odhad je zvlášť nejistý.')
            st.dataframe(table.style.format({'Body / zápas':'{:.2f}','Zápasy s bodem':'{:.1%}','Góly / zápas':'{:.2f}','Prům. čas (min)':'{:.1f}'}),hide_index=True)
            st.caption('Experiment s formou: bod '+f'{p["points"][1]:.1%}'+', gól '+f'{p["goals"][1]:.1%}'+'. Posledních 20 zápasů vyhlazených starší historií. Zatím bez ověřeného vlivu soupeře, xG, přesilovek a sestav.')
            if st.button('Historický test hráče',key='player_test12'):
                try:
                    summary,detail=player_test(data,player_day)
                    st.session_state['player_test_report12']=(key,summary,detail)
                except ValueError as e: st.info(str(e))
            tested=st.session_state.get('player_test_report12')
            if tested and tested[0]==key:
                _,summary,detail=tested
                st.dataframe(summary,hide_index=True)
                download_results('Stáhnout oba výsledky hráče (ZIP)',f'player_{pid}_results.zip',('player_summary.csv',summary),('player_detail.csv',detail))
                st.caption('Test jednotlivého hráče, jen předchozí zápasy, podmíněno účastí. Nejde o validaci celé NHL.')
    except Exception as e: st.error('Hráčská data: '+str(e))
    st.stop()
mode=st.selectbox('Soutěž / zdroj dat',['NHL API','Tipsport extraliga',*LEAGUES,'Vlastní CSV','Ruční scénář'])
d=None
mp=None
pro=False
elh_fixtures=[]
eu_fixtures=[]
if mode=='NHL API':
    year=date.today().year-(date.today().month<7)
    season=st.selectbox('Sezóna',[f'{y}{y+1}' for y in range(year,2020,-1)])
    if st.button('Stáhnout / aktualizovat NHL'):
        try:
            with st.spinner('Aktualizuji sezonu NHL…'):
                n,_=refresh_nhl(season)
            st.success(f'Uloženo {n} zápasů.')
        except Exception as e: st.error(str(e)+' Původní data ponechána.')
    try: d=nhl_data(season)
    except Exception as e: st.error('Historii nelze načíst: '+str(e)); d=None
    if d is not None: st.caption(f'Načteno {len(d)} zápasů, poslední datum {d.date.max().date()}')
    else: st.info('Klikni na stažení dat, obnov zálohu v Datovém centru nebo vyzkoušej ruční scénář.')
elif mode=='Tipsport extraliga':
    standings_panel(mode)
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
elif mode in LEAGUES:
    standings_panel(mode)
    st.caption('Samostatná historie '+mode+'. Automatický import výsledků této ligy zatím není ověřen; nahraj vlastní oprávněné CSV. Rozpis lze přidat v témže souboru s oběma skóre prázdnými.')
    ef=european_upload(mode)
    if ef:
        try:
            d,eu_fixtures=europe_history(ef)
            if not d.empty: st.caption(f'{len(d)} zápasů | {d.date.min().date()} – {d.date.max().date()} | {len(set(d.home)|set(d.away))} týmů.')
            else: st.info('Rozpis je načten, ale chybí dokončené výsledky.')
        except Exception as e: st.error(str(e))
    else: st.info('Zatím chybí historie. Bez ní model nebude generovat pravděpodobnosti.')
elif mode=='Vlastní CSV':
    st.caption('Sloupce date,home,away,hg,ag; skóre po 60 minutách, datum YYYY-MM-DD.')
    download_file('Stáhnout prázdnou šablonu CSV','date,home,away,hg,ag\n',file_name='sablona.csv')
    f=session_upload('Historické zápasy (CSV)','own14_session',types=['csv'])
    if f:
        try: d=validate(pd.read_csv(f))
        except Exception as e: st.error(str(e))
if mode=='Ruční scénář':
    st.warning('Ruční předpoklady, nikoliv predikce z ověřených dat.')
    h=st.text_input('Domácí','Hradec'); a=st.text_input('Hosté','Třinec')
    lh=st.number_input('Očekávané góly domácích',.1,12.,2.5,.1); la=st.number_input('Očekávané góly hostů',.1,12.,3.2,.1)
elif d is not None and not d.empty:
    with st.expander('Rozšíření historických dat (volitelné)'):
        extra=[] if mode in LEAGUES else session_upload('Starší sezony stejné soutěže, skóre po 60 minutách',{'NHL API':'extra_nhl14_session','Tipsport extraliga':'extra_elh14_session','Vlastní CSV':'extra_own14_session'}[mode],multiple=True,types=['csv'])
        if mode in LEAGUES: st.caption('Další sezony nahraj do stejného pole historie této ligy.')
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
            force_old=st.checkbox('Znovu stáhnout i již kompletní starší sezony',value=False)
            if st.button('Stáhnout vybrané starší sezony NHL'):
                for yr in years:
                    ss=f'{yr}{yr+1}'
                    try:
                        with st.spinner(f'Kontroluji / načítám {ss}…'):
                            n,skipped=refresh_nhl(ss,require_complete=True,force=force_old)
                        st.success(f'{ss}: {n} zápasů'+(' (již načteno)' if skipped else ''))
                    except Exception as e: st.error(f'{ss}: {e}')
            frames=[d]
            for yr in years:
                ss=f'{yr}{yr+1}'
                try:
                    old=nhl_data(ss)
                    if old is not None and complete(old,ss): frames.append(old)
                except Exception as e: st.warning(f'{ss}: {e}')
            history=merge_history(frames)
        else:
            history=merge_history([d])
            st.caption('Starší sezony této soutěže přidej výše pomocí CSV z oprávněného zdroje. Automatický archiv zatím není dostupný.')
        st.dataframe(history.assign(Sezona=season_year(history.date)).groupby('Sezona').size().rename('Zápasů').reset_index(),hide_index=True)
        download_file('Export historických dat CSV',history.to_csv(index=False),file_name='historie.csv')
        st.caption('Historie pro sezonní test a modely PRO. Základní Poisson používá vybranou sezonu.')
    teams=sorted(set(d.home)|set(d.away))
    cutoff=st.date_input('Datum predikce (datum zápasu dle zdroje)',date.today())
    past=d[d.date<pd.Timestamp(cutoff)]
    game=None
    fixtures=[]
    @st.cache_data(max_entries=16,show_spinner=False)
    def cached_elo(data): return elo(data)
    ratings=cached_elo(past)
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
    if mode in LEAGUES:
        fixtures=[(h,a) for day,h,a in eu_fixtures if day==cutoff.isoformat() and h in teams and a in teams]
        if fixtures: game=st.selectbox('Zápasy '+mode+' (domácí – hosté)',[None]+fixtures,format_func=lambda x:'Vlastní výběr týmů' if x is None else ' – '.join(x))
        else: st.caption('Pro tento den v nahraném rozpisu nejsou zápasy se známými týmy.')
        if not past.empty and (pd.Timestamp(cutoff)-past.date.max()).days>14: st.warning('Historie před zvoleným datem je starší než 14 dní.')
    if mode=='NHL API':
        st.subheader('Model PRO – NHL')
        pro=st.checkbox('Použít modely PRO (v0.8 výsledek / v0.7 góly)',value=page=='Analýza zápasu',key='pro14_analysis' if page=='Analýza zápasu' else 'pro14_lab')
        if page=='Validační laboratoř / brankáři': st.caption('PRO výpočet je zde ve výchozím stavu vypnutý, aby se zbytečně nepřepočítával před testem.')
        st.caption('Výběr modelů podle dosavadní retrospektivní validace. Nové zápasy teprve ověří skutečný přínos. Experimentální Elo platí jen pro základní režim.')
        mp=session_upload('MoneyPuck týmové statistiky (CSV nebo ZIP)','mp12_session')
        st.markdown('[MoneyPuck.com – týmová data](https://www.moneypuck.com/data.htm) · nekomerční použití. Bez aktuálního souboru lze použít náhradní v0.7 pro 1X2.')
        @st.cache_data(ttl=3600,show_spinner=False,max_entries=4)
        def dyn_live(hist,day,pairs):
            from hybrid import dynamic_forecast
            return dynamic_forecast(hist,day,pairs)
        @st.cache_data(ttl=3600,show_spinner=False,max_entries=4)
        def adv_live(hist,raw,name,day,pairs):
            import io
            from advanced import read_stats
            from hybrid import advanced_forecast
            f=io.BytesIO(raw); f.name=name
            return advanced_forecast(hist,read_stats(f),day,pairs)
    if fixtures and st.checkbox("Přehled všech zápasů dne (základní model)"):
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
@st.cache_data(max_entries=256,show_spinner=False)
def cached_predict(x,y): return predict(x,y)
p,totals,scores=cached_predict(lh,la)
model_1x2=model_goals='Základní Poisson / Elo' if d is not None and experimental else 'Základní Poisson'
pro_active=False
if pro:
    try:
        # Compute all fixtures plus selected pair in one cached run.
        pairs=tuple(sorted(set(fixtures+[(h,a)])))
        hist=history[history.date<pd.Timestamp(cutoff)]
        with st.spinner('Připravuji model PRO z dřívějších zápasů…'):
            dyn=dyn_live(hist,str(cutoff),pairs)
            adv=None
            if mp is not None:
                try: adv=adv_live(hist,mp.getvalue(),mp.name,str(cutoff),pairs)
                except Exception as e: st.warning('v0.8 není dostupná: '+str(e))
        i=pairs.index((h,a)); p=dyn['p'][i].tolist()
        model_1x2='v0.7 adaptivní (náhradní)'
        if adv is not None and (h,a) in adv['p']:
            p=adv['p'][h,a].tolist(); model_1x2='v0.8 xG + střely + odpočinek'
        else: st.info('Pro tento zápas chybí použitelné aktuální xG. Výsledek počítá náhradní model v0.7.')
        totals=list(zip((4.5,5.5,6.5),dyn['q'][i].tolist())); model_goals='v0.7 adaptivní'; pro_active=True
        st.caption(f"Gólový model: {dyn['n']} trénovacích zápasů, uzavřená historie do {dyn['train_end']}; poločas {dyn['kg'][0]} dní, váha regrese {dyn['kg'][1]:.0%}.")
        if adv is not None: st.caption(f"xG model: {adv['n']} trénovacích zápasů, historie do {adv['train_end']}, MoneyPuck do {adv['stats_end']}.")
        if fixtures:
            with st.expander('PRO – všechny zápasy dne'):
                batch=[]
                for ht,at in fixtures:
                    j=pairs.index((ht,at)); has_xg=adv is not None and (ht,at) in adv['p']; probs=adv['p'][(ht,at)] if has_xg else dyn['p'][j]
                    batch.append([ht+' – '+at,*[f'{v:.1%}' for v in probs],*[f'{v:.1%}' for v in dyn['q'][j]],'v0.8' if has_xg else 'v0.7'])
                st.dataframe(pd.DataFrame(batch,columns=['Zápas','1','X','2','O4.5','O5.5','O6.5','1X2 model']),hide_index=True)
    except Exception as e: st.warning('PRO není dostupné, zobrazuji původní model: '+str(e))
st.subheader(f'{h} – {a}')
st.caption(f"1X2: {model_1x2} | góly: {model_goals}")
st.dataframe(pd.DataFrame([[n,f'{v:.1%}',f'{1/v:.2f}' if v else '—'] for n,v in zip(['Domácí','Remíza','Hosté'],p)],columns=['Výsledek','Šance','Férový kurz']),hide_index=True,use_container_width=True)
if d is not None:
    st.caption(f'Elo po 60 minutách: {h} {ratings.get(h,1500):.0f} | {a} {ratings.get(a,1500):.0f}')
with st.expander('Vítěz včetně prodloužení – orientační scénář'):
    st.write(f'{h}: {p[0]+p[1]/2:.1%} | {a}: {p[2]+p[1]/2:.1%}')
    st.caption('Remízy rozděleny 50/50; nejde o naučený model prodloužení ani nájezdů.')
if not pro_active: st.write(f'Očekávané góly: {lh:.2f} + {la:.2f} = {lh+la:.2f}')
else: st.caption('Modely 1X2 a Over/Under jsou samostatné; nevytvářejí společné rozdělení skóre ani jednoznačný průměr gólů.')
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
with st.expander('Sestavy, brankáři a absence – poznámky'):
    from lineups import match_key, GROUPS, note_table
    lineup_day=cutoff if d is not None else st.date_input('Datum ručního scénáře',date.today(),key='lineup19_manual_day')
    lk=match_key(mode,lineup_day,h,a)
    st.caption('Platí pro tento zápas. Vyplň podle svého zdroje; poznámky nemění pravděpodobnosti. Uchovány během této návštěvy, export pro trvalé uložení.')
    ls=st.selectbox('Stav sestavy',['Nepotvrzená / pracovní','Potvrzená dle uživatele'],key='lineup19_status_'+lk)
    notes={}
    for i,team in enumerate((h,a)):
        st.markdown('**'+team+'**'); notes[team]={}
        for group in GROUPS:
            nk='lineup19_note_'+lk+'_'+str(i)+'_'+group
            store=st.session_state.setdefault('lineup19_notes',{})
            if nk not in st.session_state: st.session_state[nk]=store.get(nk,'')
            value=st.text_input(group,key=nk,placeholder='Jména odděl čárkou')
            store[nk]=value; notes[team][group]=value
    table=note_table(mode,lineup_day,h,a,notes,ls)
    if not table.empty:
        download_file('Stáhnout sestavy a poznámky CSV',table.to_csv(index=False),file_name='sestavy_'+lk+'.csv')
if not pro_active:
    st.subheader('Nejpravděpodobnější skóre'); st.dataframe(pd.DataFrame([(f'{i}:{j}',f'{v:.1%}') for i,j,v in scores],columns=['Skóre','Pravděpodobnost']),hide_index=True,use_container_width=True)
if not pro_active: st.caption('Nezávislý Poisson; domácí/venkovní síla s vyhlazením 8 ligovými zápasy. Volitelné experimentální Elo + forma.')
st.caption('Bez kalibrace brankářů, sestav a prodloužení. Férový kurz není doporučení k sázce.')
if d is not None and mode=='NHL API':
    from datetime import datetime, timezone
    st.subheader('Export predikce')
    st.caption('CSV si ulož do Souborů nebo na Disk. Cloudové uložení není trvalé. Export zaznamenává skutečný čas vytvoření a modely; starší nebo rozehrané zápasy nelze považovat za předzápasový test.')
    if st.button('Připravit záznam predikce'):
        now=datetime.now(timezone.utc); status='neověřený scénář'
        try:
            payload=get('/schedule/'+cutoff.isoformat()); matches=[g for w in payload['gameWeek'] for g in w['games'] if g.get('gameDate',w['date'])==cutoff.isoformat() and g['homeTeam']['abbrev']==h and g['awayTeam']['abbrev']==a and g.get('gameType')==2]
            if len(matches)==1 and matches[0].get('gameState') in ('FUT','PRE') and pd.Timestamp(matches[0]['startTimeUTC'])>pd.Timestamp(now): status='předzápasová (API ověřeno)'
        except Exception: pass
        rec={'created_utc':now.isoformat(),'date':cutoff.isoformat(),'home':h,'away':a,'status':status,'model_1x2':model_1x2,'model_goals':model_goals,'p1':p[0],'px':p[1],'p2':p[2],**{f'over_{k}':v for k,v in totals}}
        st.session_state['pro_export']=(cutoff,h,a,model_1x2,model_goals,tuple(p),tuple(totals),pd.DataFrame([rec]).to_csv(index=False),status)
    rec=st.session_state.get('pro_export')
    if rec and rec[:7]==(cutoff,h,a,model_1x2,model_goals,tuple(p),tuple(totals)):
        st.caption(rec[8]); download_file('Stáhnout záznam predikce CSV',rec[7],file_name=f'predikce_{cutoff}_{h}_{a}.csv')
    with st.expander('Vyhodnocení uložených předzápasových predikcí'):
        saved=st.file_uploader('Nahraj dříve uložené CSV predikcí',type='csv',accept_multiple_files=True,key='archive10')
        if saved:
            try:
                from hybrid import score_archive
                result,detail,matched,pending=score_archive(pd.concat([pd.read_csv(f) for f in saved],ignore_index=True),history,date.today())
                st.write(f'Vyhodnocené záznamy: {matched}; bez výsledku v načtené historii: {pending}.')
                if result.empty: st.info('Zatím žádné dokončené ověřené předzápasové predikce. Aktualizuj výsledky NHL.')
                else:
                    st.dataframe(result,hide_index=True)
                    download_file('Stáhnout vyhodnocení CSV',detail.to_csv(index=False),file_name='pro_vyhodnoceni.csv')
                st.caption('Vyhodnocení pouze po 60 minutách. První záznam pro kombinaci modelů a zápas; ruční CSV není nezměnitelný auditní log.')
            except Exception as e: st.error(str(e))

if d is not None and (mode in ('Tipsport extraliga','Vlastní CSV') or mode in LEAGUES):
    with st.expander('Multi-League PRO – vlastní test ligy',expanded=True):
        from leaguepro import evaluate as league_eval, forecast as league_forecast, fingerprint
        from multileague import clean
        hist=clean(history,cutoff)
        league_key=(mode,str(cutoff),fingerprint(hist))
        st.caption('Regrese s poločasy 90/180/365 dní a směsí s ligovým průměrem. Porovnání se základním Poissonem na stejných zápasech. Ladění před odděleným testem; vše po 60 minutách.')
        st.write(f'Historie: {len(hist)} zápasů před zvoleným datem.')
        if st.button('Ověřit model této ligy (PRO)',key='league_run17'):
            try:
                with st.spinner('Chronologický test ligy…'):
                    ls,ld,lc=league_eval(hist,cutoff)
                st.session_state['league_report17']=(league_key,ls,ld,lc)
            except Exception as e:
                st.session_state.pop('league_report17',None); st.info(str(e))
        report=st.session_state.get('league_report17')
        if report and report[0]==league_key:
            _,ls,ld,lc=report
            st.caption(f"Ladění: {lc['tuning']} | oddělený test: {lc['test']} od {lc['split']}. Váha regrese 1X2 {lc['w1']:.0%} ({lc['half1'] or 'bez regrese'} dní), góly {lc['wg']:.0%} ({lc['halfg'] or 'bez regrese'} dní).")
            st.dataframe(ls,hide_index=True,use_container_width=True)
            st.caption('Nižší Brier a Log Loss jsou lepší. Malý vzorek nestačí k potvrzení převahy; hlavní model zůstává beze změny.')
            try:
                pairs=list(dict.fromkeys([(h,a)]+[(x,y) for x,y in fixtures if min(((hist.home==x)|(hist.away==x)).sum(),((hist.home==y)|(hist.away==y)).sum())>=5]))
                @st.cache_data(max_entries=8,show_spinner=False)
                def league_predictions17(data,day,pairs,config):
                    return league_forecast(data,day,pairs,config)
                lp,lq=league_predictions17(hist,cutoff,tuple(pairs),lc)
                st.markdown('**Experimentální predikce této ligy – vybraný zápas**')
                st.dataframe(pd.DataFrame({'Výsledek':['Domácí','Remíza','Hosté'],'Pravděpodobnost':[f'{v:.1%}' for v in lp[0]]}),hide_index=True)
                st.dataframe(pd.DataFrame({'Hranice':['4.5','5.5','6.5'],'Over':[f'{v:.1%}' for v in lq[0]],'Under':[f'{1-v:.1%}' for v in lq[0]]}),hide_index=True)
                if len(pairs)>1:
                    with st.expander('Experimentální přehled všech dostupných zápasů dne'):
                        st.dataframe(pd.DataFrame([[x+' – '+y,*[f'{v:.1%}' for v in pp],*[f'{v:.1%}' for v in qq]] for (x,y),pp,qq in zip(pairs,lp,lq)],columns=['Zápas','1','X','2','O4.5','O5.5','O6.5']),hide_index=True)
                st.caption('Experimentální model přeučený na všech zápasech před zvoleným dnem.')
            except Exception as e: st.info(str(e))
            download_results('Stáhnout výsledky ligy (ZIP)','league_pro_results.zip',('league_summary.csv',ls),('league_detail.csv',ld),('league_config.csv',pd.DataFrame([lc])))

if page!='Validační laboratoř / brankáři':
    st.stop()

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
    with st.expander('Monte Carlo – kontrola základního Poissonova modelu'):
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
            download_file('Stáhnout výsledky testu CSV',report.to_csv(index=False),file_name='sezonni_test.csv')
        except Exception as e: st.error(str(e))

if d is not None:
    st.subheader('Dynamická validační laboratoř')
    st.caption('Přeučení každých 14 dní. Poločasy 30, 60, 90, 180 a 365 dní; kombinace s ligovým průměrem 0/50/100 %. Parametry pro 1X2 a góly vybíráme odděleně jen podle dřívějších predikcí. Tento laboratorní test nemění nastavení modelu PRO.')
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
        download_results('Stáhnout oba výsledky dynamického testu (ZIP)', 'dynamic_results.zip', ('dynamicky_test.csv',summary), ('dynamicke_predikce.csv',detail))
        st.caption('Retrospektivní vývojový test, nikoli nový nezávislý důkaz přesnosti. Nižší Brier a Log Loss jsou lepší.')

if d is not None and mode=='NHL API':
    st.subheader('Pokročilé statistiky NHL – v0.8 beta')
    st.markdown('Zdroj: [MoneyPuck.com](https://www.moneypuck.com/data.htm). Stáhni soubor **all game level data for all teams for all seasons** a nahraj jej níže. Pouze nekomerční použití.')
    st.caption('Model využívá průměry xG, střel pro a proti z posledních 15 utkání a dny odpočinku. Žádná statistika právě predikovaného dne se nepoužívá. Tento laboratorní test nemění nastavení modelu PRO.')
    st.caption('Používá soubor MoneyPuck nahraný výše v sekci PRO.')
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
        download_results('Stáhnout oba výsledky pokročilých statistik (ZIP)', 'advanced_results.zip', ('advanced_summary.csv',report), ('advanced_detail.csv',detail))
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
        download_results('Stáhnout oba výsledky speciálních formací (ZIP)', 'special_results.zip', ('special_summary.csv',summary), ('special_detail.csv',detail))
        st.caption('Všechny tři modely používají shodné zápasy a časové bloky. Tento laboratorní test nemění nastavení modelu PRO.')

    st.subheader('Individuální brankáři – v1.1 beta')
    st.markdown('Zdroj: [MoneyPuck.com – Game By Game Level Data](https://www.moneypuck.com/data.htm). Stáhni brankářské **Goalies** soubory pro aktuální a předchozí sezony, nikoli Season Level Data. Nahrát lze více CSV/ZIP najednou. Pouze nekomerční využití.')
    st.caption('Historický model odhaduje brankáře z jeho vytížení v předchozích pěti týmových zápasech, bez znalosti skutečně nastupujícího brankáře. Potvrzené sestavy nejsou v tomto zdroji dostupné před zápasem. PRO predikce a archiv zůstávají beze změny.')
    goalie_files=session_upload('MoneyPuck – zápasová data brankářů','goalie12_session',multiple=True)
    if goalie_files:
        try:
            import hashlib
            from goalies import read_goalies, GoalieBook, evaluate as goalie_evaluate, scenario as goalie_scenario
            @st.cache_data(show_spinner=False,max_entries=3)
            def goalie_load(payload):
                import io
                streams=[]
                for name,content in payload:
                    f=io.BytesIO(content); f.name=name; streams.append(f)
                return read_goalies(streams)
            goalie_data=goalie_load(tuple((f.name,f.getvalue()) for f in goalie_files))
            book=GoalieBook(goalie_data)
            day=pd.Timestamp(cutoff)
            st.caption(f'{len(goalie_data)} brankářských záznamů; {goalie_data.id.nunique()} brankářů; {goalie_data.date.min().date()} až {goalie_data.date.max().date()}.')
            st.caption('Výběr jen z hráčů s nedávnou historií v daném týmu, nejméně 5 utkání a 100 střelami; nejde o aktuální ověřenou soupisku.')
            hc=book.candidates(h,day); ac=book.candidates(a,day)
            if hc and ac:
                hid=st.selectbox(f'Brankář – {h}',hc,format_func=lambda i:book.profile(i,day)['name'],key=f'goalie11_h_{h}_{cutoff}')
                aid=st.selectbox(f'Brankář – {a}',ac,format_func=lambda i:book.profile(i,day)['name'],key=f'goalie11_a_{a}_{cutoff}')
                hp=book.profile(hid,day); ap=book.profile(aid,day)
                display=[]
                for label,key,fmt in [('Zápasy (max. 20)','n','d'),('Úspěšnost','sv','.2%'),('Úspěšnost posledních 5','sv5','.2%'),('Úspěšnost posledních 10','sv10','.2%'),('GSAx celkem','gsax','+.2f'),('GSAx / 60 min','gsax60','+.2f'),('GSAx posledních 5','gsax5','+.2f'),('GSAx posledních 10','gsax10','+.2f'),('Dny od posledního zápasu','rest','d'),('Starty / vystoupení za 14 dní','games14','d')]:
                    display.append([label,format(hp[key],fmt),format(ap[key],fmt)])
                st.dataframe(pd.DataFrame(display,columns=['Statistika',hp['name'],ap['name']]),hide_index=True)
                st.caption(f'Poslední dostupná data: {hp["last"]} / {ap["last"]}. GSAx = očekávané minus skutečně inkasované góly. Vystoupení zahrnují i střídání. Statistika max. 20 vystoupení za poslední rok.')
                if mp is not None and st.button('Spočítat experimentální scénář brankářů'):
                    try:
                        from advanced import read_stats
                        mp.seek(0); stats=read_stats(mp)
                        with st.spinner('Počítám scénář z uzavřených historických bloků…'):
                            tab,n=goalie_scenario(history,stats,goalie_data,day,h,a,hid,aid)
                        st.dataframe(tab.style.format({c:'{:.2%}' for c in tab.columns if c!='Scénář'}),hide_index=True)
                        st.caption(f'Trénink: {n} zápasů. Ruční výběr není nezávisle validovaný pro potvrzené startéry; scénář není součástí PRO ani archivu.')
                    except Exception as e: st.error(str(e))
            else: st.info('Pro jeden nebo oba týmy chybí dostatečně aktuální historie brankářů.')
            if mp is None: st.info('Pro modelový scénář a historický test nahraj také týmový MoneyPuck soubor výše.')
            else:
                goalie_key=(mp_key,tuple(hashlib.sha256(f.getbuffer()).hexdigest() for f in goalie_files))
                if st.button('Spustit historický test brankářů'):
                    try:
                        from advanced import read_stats
                        mp.seek(0); stats=read_stats(mp)
                        with st.spinner('Porovnávám modely na stejných zápasech…'):
                            st.session_state['goalie11_result']=goalie_evaluate(history[history.date<day],stats,goalie_data)
                            st.session_state['goalie11_key']=goalie_key
                    except Exception as e: st.session_state.pop('goalie11_result',None); st.error(str(e))
                if st.session_state.get('goalie11_key')==goalie_key and 'goalie11_result' in st.session_state:
                    summary,detail,n=st.session_state['goalie11_result']
                    st.write(f'Společný vzorek před trénovacím obdobím: {n} zápasů.')
                    st.dataframe(summary,hide_index=True)
                    download_results('Stáhnout oba výsledky brankářů (ZIP)', 'goalie_results.zip', ('goalie_summary.csv',summary), ('goalie_detail.csv',detail))
                    st.caption('Párový chronologický test v0.8 proti v1.1. Neověřuje skutečné potvrzené startéry ani přímo adaptivní model v0.7 pro góly.')
        except Exception as e: st.error('Brankářská data: '+str(e))
