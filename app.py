import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date
from model import validate,rates,predict,backtest
from nhl import load, get
st.set_page_config(page_title='Hockey Analytics Lab',page_icon='🏒',layout='centered')
st.markdown('## 🏒 Hockey Analytics Lab')
st.caption('v0.3 • ELH Beta • Poisson • výsledky po 60 minutách')
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
    teams=sorted(set(d.home)|set(d.away))
    cutoff=st.date_input('Datum predikce (datum zápasu dle zdroje)',date.today())
    past=d[d.date<pd.Timestamp(cutoff)]
    game=None
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
    if game: h,a=game
    else:
        h=st.selectbox('Domácí',teams); a=st.selectbox('Hosté',teams,index=min(1,len(teams)-1))
    try: lh,la=rates(past,h,a)
    except ValueError as e: st.info(str(e)); st.stop()
else: st.stop()
if h==a: st.error('Vyber dva různé týmy.'); st.stop()
p,totals,scores=predict(lh,la)
st.subheader(f'{h} – {a}')
st.dataframe(pd.DataFrame([[n,f'{v:.1%}',f'{1/v:.2f}' if v else '—'] for n,v in zip(['Domácí','Remíza','Hosté'],p)],columns=['Výsledek','Šance','Férový kurz']),hide_index=True,use_container_width=True)
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
        st.caption('Forma je zatím informativní a nemění výpočet.')
with st.expander('Potvrzení brankáři (poznámka)'):
    st.text_input('Brankář domácích',key='goalie_home'); st.text_input('Brankář hostů',key='goalie_away')
    st.caption('Jména jsou pouze poznámka; bez ověřených statistik nemění pravděpodobnosti.')
st.subheader('Nejpravděpodobnější skóre'); st.dataframe(pd.DataFrame([(f'{i}:{j}',f'{v:.1%}') for i,j,v in scores],columns=['Skóre','Pravděpodobnost']),hide_index=True,use_container_width=True)
st.caption('Nezávislý Poisson; u historických dat domácí/venkovní síla s vyhlazením 8 ligovými zápasy. Bez kalibrace brankářů, sestav a prodloužení. Férový kurz není doporučení k sázce.')
if d is not None and st.button('Historický test (walk-forward)'):
    try:
        with st.spinner('Probíhá test...'): n,b,l,acc=backtest(past)
        st.write(f'Zápasů: {n} | Brier: {b:.3f} | Log Loss: {l:.3f} | Přesnost: {acc:.1%}')
        st.caption('Test používá jen předchozí dny; nižší Brier a Log Loss jsou lepší. Zatím bez srovnávacího modelu.')
    except ValueError as e: st.info(str(e))
