import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date
from model import validate,rates,predict,backtest
from nhl import load
st.set_page_config(page_title='Hockey Analytics Lab',page_icon='🏒',layout='centered')
st.title('🏒 Hockey Analytics Lab')
st.caption('v0.1 • Poisson • výsledky po 60 minutách')
mode=st.radio('Zdroj dat',['NHL API','Vlastní CSV','Ruční scénář'],horizontal=True)
d=None
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
    teams=sorted(set(d.home)|set(d.away)); h=st.selectbox('Domácí',teams); a=st.selectbox('Hosté',teams,index=min(1,len(teams)-1))
    cutoff=st.date_input('Datum predikce',date.today()); past=d[d.date<pd.Timestamp(cutoff)]
    try: lh,la=rates(past,h,a)
    except ValueError as e: st.info(str(e)); st.stop()
else: st.stop()
if h==a: st.error('Vyber dva různé týmy.'); st.stop()
p,totals,scores=predict(lh,la)
st.subheader(f'{h} – {a}')
for c,n,v in zip(st.columns(3),['Domácí','Remíza','Hosté'],p): c.metric(n,f'{v:.1%}'); c.caption(f'Férový kurz {1/v:.2f}' if v else '—')
st.bar_chart(pd.DataFrame({'Pravděpodobnost':p},index=['Domácí','Remíza','Hosté']))
st.write(f'Očekávané góly: {lh:.2f} + {la:.2f} = {lh+la:.2f}')
rows=[]
for k,v in totals:
    for name,q in [('Over',v),('Under',1-v)]: rows.append([f'{name} {k}',f'{q:.1%}',f'{1/q:.2f}' if q else '—'])
st.subheader('Gólové hranice po 60 minutách'); st.dataframe(pd.DataFrame(rows,columns=['Hranice','Pravděpodobnost','Férový kurz']),hide_index=True,use_container_width=True)
st.subheader('Nejpravděpodobnější skóre'); st.dataframe(pd.DataFrame([(f'{i}:{j}',f'{v:.1%}') for i,j,v in scores],columns=['Skóre','Pravděpodobnost']),hide_index=True,use_container_width=True)
st.caption('Nezávislý Poisson; u historických dat domácí/venkovní síla s vyhlazením 8 ligovými zápasy. Bez kalibrace brankářů, sestav a prodloužení. Férový kurz není doporučení k sázce.')
if d is not None and st.button('Historický test (walk-forward)'):
    try:
        with st.spinner('Probíhá test...'): n,b,l,acc=backtest(past)
        st.write(f'Zápasů: {n} | Brier: {b:.3f} | Log Loss: {l:.3f} | Přesnost: {acc:.1%}')
        st.caption('Test používá jen předchozí dny; nižší Brier a Log Loss jsou lepší. Zatím bez srovnávacího modelu.')
    except ValueError as e: st.info(str(e))
