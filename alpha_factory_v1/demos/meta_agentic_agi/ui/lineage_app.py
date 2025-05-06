# ui/lineage_app.py  – 60 LoC
import sqlite3, pandas as pd, streamlit as st, json, time, altair as alt, os
DB=os.path.join(os.path.dirname(__file__),'..','meta_agentic_agi_demo.sqlite')
st.set_page_config('Lineage – Meta-Agentic α-AGI',layout="wide")
while True:
    df=pd.read_sql('select * from lineage',sqlite3.connect(DB))
    if df.empty: st.warning('Waiting for first generation…'); time.sleep(1); st.experimental_rerun()
    df['fitness']=df['fitness'].apply(json.loads); exp=pd.json_normalize(df['fitness'])
    chart=alt.Chart(exp.assign(gen=df['gen'])).mark_line(point=True).encode(
        x='gen', y='accuracy', tooltip=['latency','cost','carbon','novelty'])
    st.altair_chart(chart,use_container_width=True)
    st.dataframe(df[['gen','ts','code']].sort_values('gen',ascending=False),height=500)
    time.sleep(4); st.experimental_rerun()
