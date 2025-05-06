#!/usr/bin/env python3
"""
Streamlit dashboard – live lineage & fitness explorer (≈ 70 LoC)
Run with:   streamlit run ui/lineage_app.py
"""
import json, os, sqlite3, time, pandas as pd, altair as alt, streamlit as st
ROOT = os.path.abspath(os.path.join(__file__, '..', '..', 'meta_agentic_agi_demo.sqlite'))

st.set_page_config(page_title="Meta-Agentic Lineage", layout="wide")
st.title("📈 Meta-Agentic α-AGI – Lineage Dashboard")

def load_df():
    con = sqlite3.connect(ROOT)
    df  = pd.read_sql("SELECT * FROM lineage", con)
    con.close()
    if df.empty:
        return df
    # explode fitness json
    fit = df['fitness'].apply(json.loads).apply(pd.Series)
    return pd.concat([df.drop(columns='fitness'), fit], axis=1)

placeholder = st.empty()
while True:
    df = load_df()
    if df.empty:
        st.info("Waiting for first generation…")
        time.sleep(2)
        st.experimental_rerun()

    latest = df.sort_values('gen').tail(1)
    st.subheader(f"Latest generation : {int(latest['gen'])}  |  Accuracy {latest['accuracy'].iat[0]:.3f}")

    c1, c2 = st.columns([2,1])
    with c1:
        base = alt.Chart(df).encode(x='gen:Q')
        chart = base.mark_line(point=True).encode(
            y=alt.Y('accuracy:Q', title='Accuracy'),
            tooltip=['gen','accuracy','latency','cost','carbon','novelty']
        )
        st.altair_chart(chart, use_container_width=True)
    with c2:
        st.dataframe(df[['gen','accuracy','latency','cost','carbon','rank']].sort_values('gen', ascending=False),
                     height=420)

    st.divider()
    st.write("### Source of top-ranked agents")
    for _, row in df.query("rank == 1").sort_values('gen').iterrows():
        st.markdown(f"**Gen {int(row.gen)}**   _acc {row.accuracy:.3f}_")
        st.code(row.code, language='python')

    time.sleep(4)
    placeholder.empty()   # refresh loop
