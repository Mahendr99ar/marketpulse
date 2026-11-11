"""MarketPulse demo app.

Run after the pipeline:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from marketpulse.config import load_config
from marketpulse.generation.brief import generate_brief
from marketpulse.generation.facts import fmt_change, fmt_level
from marketpulse.generation.writers import get_writer
from marketpulse.pipeline import Store, _retriever, make_facts
from marketpulse.regimes import regime_names

st.set_page_config(page_title="MarketPulse", layout="wide")
cfg = load_config()

dataset = st.sidebar.radio("Dataset", ["demo", "real"], help="demo = synthetic data")
backend = st.sidebar.selectbox("Brief writer", ["template", "ollama", "groq", "hf"])
store = Store(cfg, dataset)


@st.cache_resource(show_spinner="Building demo data (about 30 seconds, first visit only)")
def ensure_demo_data() -> bool:
    """On a fresh server (e.g. Streamlit Cloud) the demo files do not exist yet."""
    from marketpulse import pipeline

    demo = Store(cfg, "demo")
    expected = [i.id for i in cfg.instruments]
    stale = demo.exists("levels") and list(demo.load("levels").columns) != expected
    if stale or not demo.exists("flags_regime"):  # missing, or built by an older version
        pipeline.step_data(cfg, "demo")
        pipeline.step_regimes(cfg, "demo")
        pipeline.step_detect(cfg, "demo")
    return True


if dataset == "demo":
    ensure_demo_data()
try:
    levels = store.load("levels")
    proba = store.load("regime_proba")
    flags = store.load("flags_regime")
    news = store.load("news")
except FileNotFoundError as e:
    st.error(f"{e}\n\nRun: python -m marketpulse.cli all {'--demo' if dataset == 'demo' else ''}")
    st.stop()

if dataset == "demo":
    st.warning("Synthetic data. Nothing on this page describes real markets.")


@st.cache_resource
def retriever_for(ds: str):
    return _retriever(cfg, Store(cfg, ds).load("news"))


names = regime_names(proba.filter(like="p_").shape[1])
colors = ["#2a9d8f", "#e9c46a", "#e76f51", "#6d597a", "#264653"]

st.title("MarketPulse")
st.caption("Out-of-sample market regimes, unusual moves, and fact-checked daily briefs")

# ---- regime chart ----
spx = levels["SPX"].loc[proba.index]
fig = go.Figure()
for i, name in enumerate(names):
    mask = proba["regime"] == i
    fig.add_trace(go.Scatter(x=spx.index[mask], y=spx[mask], mode="markers",
                             marker=dict(size=3, color=colors[i]), name=name))
fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                  title="S&P 500 coloured by filtered regime (no look-ahead)")
st.plotly_chart(fig, width="stretch")

# ---- pick a day ----
flag_days = flags.any(axis=1)
flag_days = flag_days[flag_days].index
default = flag_days.max() if len(flag_days) else proba.index.max()
day = st.sidebar.date_input("Date", value=default.date(),
                            min_value=proba.index.min().date(), max_value=proba.index.max().date())
only_flagged = st.sidebar.checkbox("Jump to nearest day with an unusual move", value=True)
day = pd.Timestamp(day)
if only_flagged and len(flag_days):
    day = flag_days[abs(flag_days - day).argmin()]
elif day not in proba.index:
    day = proba.index[proba.index.get_indexer([day], method="pad")[0]]

facts = make_facts(cfg, store, day, retriever_for(dataset))
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader(f"{facts.date}")
    st.metric("Regime", facts.regime, f"{facts.regime_probability}% probability", delta_color="off")
    st.dataframe(pd.DataFrame([{
        "market": m.name, "level": fmt_level(m), "change": fmt_change(m),
        "z (regime)": m.z, "unusual": "yes" if m.flagged else ""} for m in facts.moves]),
        hide_index=True, width="stretch")
    for iid, items in facts.news.items():
        if items:
            st.markdown(f"**News for {cfg.instrument(iid).name}**")
            for n in items:
                st.markdown(f"- {n.title} ({n.source})")

with c2:
    st.subheader("Daily brief")
    if st.button("Generate brief", type="primary"):
        with st.spinner("Writing and fact-checking"):
            try:
                res = generate_brief(facts, get_writer(cfg, backend), cfg)
            except Exception as e:  # missing Ollama, API key, etc.
                st.error(f"Writer '{backend}' failed: {e}")
                st.stop()
        if res["status"] == "passed":
            st.success(f"Passed fact check (attempt {res['attempts']})")
        else:
            st.error("Failed fact check after retries: needs human review")
        st.write(res["draft"])
        with st.expander("Attempts and fact-check feedback"):
            for h in res["history"]:
                st.markdown(f"**Attempt {h['attempt']}**: number accuracy {h['number_accuracy']:.0%}")
                st.text(h["draft"])
                if h["problems"]:
                    st.code(h["problems"])

st.divider()
q = st.text_input("Search the news archive")
if q and retriever_for(dataset) is None:
    st.info("No news downloaded yet. Run: python -m marketpulse.cli news")
elif q:
    st.dataframe(retriever_for(dataset).search_all(q, k=10)[["date", "title", "source", "score"]],
                 hide_index=True, width="stretch")
