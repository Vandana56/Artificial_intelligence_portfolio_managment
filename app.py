"""Streamlit frontend. Run: streamlit run app.py"""
import pandas as pd
import streamlit as st
from backend.service import run_analysis

st.set_page_config(page_title="AI Portfolio Manager", page_icon="📈", layout="wide")
st.title("📈 AI Financial Portfolio Management")

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("Portfolio")
    tickers = [t.strip().upper() for t in
               st.text_input("Tickers (comma separated)", "AAPL,MSFT,GOOGL,JPM,XOM").split(",") if t.strip()]
    years = st.slider("History (years)", 1, 5, 3)
    rf = st.number_input("Risk-free rate", 0.0, 0.2, 0.04, 0.005)
    live = st.checkbox("Use live Yahoo Finance data", True)
    st.caption("Weights (%)")
    raw_w = [st.number_input(t, 0.0, 100.0, round(100 / len(tickers), 2)) for t in tickers]
    go = st.button("Analyse", type="primary")

if len(tickers) < 2:
    st.warning("Enter at least 2 tickers."); st.stop()

@st.cache_data(show_spinner="Crunching data and training models...")
def cached(tk, w, y, r, l):
    return run_analysis(list(tk), list(w), y, r, l)

if go or "res" not in st.session_state:
    if sum(raw_w) == 0:
        st.error("Weights must sum to more than 0."); st.stop()
    st.session_state.res = cached(tuple(tickers), tuple(raw_w), years, rf, live)
res = st.session_state.res
st.caption(f"Data source: {res['source']} · Model: {res['model_name']}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk", "AI Forecast", "Optimizer"])

with tab1:
    p = res["portfolio"]
    c = st.columns(4)
    c[0].metric("Annual Return", f"{p['Annual Return']:.1%}")
    c[1].metric("Volatility", f"{p['Volatility']:.1%}")
    c[2].metric("Sharpe", f"{p['Sharpe']:.2f}")
    c[3].metric("Max Drawdown", f"{p['Max Drawdown']:.1%}")
    st.subheader("Portfolio growth (base 100)")
    st.line_chart(res["portfolio_curve"])
    st.subheader("Allocation")
    st.bar_chart(res["weights"])

with tab2:
    st.subheader("Per-asset risk & AI risk tier (KMeans)")
    st.dataframe(res["per_asset"].style.format(
        {c: "{:.2%}" for c in res["per_asset"].columns if c not in ("Sharpe", "Risk Tier")} | {"Sharpe": "{:.2f}"}))
    c = st.columns(2)
    c[0].metric("Portfolio VaR 95% (1d)", f"{res['portfolio']['VaR 95% (1d)']:.2%}")
    c[1].metric("Portfolio CVaR 95% (1d)", f"{res['portfolio']['CVaR 95% (1d)']:.2%}")
    st.subheader("Correlation")
    st.dataframe(res["corr"].style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1).format("{:.2f}"))

with tab3:
    st.subheader("5-day direction forecast")
    rows = [{"Ticker": t, "Prob. Up": f["prob_up"], "Signal": "Bullish" if f["prob_up"] > .55 else
             "Bearish" if f["prob_up"] < .45 else "Neutral", "Test Accuracy": f["accuracy"]}
            for t, f in res["forecasts"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Ticker").style.format({"Prob. Up": "{:.1%}", "Test Accuracy": "{:.1%}"}))
    pick = st.selectbox("Feature importance for", list(res["forecasts"]))
    st.bar_chart(res["forecasts"][pick]["importance"])
    st.caption("Educational only. Accuracy near 50% is normal for markets; this is not financial advice.")

with tab4:
    st.subheader("Efficient frontier")
    st.scatter_chart(res["frontier"], x="Volatility", y="Return", color="Sharpe")
    c = st.columns(2)
    for col, key, name in ((c[0], "max_sharpe", "Max Sharpe"), (c[1], "min_vol", "Min Volatility")):
        o = res[key]
        col.markdown(f"**{name}** — Return {o['return']:.1%}, Vol {o['vol']:.1%}, Sharpe {o['sharpe']:.2f}")
        col.bar_chart(o["weights"])
