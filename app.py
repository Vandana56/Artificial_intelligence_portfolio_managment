"""Streamlit frontend. Run: python -m streamlit run app.py"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from backend.service import run_analysis

st.set_page_config(page_title="AI Portfolio Manager", page_icon="📈", layout="wide")
# hide the +/- stepper buttons on number inputs (type values instead)
st.markdown(
    "<style>[data-testid='stNumberInputStepUp'],[data-testid='stNumberInputStepDown']{display:none;}</style>",
    unsafe_allow_html=True,
)
st.title("📈 AI Financial Portfolio Management")

# ---------------- Sidebar inputs ----------------
with st.sidebar:
    st.header("Portfolio")
    tickers = [t.strip().upper() for t in
               st.text_input("Tickers (comma separated)", "AAPL,MSFT,GOOGL,JPM,XOM").split(",") if t.strip()]
    years = st.slider("History (years)", 1, 5, 3)
    rf_pct = st.number_input("Risk-free rate (%)", 0.0, 20.0, 4.0, 0.25, format="%.2f")
    rf = rf_pct / 100
    live = st.checkbox("Use live Yahoo Finance data", True)
    st.caption("Weights (%) - rescaled to 100% automatically")
    raw_w = [st.number_input(t, 0.0, 100.0, round(100 / max(len(tickers), 1), 1), step=1.0, format="%.1f") for t in tickers]
    go = st.button("Analyse", type="primary")

if len(tickers) < 2:
    st.warning("Enter at least 2 tickers."); st.stop()


@st.cache_data(show_spinner="Downloading prices, training models...")
def cached(tk, w, y, r, l):
    return run_analysis(list(tk), list(w), y, r, l)


if go or "res" not in st.session_state:
    if sum(raw_w) == 0:
        st.error("Weights must sum to more than 0."); st.stop()
    st.session_state.res = cached(tuple(tickers), tuple(raw_w), years, rf, live)
res = st.session_state.res
st.caption(f"Data source: **{res['source']}** · Model: **{res['model_name']}** · "
           f"{len(res['returns'])} trading days analysed")

tab1, tab2, tab3, tab4= st.tabs(["Overview", "Risk", "AI Forecast", "Optimizer"])

# ---------------- Overview ----------------
with tab1:
    p = res["portfolio"]
    c = st.columns(4)
    c[0].metric("Annual Return", f"{p['Annual Return']:.1%}")
    c[1].metric("Volatility", f"{p['Volatility']:.1%}")
    c[2].metric("Sharpe", f"{p['Sharpe']:.2f}")
    c[3].metric("Max Drawdown", f"{p['Max Drawdown']:.1%}")
    s = p["Sharpe"]
    verdict = "good" if s > 1 else "moderate" if s > 0.5 else "weak"
    st.info(f"Over the period, your portfolio earned about **{p['Annual Return']:.1%}** a year with "
            f"**{p['Volatility']:.1%}** volatility. Risk-adjusted return (Sharpe {s:.2f}) is **{verdict}**. "
            f"Its worst peak-to-low fall was **{p['Max Drawdown']:.1%}**.")
    st.subheader("Portfolio growth (starting value = 100)")
    st.line_chart(res["portfolio_curve"])
    st.subheader("Allocation (%)")
    st.bar_chart(res["weights"] * 100)

# ---------------- Risk ----------------
with tab2:
    st.subheader("Per-asset risk & AI risk tier (KMeans)")
    pa = res["per_asset"]
    fmt = {c: "{:.2%}" for c in pa.columns if c not in ("Sharpe", "Risk Tier", "Risk Score")}
    fmt["Sharpe"] = "{:.2f}"
    fmt["Risk Score"] = "{:.2f}"
    st.dataframe(pa.style.format(fmt))
    st.caption("Risk Score = average z-score of volatility, max drawdown and VaR. Tiers are relative to the stocks you entered, not absolute.")
    c = st.columns(2)
    c[0].metric("Portfolio VaR 95% (1 day)", f"{res['portfolio']['VaR 95% (1d)']:.2%}",
                help="Loss exceeded on only the worst 5% of days")
    c[1].metric("Portfolio CVaR 95% (1 day)", f"{res['portfolio']['CVaR 95% (1d)']:.2%}",
                help="Average loss on those worst 5% of days")
    st.subheader("Correlation")
    corr = res["corr"]
    st.dataframe(corr.style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1).format("{:.2f}"))
    pairs = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    hi, lo = pairs.idxmax(), pairs.idxmin()
    st.info(f"Most similar pair: **{hi[0]} & {hi[1]}** ({pairs.max():.2f}). "
            f"Best diversifier pair: **{lo[0]} & {lo[1]}** ({pairs.min():.2f}). "
            "Lower correlation = better diversification.")

# ---------------- AI Forecast ----------------
with tab3:
    st.subheader("5-day direction forecast")
    st.caption("Predicts only whether each stock will be HIGHER in 5 trading days (yes/no). "
               "Scored on the final 20% of history, which the model never saw, with a 5-day gap "
               "between training and test data so labels cannot overlap.")
    rows = []
    for t, f in res["forecasts"].items():
        m, pu = f["metrics"], f["prob_up"]
        rows.append({"Ticker": t, "Prob. Up": pu,
                     "Signal": "Bullish" if pu > .55 else "Bearish" if pu < .45 else "Neutral",
                     "Accuracy": m["accuracy"], "Baseline": m["baseline"],
                     "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                     "Walk-fwd Acc.": f["cv"]["acc_mean"],
                     "Beats baseline?": "Yes" if m["accuracy"] - m["baseline"] > 0.03 else "No"})
    pct = {c: "{:.1%}" for c in ["Prob. Up", "Accuracy", "Baseline", "Precision", "Recall", "F1", "Walk-fwd Acc."]}
    st.dataframe(pd.DataFrame(rows).set_index("Ticker").style.format(pct, na_rep="-"))
    st.caption("Baseline = accuracy of always guessing the more common direction. A model only has "
               "an edge if it beats it. Walk-fwd Acc. = average accuracy over 5 time-ordered folds.")
    wavg = sum(res["weights"][t] * f["prob_up"] for t, f in res["forecasts"].items())
    st.metric("Weighted outlook (your weights x each stock's Prob. Up)", f"{wavg:.1%}",
              help="Average of the stock signals, not a separate model")
    st.warning("A confident Prob. Up with weak out-of-sample scores means the model is often wrong on "
               "unseen data. Treat signals as a learning tool, not trading advice.")
    pick = st.selectbox("Details for", list(res["forecasts"]))
    f = res["forecasts"][pick]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion matrix (test set)**")
        st.dataframe(pd.DataFrame(f["confusion"], index=["Actual Down", "Actual Up"],
                                  columns=["Pred Down", "Pred Up"]))
        st.caption(f"Train rows: {f['n_train']} · gap: {f['horizon']} · test rows: {f['n_test']} · "
                   f"AUC: {f['metrics']['auc']:.2f} · walk-fwd F1: {f['cv']['f1_mean']:.2f}")
    with c2:
        st.markdown("**Feature importance**")
        st.bar_chart(f["importance"])

# ---------------- Optimizer ----------------
with tab4:
    ms, mv, p = res["max_sharpe"], res["min_vol"], res["portfolio"]
    st.subheader("Efficient frontier")
    cloud, line = res["frontier"]["random"], res["frontier"]["frontier"]
    marks = pd.DataFrame({"Volatility": [p["Volatility"], ms["vol"], mv["vol"]],
                          "Return": [p["Annual Return"], ms["return"], mv["return"]],
                          "Portfolio": ["Current", "Max Sharpe", "Min Volatility"]})
    xs = alt.X("Volatility:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="%"))
    ys = alt.Y("Return:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="%"))
    chart = (
        alt.Chart(cloud).mark_circle(size=14, opacity=0.35).encode(
            x=xs, y=ys, color=alt.Color("Sharpe:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["Volatility:Q", "Return:Q", "Sharpe:Q"])
        + alt.Chart(line).mark_line(color="red", strokeWidth=3).encode(x=xs, y=ys)
        + alt.Chart(marks).mark_point(size=180, filled=True, stroke="black").encode(
            x=xs, y=ys, color=alt.Color("Portfolio:N"), tooltip=["Portfolio:N", "Volatility:Q", "Return:Q"])
    ).resolve_scale(color="independent")
    st.altair_chart(chart, use_container_width=True)
    st.caption("Dots = 5000 random long-only portfolios. Red line = true efficient frontier "
               "(lowest risk for each target return).")
    st.subheader("Current vs optimised")
    summ = pd.DataFrame({
        "Current": [p["Annual Return"], p["Volatility"], p["Sharpe"]],
        "Max Sharpe": [ms["return"], ms["vol"], ms["sharpe"]],
        "Min Volatility": [mv["return"], mv["vol"], mv["sharpe"]]},
        index=["Annual Return", "Volatility", "Sharpe"])
    show = summ.copy().astype(object)
    for col in summ.columns:
        show[col] = [f"{summ[col].iloc[0]:.1%}", f"{summ[col].iloc[1]:.1%}", f"{summ[col].iloc[2]:.2f}"]
    st.dataframe(show)
    st.subheader("Suggested weights")
    cmp = pd.DataFrame({"Current": res["weights"], "Max Sharpe": ms["weights"], "Min Volatility": mv["weights"]})
    cmp["Change to reach Max Sharpe"] = cmp["Max Sharpe"] - cmp["Current"]
    st.dataframe(cmp.style.format("{:.1%}"))
    st.caption("Optimised on past data only; past performance does not guarantee future results.")

