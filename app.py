"""Streamlit dashboard.

The first tab is Data Health, not Stock Analysis. That ordering is
deliberate: if the data layer is degraded, every other tab is decorative.
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import streamlit as st

from mi.errors import DataUnavailable, ModelNotQualified
from mi.etf import (concentration_table, coverage_note, fetch_many,
                    overlap_matrix, shared_exposure)
from mi.indicators import indicator_pack
from mi.macro import (GOLD_PLAYBOOK, classify, framework_status, load_macro,
                      metals_panel, staleness_report)
from mi.ml import DirectionalModel, assemble
from mi.providers import default_router

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stock Intelligence", layout="wide")

START = "2015-01-01"
TODAY = dt.date.today().isoformat()


@st.cache_resource
def get_router():
    return default_router()


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(symbol: str):
    p = get_router().ohlcv(symbol, START, TODAY)
    return p.data, p.audit()


@st.cache_data(ttl=3600, show_spinner=False)
def get_macro():
    p = load_macro(get_router(), start=START)
    return p.data, staleness_report(p)


def failure_box(e: Exception, what: str):
    st.error(f"**{what} unavailable.** Nothing is being estimated or filled in.")
    st.code(str(e))


router = get_router()

st.sidebar.title("Stock Intelligence")
symbol = st.sidebar.text_input("Symbol", "MU").upper().strip()
horizon = st.sidebar.selectbox("Model horizon (days)", [3, 5, 10, 21], index=1)
st.sidebar.caption("Providers: " + ", ".join(p.name for p in router.providers) or "none configured")
if router.skipped:
    st.sidebar.warning("No key: " + ", ".join(router.skipped))

tabs = st.tabs(["Data Health", "Stock", "Macro & Metals", "Model", "ETF Overlap"])

# ---------------------------------------------------------------- health
with tabs[0]:
    st.header("Data Health")
    st.caption(
        "Every number in this app carries a provider and an as-of date. "
        "If a field cannot be sourced, the app says so instead of substituting."
    )
    h = router.health()
    c1, c2 = st.columns(2)
    c1.subheader("Configured")
    c1.write(h.ok or "none")
    c2.subheader("Unavailable")
    c2.write(h.failed or "none")

    try:
        _, macro_stale = get_macro()
        st.subheader("Macro series freshness")
        st.caption("Each series is judged against its own release cadence, not a global threshold.")
        st.dataframe(macro_stale, width="stretch")
    except DataUnavailable as e:
        failure_box(e, "Macro catalogue")

    if router.log:
        st.subheader("Provider attempt log")
        st.dataframe(router.attempt_table(), width="stretch")

# ---------------------------------------------------------------- stock
with tabs[1]:
    st.header(f"{symbol}")
    try:
        px, audit = get_prices(symbol)
        ind = indicator_pack(px)
        last = ind.iloc[-1]
        c = px["close"].iloc[-1]

        k = st.columns(6)
        k[0].metric("Close", f"{c:,.2f}")
        k[1].metric("RSI(14)", f"{last['rsi14']:.0f}")
        k[2].metric("vs EMA21", f"{last['dist_ema21']*100:+.1f}%")
        k[3].metric("vs SMA200", f"{last['dist_sma200']*100:+.1f}%")
        k[4].metric("RVOL(20)", f"{last['rvol20']:.2f}x")
        k[5].metric("Structure", {1: "Bullish", -1: "Bearish", 0: "None"}[int(last["trend"])])

        st.line_chart(px[["close"]].join(ind[["ema21", "ema50", "sma200"]]).tail(400))

        recent = ind.tail(60)
        events = []
        if recent["choch"].any():
            events.append(f"CHoCH on {recent.index[recent['choch']][-1].date()}")
        if recent["bos"].any():
            events.append(f"BOS on {recent.index[recent['bos']][-1].date()}")
        if recent["sweep_low"].any():
            events.append(f"Sell-side sweep on {recent.index[recent['sweep_low']][-1].date()}")
        if recent["sweep_high"].any():
            events.append(f"Buy-side sweep on {recent.index[recent['sweep_high']][-1].date()}")
        st.subheader("Structure events (60d)")
        st.write(events or "None")

        with st.expander("Provenance"):
            st.dataframe(audit, width="stretch")
    except DataUnavailable as e:
        failure_box(e, f"Price data for {symbol}")

# ---------------------------------------------------------------- macro
with tabs[2]:
    st.header("Macro & Metals")
    try:
        macro, _ = get_macro()
        call = classify(macro)
        c1, c2 = st.columns([1, 2])
        c1.metric("Regime", call.regime)
        c1.caption(f"Confidence: {call.confidence}")
        c2.write(GOLD_PLAYBOOK.get(call.regime, ""))
        st.json(call.detail)
        if call.missing:
            st.warning("Classified without: " + ", ".join(call.missing))

        st.subheader("Gold / silver")
        gold_sym = st.text_input("Gold proxy", "GLD")
        silver_sym = st.text_input("Silver proxy", "SLV")
        try:
            g, _ = get_prices(gold_sym)
            s, _ = get_prices(silver_sym)
            prices = pd.DataFrame({"gold": g["close"], "silver": s["close"]}).dropna()
            panel = metals_panel(prices, macro)
            fw = framework_status(panel)
            st.info(f"**Real-yield framework: {fw['status']}** — {fw['detail']}")
            latest = panel.dropna(how="all").iloc[-1]
            m = st.columns(4)
            m[0].metric("Gold/Silver ratio", f"{latest['gs_ratio']:.1f}")
            m[1].metric("G/S z(250d)", f"{latest.get('gs_z', float('nan')):.2f}")
            m[2].metric("Gold drawdown", f"{latest['gold_dd']*100:.1f}%")
            if "gold_cad" in panel:
                m[3].metric("Gold (CAD)", f"{latest['gold_cad']:,.0f}")
            st.line_chart(panel[["gs_ratio"]].tail(750))
            cols = [c for c in ("gold", "real_yield_10y") if c in panel]
            st.line_chart(panel[cols].tail(750))
        except DataUnavailable as e:
            failure_box(e, "Metals prices")
    except DataUnavailable as e:
        failure_box(e, "Macro data")

# ---------------------------------------------------------------- model
with tabs[3]:
    st.header("Directional model")
    st.caption(
        "This model is not allowed to show you a probability until it beats "
        "the base rate out of sample, across folds, and survives a block "
        "permutation test. If it fails, you get the reasons instead of a number."
    )
    if st.button(f"Train {horizon}-day model on {symbol}"):
        try:
            px, _ = get_prices(symbol)
            try:
                macro, _ = get_macro()
            except DataUnavailable:
                macro = None
                st.warning("Training without macro features — FRED unavailable.")
            X, y = assemble(px, macro, horizon=horizon)
            with st.spinner(f"Walk-forward across {len(X)} rows..."):
                model = DirectionalModel(horizon=horizon).fit(X, y)
            v = model.verdict()

            if model.qualified_:
                st.success(f"QUALIFIED — Brier skill {v['brier_skill']:.4f}, p={v['permutation_p']}")
                p = model.predict_proba(X.tail(1)).iloc[0]
                st.metric(f"P(up over {horizon}d, net of costs)", f"{p:.1%}")
                st.caption(
                    f"Reference: the model was right {v['base_rate']:.1%} of the time by base "
                    f"rate alone. Read this probability relative to that, not to 50%."
                )
            else:
                st.error("NOT QUALIFIED — no probability will be shown.")
                for r in v["reasons"]:
                    st.write(f"- {r}")

            c1, c2 = st.columns(2)
            c1.subheader("Out-of-sample scorecard")
            c1.json({k: v[k] for k in v if k != "reasons"})
            c2.subheader("Calibration")
            c2.dataframe(model.reliability_, width="stretch")
            st.subheader("Hit rate by predicted decile")
            st.dataframe(model.edge_summary(), width="stretch")
        except (DataUnavailable, ModelNotQualified, ValueError) as e:
            failure_box(e, "Model training")

# ---------------------------------------------------------------- etf
with tabs[4]:
    st.header("ETF overlap")
    raw = st.text_input("ETFs (comma separated)", "SMH, SOXX, QQQ")
    etfs = [e.strip().upper() for e in raw.split(",") if e.strip()]
    for n in coverage_note(etfs):
        st.warning(n)
    if st.button("Analyse"):
        holdings, failures = fetch_many(router, etfs)
        for etf, why in failures.items():
            st.error(f"{etf}: {why}")
        if len(holdings) >= 1:
            st.subheader("Concentration")
            st.dataframe(concentration_table(holdings), width="stretch")
        if len(holdings) >= 2:
            st.subheader("Shared weight")
            st.caption(
                "Fraction of capital genuinely duplicated if held at equal size. "
                "Shared name counts are shown separately because they overstate overlap."
            )
            st.dataframe(overlap_matrix(holdings, "weight"), width="stretch")
            st.subheader("Shared names (Jaccard)")
            st.dataframe(overlap_matrix(holdings, "name"), width="stretch")
            st.subheader("Look-through exposure (equal dollar weight)")
            st.dataframe(shared_exposure(holdings), width="stretch")
