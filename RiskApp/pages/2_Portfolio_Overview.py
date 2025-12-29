import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf

# Page Config
from helpers import add_big_sidebar_logo
add_big_sidebar_logo(width_px=220,)
st.set_page_config(page_title="Portfolio Overview", layout="wide")
st.title("Portfolio Overview")

#Guard and session definitions

if "tickers" not in st.session_state or len(st.session_state.tickers) == 0:
    st.warning("No portfolio found. Go to **Portfolio Builder** first.")
    st.stop()

tickers = st.session_state.tickers
start_bldr = st.session_state.get("start", pd.Timestamp("2018-01-01").date())
end_bldr   = st.session_state.get("end", pd.Timestamp.today().date())
rf         = float(st.session_state.get("rf", 0.04))
sym        = st.session_state.get("currency_symbol", "£")
invest_amt = float(st.session_state.get("portfolio_value_base", 1_000_000.0))

units_dict = st.session_state.get("investment_units")          # {ticker: units}
units_start_date = st.session_state.get("investment_common_start")  # date
start_px_map = st.session_state.get("investment_start_prices") # {ticker: price at common start}

# Fallback to equal weights if none
weights_dict = st.session_state.get("weights", {})
w = np.array([weights_dict.get(t, 0.0) for t in tickers], dtype=float)
if w.sum() == 0:
    w = np.ones(len(tickers))/len(tickers)
else:
    w = w / w.sum()

# Load price data
@st.cache_data(show_spinner=True)
def load_prices(tickers, start, end):
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    df = yf.download(
        tickers, start=str(s), end=str(e), auto_adjust=True,
        progress=False, threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna(how="all")

# If we have a common start from Builder, start from there (so the value series begins at the buy date)
if units_start_date:
    true_start = max(pd.Timestamp(units_start_date).date(), pd.Timestamp(start_bldr).date())
else:
    true_start = pd.Timestamp(start_bldr).date()

prices = load_prices(tickers, true_start, end_bldr)
if prices.empty:
    st.error("No price data returned. Try expanding the date range.")
    st.stop()

# Keep only columns that exist in prices
tickers_avail = [t for t in tickers if t in prices.columns]
if len(tickers_avail) == 0:
    st.error("None of the tickers have price data in the selected period.")
    st.stop()

prices = prices[tickers_avail]

# VALUE SERIES (Preferred: units-based from Builder)
have_units = isinstance(units_dict, dict) and len(units_dict) > 0 and units_start_date is not None

if have_units:
    # intersect units with available tickers
    units = {t: float(units_dict[t]) for t in tickers_avail if t in units_dict}
    if len(units) == 0:
        have_units = False

if have_units:
    # portfolio value through time
    value_series = (prices[units.keys()] * pd.Series(units)).sum(axis=1)
    # align to start
    value_series = value_series.dropna()
    # daily returns from valued series
    port_rets = value_series.pct_change().dropna()
    # derived curves
    curve_norm = value_series / value_series.iloc[0]  # "£1 invested" curve
else:
    # Fallback: returns via weights (no money notionals)
    rets = prices.pct_change().dropna()
    port_rets = (rets @ (w[:len(rets.columns)] if len(w) == len(tickers) else np.ones(len(rets.columns))/len(rets.columns))).dropna()
    # Build a synthetic value series starting at invest_amt for visualization
    curve_norm = (1 + port_rets).cumprod()
    value_series = invest_amt * curve_norm

# 3) METRICS

def _infer_periods_per_year(dt_index: pd.DatetimeIndex) -> int:
    if not isinstance(dt_index, pd.DatetimeIndex) or len(dt_index) < 3:
        return 252
    gaps = pd.Series(dt_index).diff().dt.days.dropna()
    if gaps.empty:
        return 252
    med = float(gaps.median())
    if med <= 2:
        return 252
    if med <= 8:
        return 52
    if med <= 20:
        return 12
    return max(1, int(round(365 / max(1.0, med))))

_ppyear = _infer_periods_per_year(port_rets.index)
ann_ret = port_rets.mean() * _ppyear
ann_vol = port_rets.std(ddof=1) * np.sqrt(_ppyear)
sharpe  = (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan

# Max drawdown on the value curve
curve_val = value_series.copy()
peak = curve_val.cummax()
mdd = float((curve_val/peak - 1).min())

total_return = (curve_val.iloc[-1] / curve_val.iloc[0]) - 1
# CAGR over years
n_days = (curve_val.index[-1] - curve_val.index[0]).days
years = max(n_days/365.25, 1e-9)
cagr = (curve_val.iloc[-1]/curve_val.iloc[0])**(1/years) - 1

# 4) DISPLAY METRICS

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Value", f"{sym}{curve_val.iloc[-1]:,.0f}")
c2.metric("Total Return", f"{total_return:.2%}")
c3.metric("CAGR", f"{cagr:.2%}")
c4.metric("Sharpe", f"{sharpe:.2f}")

# 5) CHARTS

st.divider()
st.subheader(f"Portfolio Value — {sym}{invest_amt:,.0f} invested on start date")

# Show actual currency value curve
st.line_chart(curve_val.rename("Portfolio Value"), use_container_width=True)

# 6) PER-TICKER BREAKDOWN (weights % + money + units + P/L)
st.divider()
st.subheader("Per-Ticker Breakdown")

if have_units:
    # Current prices, start prices
    last_px = prices.iloc[-1]
    start_px_series = pd.Series({t: start_px_map.get(t, np.nan) for t in units.keys()})
    # Current notionals & weights
    curr_vals = last_px[list(units.keys())] * pd.Series(units)
    curr_weights = curr_vals / curr_vals.sum()
    pnl_abs = curr_vals - (start_px_series * pd.Series(units))
    pnl_pct = pnl_abs / (start_px_series * pd.Series(units))

    tbl = pd.DataFrame({
        "Units": pd.Series(units),
        "Start Px": start_px_series,
        "Last Px": last_px[list(units.keys())],
        "Current Value": curr_vals,
        "Weight %": curr_weights,
        "P/L %": pnl_pct
    })

    # Also show initial notionals from session if present
    init_notionals = st.session_state.get("investment_notionals", {})
    if isinstance(init_notionals, dict) and len(init_notionals) > 0:
        tbl.insert(1, "Start Notional", pd.Series(init_notionals))

    # Formatting
    st.dataframe(
        tbl.reindex(index=tickers_avail).style.format({
            "Units": "{:,.4f}",
            "Start Px": "{:,.2f}",
            "Last Px": "{:,.2f}",
            "Current Value": lambda x: f"{sym}{x:,.0f}",
            "Weight %": "{:.2%}",
            "P/L %": "{:.2%}",
            "Start Notional": lambda x: f"{sym}{x:,.0f}" if pd.notna(x) else "—",
        }),
        use_container_width=True
    )
else:
    # No units: show weights view only
    weights_df = pd.DataFrame({"Ticker": tickers_avail, "Weight": w[:len(tickers_avail)]}).set_index("Ticker")
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}), use_container_width=True)

    fig, ax = plt.subplots()
    ax.pie(weights_df["Weight"], labels=weights_df.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Portfolio Allocation")
    st.pyplot(fig)
