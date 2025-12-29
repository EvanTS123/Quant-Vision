import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import arch 


# Page Config
from helpers import add_big_sidebar_logo
add_big_sidebar_logo(width_px=220,)
st.set_page_config(page_title="Volatility Models", layout="wide")
st.title("Volatility Models — Flat vs EWMA vs GARCH")

# GUARD and SHARED STATE

if "tickers" not in st.session_state or len(st.session_state.tickers) == 0:
    st.warning("No portfolio found. Go to **Portfolio Builder** first.")
    st.stop()

tickers = st.session_state.tickers
start_sel = st.session_state.get("start", pd.Timestamp("2018-01-01").date())
end_sel   = st.session_state.get("end", pd.Timestamp.today().date())
rf        = float(st.session_state.get("rf", 0.04))
sym       = st.session_state.get("currency_symbol", "£")

# weights (fallback to equal)
w_dict = st.session_state.get("weights", {})
w = np.array([w_dict.get(t, 0.0) for t in tickers], dtype=float)
w = (w / w.sum()) if w.sum() else np.ones(len(tickers))/len(tickers)

# optional fixed units (preferred if available)
units_dict   = st.session_state.get("investment_units")            # {ticker: units}
common_start = st.session_state.get("investment_common_start")     # date or None
have_units = isinstance(units_dict, dict) and len(units_dict) > 0 and common_start is not None

# DATA LOADER

@st.cache_data(show_spinner=True)
def load_prices(tickers, start, end):
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    df = yf.download(tickers, start=str(s), end=str(e),
                     auto_adjust=True, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna(how="all")

# Align start with true buy date if units exist
true_start = max(pd.Timestamp(start_sel).date(), pd.Timestamp(common_start).date()) if have_units else pd.Timestamp(start_sel).date()
prices = load_prices(tickers, true_start, end_sel)
if prices.empty:
    st.error("No price data returned. Try expanding the date range.")
    st.stop()

tickers_avail = [t for t in tickers if t in prices.columns]
prices = prices[tickers_avail]

# Portfolio return series

if have_units:
    units = {t: float(units_dict[t]) for t in tickers_avail if t in units_dict}
    if len(units) == 0:
        have_units = False

if have_units:
    # Value-based returns from fixed units
    value_series = (prices[list(units.keys())] * pd.Series(units)).sum(axis=1).dropna()
    r = value_series.pct_change().dropna().rename("Portfolio")
else:
    # Weights-based returns
    w2 = np.array([w_dict.get(t, 0.0) for t in tickers_avail], dtype=float)
    w2 = (w2 / w2.sum()) if w2.sum() else np.ones(len(tickers_avail))/len(tickers_avail)
    returns = prices.pct_change().dropna()
    r = (returns @ w2).rename("Portfolio")

if r.empty:
    st.error("No return data. Try different tickers or dates.")
    st.stop()

st.caption(
    f"Period: **{pd.Timestamp(r.index[0]).date()} → {pd.Timestamp(r.index[-1]).date()}** "
    + ("(valued from fixed units bought on the common start date)" if have_units
       else "(weights-based approximation)")
)

# Controls for vol models

c1, c2, c3 = st.columns([1,1,1])
roll_win = c1.slider("Rolling window (days)", 10, 120, 60, 1)
lam      = c2.slider("EWMA λ (decay)", 0.80, 0.99, 0.94, 0.01)

# Vol series

# Flat (constant) daily/annualised
flat_vol_daily  = r.std(ddof=1)

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

_ppyear = _infer_periods_per_year(r.index)
flat_vol_annual = flat_vol_daily * np.sqrt(_ppyear)

# Rolling realised vol
rolling_vol_daily  = r.rolling(roll_win).std()
rolling_vol_annual = rolling_vol_daily * np.sqrt(_ppyear)

# EWMA vol
ewma_var        = (r**2).ewm(alpha=(1 - lam)).mean()
ewma_vol_daily  = np.sqrt(ewma_var)
ewma_vol_annual = ewma_vol_daily * np.sqrt(_ppyear)

# GARCH
garch_vol_daily = None
garch_vol_annual = None
garch_msg = None

from arch import arch_model
pct = (r * 100.0)
am = arch_model(pct, vol="Garch", p=1, q=1, mean="Zero")
res = am.fit(update_freq=0, disp="off")
garch_vol_daily = (res.conditional_volatility / 100.0)  # back to daily return units
garch_vol_annual = garch_vol_daily * np.sqrt(_ppyear)

# KPI strip

k1, k2, k3 = st.columns(3)
k1.metric("Flat Vol (ann.)", f"{flat_vol_annual:.2%}")
k2.metric(f"Rolling {roll_win}d Vol (ann.)",
          f"{rolling_vol_annual.dropna().iloc[-1]:.2%}" if rolling_vol_annual.dropna().size else "N/A")
k3.metric(f"EWMA Vol (ann.) λ={lam:.2f}",
          f"{ewma_vol_annual.dropna().iloc[-1]:.2%}" if ewma_vol_annual.dropna().size else "N/A")

if garch_vol_annual is not None:
    st.metric("GARCH(1,1) Vol (ann.)", f"{garch_vol_annual.dropna().iloc[-1]:.2%}")
elif garch_msg:
    st.info(garch_msg)

# Chart: Vol over time 

st.subheader("Annualized Volatility Over Time")
fig, ax = plt.subplots(figsize=(10,4))
rolling_vol_annual.plot(ax=ax, label=f"Rolling {roll_win}d", alpha=0.9)
ewma_vol_annual.plot(ax=ax, label=f"EWMA λ={lam:.2f}", alpha=0.9)
if garch_vol_annual is not None:
    garch_vol_annual.plot(ax=ax, label="GARCH(1,1)", alpha=0.9)
ax.axhline(flat_vol_annual, color="gray", linestyle="--", label=f"Flat = {flat_vol_annual:.2%}")
ax.set_ylabel("Annualized Volatility")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
st.pyplot(fig)

# Forcast pannel (EWMA / GARCH)

st.divider()
st.subheader("Volatility Forecast")

h = st.slider("Forecast horizon (days)", 1, 30, 10, 1)

# EWMA: approximate h-day volatility = last daily vol * sqrt(h)
ewma_last_daily = ewma_vol_daily.dropna().iloc[-1] if ewma_vol_daily.dropna().size else np.nan
ewma_h_day      = ewma_last_daily * np.sqrt(h)
ewma_h_ann      = ewma_h_day * np.sqrt(_ppyear / h)  # same as last daily * sqrt(periods/yr)

cols = st.columns(2)
cols[0].metric(f"EWMA {h}d Vol (h-day σ)", f"{ewma_h_day:.2%}" if pd.notna(ewma_h_day) else "N/A")
cols[0].caption(f"Annualized equivalent ≈ last EWMA daily × √{_ppyear}")

if garch_vol_daily is not None and garch_vol_daily.dropna().size:
    garch_last_daily = garch_vol_daily.dropna().iloc[-1]
    garch_h_day      = garch_last_daily * np.sqrt(h)
    cols[1].metric(f"GARCH {h}d Vol (h-day σ)", f"{garch_h_day:.2%}")
else:
    cols[1].caption("Enable GARCH (and install `arch`) to see GARCH forecast.")
