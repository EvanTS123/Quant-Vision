import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm, skew, kurtosis


# Page Config
from helpers import add_big_sidebar_logo
add_big_sidebar_logo(width_px=220,)
st.set_page_config(page_title="Risk Metrics — Historical VaR/CVaR", layout="wide")
st.title("Risk Metrics")
st.subheader("Historical & Parametric VaR/CVaR (Portfolio)")

# Guard and shared state

if "tickers" not in st.session_state or len(st.session_state.tickers) == 0:
    st.warning("No portfolio found. Go to **Portfolio Builder** first.")
    st.stop()

tickers = st.session_state.tickers
start   = st.session_state.get("start", pd.Timestamp("2018-01-01").date())
end     = st.session_state.get("end", pd.Timestamp.today().date())
sym     = st.session_state.get("currency_symbol", "£")
invest_amt = float(st.session_state.get("portfolio_value_base", 1_000_000.0))

# weights (fallback to equal if missing)
w_dict = st.session_state.get("weights", {})
if not w_dict or any(t not in w_dict for t in tickers):
    w = np.ones(len(tickers)) / len(tickers)
else:
    w = np.array([float(w_dict[t]) for t in tickers], dtype=float)
    s = w.sum()
    w = (w / s) if s else np.ones(len(tickers)) / len(tickers)

# units from Portfolio Builder (preferred for money valuation)
units_dict   = st.session_state.get("investment_units")              # {ticker: units}
common_start = st.session_state.get("investment_common_start")       # date
have_units = isinstance(units_dict, dict) and len(units_dict) > 0 and common_start is not None

# Data loader

@st.cache_data(show_spinner=True)
def load_prices(tickers, start, end):
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    df = yf.download(
        tickers,
        start=str(s),
        end=str(e),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna(how="all")

# If we have a true buy date from Builder, start from there (so money series matches the units)
true_start = max(pd.Timestamp(start).date(), pd.Timestamp(common_start).date()) if have_units else pd.Timestamp(start).date()
prices = load_prices(tickers, true_start, end)
if prices.empty:
    st.error("No price data returned. Try expanding the date range.")
    st.stop()

# Align tickers to what we actually downloaded
tickers = [t for t in tickers if t in prices.columns]
w = np.array([w_dict.get(t, 0.0) for t in tickers], dtype=float)
w = (w / w.sum()) if w.sum() else np.ones(len(tickers))/len(tickers)

# RETURNS + VALUE SERIES

returns = prices.pct_change().dropna()
r_port  = (returns @ w).rename("Portfolio")  # percent return series for methods

if have_units:
    # units-based money valuation
    units = {t: float(units_dict[t]) for t in tickers if t in units_dict}
    value_series = (prices[list(units.keys())] * pd.Series(units)).sum(axis=1).dropna()
else:
    # Fallback: synthetic value series
    value_series = invest_amt * (1 + r_port).cumprod()

if value_series.empty or r_port.empty:
    st.error("No return/value series after processing. Try different dates or tickers.")
    st.stop()

current_value = float(value_series.iloc[-1])

# HEADER INFO

st.subheader(
    f"Period: {pd.Timestamp(value_series.index[0]).date()} → {pd.Timestamp(value_series.index[-1]).date()}  "
    f"•  Sample size: {len(r_port):,} days"
)
st.caption(
    ("Valued using fixed **units** from the Portfolio Builder (true buy date) — " if have_units
     else "Valued using a weights-based synthetic path — ")
    + "Yahoo Finance adjusted closes (splits/dividends)."
)
st.divider()

# 1) HISTORICAL VaR / CVaR (DAILY, empirical)

st.header("Historical VaR / CVaR")

def hist_var_cvar(series: pd.Series, alpha: float):
    """Historical (empirical) DAILY VaR/CVaR on the full sample."""
    s = pd.Series(series).dropna()
    q = float(np.quantile(s, 1 - alpha))       # cutoff return (likely negative)
    var_loss  = -q                              # positive loss magnitude
    tail = s[s <= q]
    cvar_loss = float(-tail.mean()) if len(tail) else np.nan
    return q, var_loss, cvar_loss

alphas = [0.95, 0.99]
daily = {a: dict(zip(["q", "var", "cvar"], hist_var_cvar(r_port, a))) for a in alphas}

# Table with money terms
hist_table = pd.DataFrame({
    "Confidence":        [f"{int(a*100)}%" for a in alphas],
    "VaR (loss, %)":     [daily[a]["var"] for a in alphas],
    "CVaR (loss, %)":    [daily[a]["cvar"] for a in alphas],
    "VaR (loss, money)": [daily[a]["var"] * current_value for a in alphas],
    "CVaR (loss, money)":[daily[a]["cvar"] * current_value for a in alphas],
    "Cutoff return":     [daily[a]["q"] for a in alphas],
}).set_index("Confidence")

def fmt_money(x): return f"{sym}{x:,.0f}" if pd.notna(x) else "—"

st.dataframe(
    hist_table.style.format({
        "VaR (loss, %)": "{:.2%}",
        "CVaR (loss, %)": "{:.2%}",
        "VaR (loss, money)": fmt_money,
        "CVaR (loss, money)": fmt_money,
        "Cutoff return": "{:.2%}",
    }),
    use_container_width=True
)

# Histogram with VaR/CVaR bands
st.subheader("Daily return distribution with Historical VaR & CVaR (95% & 99%)")
fig, ax = plt.subplots(figsize=(10, 5.5))
n, bins, _ = ax.hist(r_port, bins=60, edgecolor="black", color="lightgray")
ymax = max(n)

palette = {0.95: "tab:red", 0.99: "orange"}
for a in alphas:
    q      = daily[a]["q"]
    var_d  = daily[a]["var"]
    cvar_d = daily[a]["cvar"]
    tail_mean = r_port[r_port <= q].mean()

    ax.fill_betweenx([0, ymax], bins[0], q, color=palette[a], alpha=0.18,
                     label=f"Tail {int(a*100)}% (daily)")
    ax.axvline(q, color=palette[a], linestyle="--", linewidth=2,
               label=f"VaR {int(a*100)}% = {var_d:.2%} • {fmt_money(var_d*current_value)}")
    ax.axvline(tail_mean, color=palette[a], linestyle="--", linewidth=1.8, dashes=(6,2),
               label=f"CVaR {int(a*100)}% = {cvar_d:.2%} • {fmt_money(cvar_d*current_value)}")

ax.set_title("Portfolio Daily Returns — Historical VaR/CVaR (95% & 99%)")
ax.set_xlabel("Daily return")
ax.set_ylabel("Frequency")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right")
st.pyplot(fig)

st.caption(
    "VaR is the empirical loss threshold (5%/1% tails). CVaR is the average of losses beyond that threshold. "
    "Money values use your portfolio’s **current value**."
)

st.divider()

# 2) PARAMETRIC VaR — Normal & Cornish–Fisher (DAILY)

st.header("Parametric VaR — Normal vs Cornish–Fisher")

mu  = float(r_port.mean())
sig = float(r_port.std(ddof=1))
g1  = float(skew(r_port, bias=False))                  # skewness
g2  = float(kurtosis(r_port, fisher=True, bias=False)) # excess kurtosis

def normal_var_cvar(mu, sig, alpha):
    z = norm.ppf(1 - alpha)         # negative
    q = mu + sig * z                 # cutoff return (daily)
    var_loss  = -q
    cvar_loss = -(mu - sig * norm.pdf(z) / (1 - alpha))
    return {"z": z, "q": q, "VaR": var_loss, "CVaR": cvar_loss}

def cornish_fisher_z(z, g1, g2):
    z2, z3 = z*z, z*z*z
    return (z
            + (1/6.0)  * (z2 - 1)     * g1
            + (1/24.0) * (z3 - 3*z)   * g2
            - (1/36.0) * (2*z3 - 5*z) * (g1**2))

def cornish_fisher_var(mu, sig, alpha, g1, g2, series):
    z    = norm.ppf(1 - alpha)
    z_cf = cornish_fisher_z(z, g1, g2)
    q_cf = mu + sig * z_cf
    var_cf  = -q_cf
    tail = series[series <= q_cf]
    cvar_cf = float(-tail.mean()) if len(tail) else np.nan
    return {"z_cf": z_cf, "q_cf": q_cf, "VaR_CF": var_cf, "CVaR_CF": cvar_cf}

alphas = [0.95, 0.99]
rows = []
for a in alphas:
    nrm = normal_var_cvar(mu, sig, a)
    cf  = cornish_fisher_var(mu, sig, a, g1, g2, r_port)
    rows.append({
        "Confidence": f"{int(a*100)}%",
        "Normal VaR (%)":  nrm["VaR"],
        "Normal CVaR (%)": nrm["CVaR"],
        "Normal VaR (money)":  nrm["VaR"] * current_value,
        "Normal CVaR (money)": nrm["CVaR"] * current_value,
        "CF VaR (%)":      cf["VaR_CF"],
        "CF CVaR (%)":     cf["CVaR_CF"],
        "CF VaR (money)":  cf["VaR_CF"] * current_value,
        "CF CVaR (money)": cf["CVaR_CF"] * current_value,
    })

param_df = pd.DataFrame(rows).set_index("Confidence")
st.dataframe(
    param_df.style.format({
        "Normal VaR (%)": "{:.2%}",
        "Normal CVaR (%)": "{:.2%}",
        "CF VaR (%)": "{:.2%}",
        "CF CVaR (%)": "{:.2%}",
        "Normal VaR (money)": fmt_money,
        "Normal CVaR (money)": fmt_money,
        "CF VaR (money)": fmt_money,
        "CF CVaR (money)": fmt_money,
    }),
    use_container_width=True
)
st.caption("* CF CVaR shown as a hybrid empirical tail mean under the CF cutoff (no closed form).")

# Plot: overlay Normal & CF cutoffs on the daily histogram
fig_p, ax_p = plt.subplots(figsize=(10, 5.5))
n_p, bins_p, _ = ax_p.hist(r_port, bins=60, edgecolor="black", color="lightgray")
palette = {0.95: "tab:red", 0.99: "orange"}

for a, color in palette.items():
    nrm = normal_var_cvar(mu, sig, a)
    cf  = cornish_fisher_var(mu, sig, a, g1, g2, r_port)

    ax_p.axvspan(bins_p[0], nrm["q"], color=color, alpha=0.12, label=f"Tail {int(a*100)}% (Normal)")
    ax_p.axvline(nrm["q"], color=color, linestyle="--", linewidth=2,
                 label=f"Normal VaR {int(a*100)}% = {nrm['VaR']:.2%} • {fmt_money(nrm['VaR']*current_value)}")
    ax_p.axvline(cf["q_cf"], color=color, linestyle="-.", linewidth=2,
                 label=f"CF VaR {int(a*100)}% = {cf['VaR_CF']:.2%} • {fmt_money(cf['VaR_CF']*current_value)}")

ax_p.set_title("Parametric Daily Cutoffs — Normal vs Cornish–Fisher (95% & 99%)")
ax_p.set_xlabel("Daily return"); ax_p.set_ylabel("Frequency")
ax_p.grid(True, linestyle="--", alpha=0.5)
ax_p.legend(loc="upper right")
st.pyplot(fig_p)

st.caption(
    "Normal VaR assumes Gaussian returns. Cornish–Fisher adjusts the quantile using sample skewness and excess kurtosis, "
    "capturing asymmetry and fat tails. Money values use **current portfolio value**."
)

