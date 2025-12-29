import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
import io


# PAGE CONFIG
from helpers import add_big_sidebar_logo
add_big_sidebar_logo(width_px=220,)
st.set_page_config(page_title="Portfolio Builder", layout="wide")


# Session_state (defaults)
if "tickers" not in st.session_state:
    st.session_state.tickers = []

if "weights" not in st.session_state:
    st.session_state.weights = {}

if "start" not in st.session_state:
    st.session_state.start = pd.Timestamp("2024-11-04").date()

if "end" not in st.session_state:
    st.session_state.end = pd.Timestamp("2025-11-04").date()

# simple value + symbol
if "portfolio_value_base" not in st.session_state:
    st.session_state.portfolio_value_base = 1_000_000.0
if "currency_symbol" not in st.session_state:
    st.session_state.currency_symbol = "£"

# Market & RF tickers (defaults)
if "market_symbol" not in st.session_state:
    st.session_state.market_symbol = "^GSPC"
if "rf_ticker" not in st.session_state:
    st.session_state.rf_ticker = "^TNX"

# These will be produced after prices are fetched
for k in ["investment_common_start",
          "investment_start_prices",
          "investment_notionals",
          "investment_units"]:
    if k not in st.session_state:
        st.session_state[k] = None

# Validate ticker function
@st.cache_data(show_spinner=False)
def is_valid_stock_ticker(ticker_symbol: str) -> bool:
    ticker_symbol = ticker_symbol.strip().upper()
    try:
        stock_data = yf.download(ticker_symbol, period="5d", auto_adjust=True, progress=False, threads=False)
        return not stock_data.empty
    except Exception:
        return False
st.title("Build Your Portfolio")

st.subheader("Portfolio Value")

# Portfolio value and currency
c_val1, c_val2 = st.columns([2, 1])
with c_val1:
    portfolio_value = st.number_input(
        "Enter your total portfolio value",
        min_value=0.0,
        value=float(st.session_state.get("portfolio_value_base", 1_000_000.0)),
        step=1_000.0
    )
with c_val2:
    currency_symbol = st.selectbox(
        "Currency",
        ["£", "$", "€"],
        index=["£", "$", "€"].index(st.session_state.get("currency_symbol", "£"))
    )

st.markdown(f"### Portfolio Value: **{currency_symbol}{portfolio_value:,.0f}**")
st.session_state.portfolio_value_base = float(portfolio_value)
st.session_state.currency_symbol = currency_symbol
sym = st.session_state.currency_symbol

# Add tickers
with st.form("add_form"):
    st.caption("Enter tickers (e.g., SPY, AAPL, TLT)")
    user_ticker_input = st.text_input("Add ticker(s)", placeholder="SPY, AAPL, TLT")
    ticker_submit_button = st.form_submit_button("Add to portfolio")

if ticker_submit_button and user_ticker_input.strip():
    candidate_tickers = [ticker.strip().upper() for ticker in user_ticker_input.split(",") if ticker.strip()]
    successfully_added_tickers, already_existing_tickers, invalid_ticker_symbols = [], [], []
    for ticker_symbol in candidate_tickers:
        if ticker_symbol in st.session_state.tickers:
            already_existing_tickers.append(ticker_symbol)
        elif is_valid_stock_ticker(ticker_symbol):
            st.session_state.tickers.append(ticker_symbol)
            # default equal weight on add
            st.session_state.weights[ticker_symbol] = st.session_state.weights.get(
                ticker_symbol, round(1 / len(st.session_state.tickers), 4)
            )
            successfully_added_tickers.append(ticker_symbol)
        else:
            invalid_ticker_symbols.append(ticker_symbol)
    if successfully_added_tickers:
        st.success(f"Added: {', '.join(successfully_added_tickers)}")
        # Equal-weight all tickers so weights sum to 1
        number_of_tickers = len(st.session_state.tickers)
        if number_of_tickers > 0:
            equal_weight = float(np.round(1.0 / number_of_tickers, 4))
            for ticker in st.session_state.tickers:
                st.session_state.weights[ticker] = equal_weight
            st.info("Weights set to equal allocation (sum to 1).")
    if already_existing_tickers:
        st.info(f"Already added: {', '.join(already_existing_tickers)}")
    if invalid_ticker_symbols:
        st.error(f"Invalid tickers: {', '.join(invalid_ticker_symbols)}")

# DISPLAY / EDIT PORTFOLIO
st.subheader("Portfolio")
if not st.session_state.tickers:
    st.info("No tickers added.")
    st.stop()

portfolio_display_columns = st.columns(min(4, len(st.session_state.tickers)))
tickers_to_remove = []
for index, ticker_symbol in enumerate(st.session_state.tickers):
    if portfolio_display_columns[index % 4].button(f"Remove {ticker_symbol}", key=f"remove_{ticker_symbol}"):
        tickers_to_remove.append(ticker_symbol)
for ticker_to_remove in tickers_to_remove:
    st.session_state.tickers.remove(ticker_to_remove)
    st.session_state.weights.pop(ticker_to_remove, None)
if tickers_to_remove and st.session_state.tickers:
    # Re-equalize after removals
    remaining_ticker_count = len(st.session_state.tickers)
    equal_weight_after_removal = float(np.round(1.0 / remaining_ticker_count, 4))
    for remaining_ticker in st.session_state.tickers:
        st.session_state.weights[remaining_ticker] = equal_weight_after_removal
    st.info("Weights rebalanced to equal allocation after removals.")

# Date range
date_column_1, date_column_2 = st.columns(2)
analysis_start_date = date_column_1.date_input("Start date", st.session_state.start, key="builder_start")
analysis_end_date = date_column_2.date_input("End date", st.session_state.end, key="builder_end")
st.session_state.start = pd.Timestamp(analysis_start_date).date()
st.session_state.end = pd.Timestamp(analysis_end_date).date()

st.markdown("### Market & Risk-Free Tickers")
market_rf_column_1, market_rf_column_2 = st.columns(2)
market_benchmark_symbol = market_rf_column_1.text_input(
    "Market Portfolio Ticker (Yahoo symbol)",
    value=st.session_state.get("market_symbol", "^GSPC"),
    help="Examples: ^GSPC (S&P 500), ^FTSE (FTSE 100), ^IXIC (NASDAQ)"
)
st.session_state.market_symbol = market_benchmark_symbol.strip()

risk_free_rate_ticker = market_rf_column_2.text_input(
    "Risk-Free Rate Ticker (Yahoo symbol)",
    value=st.session_state.get("rf_ticker", "^IRX"),
    help="Examples: ^IRX (13-week T-Bill yield), ^TNX (10-year Treasury yield)"
)
st.session_state.rf_ticker = risk_free_rate_ticker.strip()

@st.cache_data(show_spinner=False)
def fetch_latest_stock_price(ticker_symbol):
    try:
        stock_data = yf.download(ticker_symbol, period="5d", auto_adjust=True, progress=False, threads=False)
        if stock_data.empty:
            return None
        return float(stock_data["Close"].iloc[-1])
    except Exception:
        return None

market_display_column, rf_display_column = st.columns(2)
current_risk_free_value = fetch_latest_stock_price(st.session_state.rf_ticker) if st.session_state.rf_ticker else None
current_market_value = fetch_latest_stock_price(st.session_state.market_symbol) if st.session_state.market_symbol else None
if current_risk_free_value is not None:
    rf_display_column.metric(f"{st.session_state.rf_ticker} (Risk-Free Yield Level)", f"{current_risk_free_value:.2f}")
else:
    rf_display_column.info("Enter a valid RF ticker (e.g., ^IRX).")
if current_market_value is not None:
    market_display_column.metric(f"{st.session_state.market_symbol} (Market Index Level)", f"{current_market_value:,.2f}")
else:
    market_display_column.info("Enter a valid market index ticker (e.g., ^GSPC).")

# WEIGHTS EDITOR
st.subheader("Weights")

# Quick equalize button
if st.button("Set Equal Weights"):
    number_of_tickers = len(st.session_state.tickers)
    if number_of_tickers > 0:
        equal_weight_value = float(np.round(1.0 / number_of_tickers, 4))
        for ticker_symbol in st.session_state.tickers:
            st.session_state.weights[ticker_symbol] = equal_weight_value
        st.success("All weights set to equal (sum to 1).")

individual_weight_inputs = []
for ticker_symbol in st.session_state.tickers:
    individual_weight_inputs.append(
        st.number_input(
            f"{ticker_symbol} weight",
            key=f"weight_{ticker_symbol}",
            value=float(st.session_state.weights.get(ticker_symbol, 0.0)),
            step=0.05,
        )
    )

total_weights = sum(individual_weight_inputs)
if total_weights == 0:
    normalized_weights = [1 / len(individual_weight_inputs)] * len(individual_weight_inputs)
else:
    normalized_weights = [individual_weight / total_weights for individual_weight in individual_weight_inputs]

for ticker_symbol, normalized_weight in zip(st.session_state.tickers, normalized_weights):
    st.session_state.weights[ticker_symbol] = float(np.round(normalized_weight, 4))

