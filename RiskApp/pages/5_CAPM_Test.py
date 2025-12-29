import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Page Config
st.set_page_config(page_title="CAPM Analysis", layout="wide")
from helpers import add_big_sidebar_logo
add_big_sidebar_logo(width_px=220)
st.title("CAPM Model Test")

# Load session state data

# Check for required inputs from Portfolio Builder
required_keys = ["tickers", "weights", "start", "end", "market_symbol"]
missing = [k for k in required_keys if k not in st.session_state]

if missing:
    st.error(f"Missing data: {', '.join(missing)}. Please complete Portfolio Builder first.")
    st.stop()

# Load session state variables
stock_ticker_list = st.session_state.tickers
portfolio_weights_dictionary = st.session_state.weights  
analysis_start_date = st.session_state.start
analysis_end_date = st.session_state.end
market_benchmark_symbol = st.session_state.market_symbol
risk_free_rate_ticker = st.session_state.get("rf_ticker", "").strip()

# load price data

@st.cache_data(show_spinner=True)
def load_historical_price_data(symbol_list, start_date, end_date):
    """Load price data for all symbols"""
    try:
        price_data = yf.download(symbol_list, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data = price_data["Close"]
        return price_data.dropna(how="all")
    except Exception as error:
        st.error(f"Failed to load price data: {error}")
        return pd.DataFrame()

# Load prices for all assets + market benchmark
all_required_symbols = list(set(stock_ticker_list + [market_benchmark_symbol]))

historical_stock_prices = load_historical_price_data(all_required_symbols, analysis_start_date, analysis_end_date)

# Calculate logarithmic returns
daily_stock_returns = np.log(historical_stock_prices).diff().dropna()

# Check that market symbol exists
if market_benchmark_symbol not in daily_stock_returns.columns:
    st.error(f"Market symbol '{market_benchmark_symbol}' not found in data")
    st.stop()

# Filter available tickers
stocks_with_data = [ticker for ticker in stock_ticker_list if ticker in daily_stock_returns.columns]
stocks_missing_data = [ticker for ticker in stock_ticker_list if ticker not in daily_stock_returns.columns]

if stocks_missing_data:
    st.warning(f"Missing tickers: {stocks_missing_data}")

if not stocks_with_data:
    st.error("No valid tickers available")
    st.stop()

# CALCULATE PORTFOLIO RETURNS

# Get weights for available tickers (only stocks we have data for)
portfolio_weights_array = np.array([portfolio_weights_dictionary.get(ticker, 0.0) for ticker in stocks_with_data])
normalized_portfolio_weights = portfolio_weights_array / portfolio_weights_array.sum()  # Normalise weights

# Calculate portfolio returns
weighted_portfolio_returns = (daily_stock_returns[stocks_with_data] @ normalized_portfolio_weights).rename("Portfolio")
market_benchmark_returns = daily_stock_returns[market_benchmark_symbol]

# Get risk-free rate
@st.cache_data
def get_daily_risk_free_rate(risk_free_ticker_symbol):
    """Get risk-free rate from ticker symbol"""
    if not risk_free_ticker_symbol:
        return 0.0
    try:
        risk_free_data = yf.download(risk_free_ticker_symbol, period="5d", auto_adjust=False, progress=False)
        if not risk_free_data.empty and "Close" in risk_free_data.columns:
            raw_value = float(risk_free_data["Close"].iloc[-1])
            # Check if value is already in decimal (< 1) or percentage (> 1)
            if raw_value > 1:
                # Likely percentage format (e.g., 5.25 for 5.25%)
                annual_risk_free_rate = raw_value / 100.0
            else:
                # Already in decimal format (e.g., 0.0525 for 5.25%)
                annual_risk_free_rate = raw_value
            
            return annual_risk_free_rate / 252  # Convert to daily
        return 0.0
    except Exception as error:
        st.warning(f"Could not fetch risk-free rate from {risk_free_ticker_symbol}: {error}")
        return 0.0

daily_risk_free_rate = get_daily_risk_free_rate(risk_free_rate_ticker)
annual_risk_free_rate = daily_risk_free_rate * 252

# Display sample of return data
st.dataframe(daily_stock_returns.sample(10))
st.dataframe(daily_stock_returns)
# Run CAPM regression using statsmodels

st.divider()
st.subheader("Asset Selection")

# Asset selection dropdown
asset_analysis_options = ["Portfolio"] + stocks_with_data
selected_asset_for_analysis = st.selectbox("Choose an asset for CAPM analysis:", asset_analysis_options)

# Prepare data for regression
if selected_asset_for_analysis == "Portfolio":
    chosen_asset_returns = weighted_portfolio_returns
else:
    chosen_asset_returns = daily_stock_returns[selected_asset_for_analysis]

# Calculate excess returns
common_trading_dates = chosen_asset_returns.index.intersection(market_benchmark_returns.index)
asset_excess_returns = (chosen_asset_returns.loc[common_trading_dates] - daily_risk_free_rate).dropna()
market_excess_returns = (market_benchmark_returns.loc[common_trading_dates] - daily_risk_free_rate).dropna()

# Align the excess return data
excess_returns_data = pd.DataFrame({
    'asset_excess_returns': asset_excess_returns,
    'market_excess_returns': market_excess_returns
}).dropna()

if len(excess_returns_data) < 10:
    st.error("Insufficient data for regression (need at least 10 observations)")
    st.stop()

# Run CAPM regression: asset_excess = alpha + beta * market_excess + error
market_excess_with_constant = sm.add_constant(excess_returns_data['market_excess_returns'])  # Add intercept
asset_excess_dependent_variable = excess_returns_data['asset_excess_returns']

# Fit the CAPM model
capm_regression_model = sm.OLS(asset_excess_dependent_variable, market_excess_with_constant).fit()

# Display statsmodels results

st.divider()
st.subheader(f"CAPM Regression Results: {selected_asset_for_analysis}")

# Show full statsmodels summary
st.code(str(capm_regression_model.summary()), language=None)

# Extract key results
alpha_coefficient = capm_regression_model.params['const']
beta_coefficient = capm_regression_model.params['market_excess_returns']
r_squared_value = capm_regression_model.rsquared
number_of_observations = int(capm_regression_model.nobs)

# Visualisation of results

st.divider()
st.subheader("Regression Visualisation")

# Get the actual data used in regression
regression_asset_returns = capm_regression_model.model.endog  # Asset excess returns
regression_market_returns = capm_regression_model.model.exog[:, 1]  # Market excess returns (exclude constant column)

# Create scatter plot with regression line
figure, axis = plt.subplots(figsize=(10, 6))

# Scatter plot of data points
axis.scatter(regression_market_returns, regression_asset_returns, alpha=0.6, s=20, label="Daily observations")

# Regression line
market_return_range = np.linspace(regression_market_returns.min(), regression_market_returns.max(), 100)
fitted_asset_returns = alpha_coefficient + beta_coefficient * market_return_range
axis.plot(market_return_range, fitted_asset_returns, 'r-', linewidth=2, 
        label=f'CAPM Line: α={alpha_coefficient:.4f}, β={beta_coefficient:.3f}')

# Add zero lines
axis.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
axis.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

# Labels and formatting
axis.set_xlabel('Market Excess Returns')
axis.set_ylabel(f'{selected_asset_for_analysis} Excess Returns')
axis.set_title(f'CAPM Regression: {selected_asset_for_analysis} vs {market_benchmark_symbol}')
axis.legend()
axis.grid(True, alpha=0.3)

st.pyplot(figure)
