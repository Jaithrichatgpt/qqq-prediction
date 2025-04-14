import streamlit as st
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

st.title("QQQ Daily Prediction Backtest with Computed Indicators")

@st.cache_data
def load_data():
    try:
        # Load CSV file; adjust the file name/path if necessary.
        data = pd.read_csv("Download Data - FUND_US_XNAS_QQQ.csv", parse_dates=["Date"])
        data.set_index("Date", inplace=True)
        # Convert price columns to numeric
        cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

data = load_data()

if data is None or data.empty:
    st.error("❌ No data loaded. Please check your CSV file.")
    st.stop()

st.subheader("Raw Data Preview")
st.dataframe(data.head())

# --- Calculate Technical Indicators ---
st.header("Technical Indicators Calculation")

# RSI Calculation (14-day)
data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()

# MACD Calculation (default parameters: fast=12, slow=26, signal=9)
macd_obj = ta.trend.MACD(close=data["Close"])
data["MACD"] = macd_obj.macd()
data["MACD_Signal"] = macd_obj.macd_signal()

# Bollinger Bands Calculation (window=20, std=2)
bb_obj = ta.volatility.BollingerBands(close=data["Close"])
data["BB_High"] = bb_obj.bollinger_hband()
data["BB_Low"] = bb_obj.bollinger_lband()

st.write("Computed Technical Indicators:")
st.dataframe(data[["RSI", "MACD", "MACD_Signal", "BB_High", "BB_Low"]].head())

# Continue with the rest of your app (e.g., prediction score, backtest calculations, plots)
