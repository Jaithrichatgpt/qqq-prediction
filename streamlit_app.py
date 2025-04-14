import streamlit as st
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

st.title("QQQ Daily Prediction Backtest from CSV (Download Data - FUND_US_XNAS_QQQ)")

@st.cache_data
def load_data():
    """
    Load QQQ daily data from a CSV file named:
      'Download Data - FUND_US_XNAS_QQQ.csv'
    
    Ensure the file is in the same directory as this script 
    and that it's a valid CSV with the correct columns: 
    Date, Open, High, Low, Close, Volume.
    """
    try:
        # Read the CSV file; updating the file name to match your downloaded file
        data = pd.read_csv("Download Data - FUND_US_XNAS_QQQ.csv", parse_dates=["Date"])
        data.set_index("Date", inplace=True)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

data = load_data()

if data is None or data.empty:
    st.error("❌ No data found. Please check that your CSV file has the correct name and is in the same folder.")
    st.stop()

# Quick data preview
st.subheader("Data Preview")
st.dataframe(data.head())

# Check if columns exist
required_cols = ["Open", "High", "Low", "Close", "Volume"]
for col in required_cols:
    if col not in data.columns:
        st.error(f"Missing expected column: {col}")
        st.stop()

# --- Calculate Technical Indicators ---
st.header("Technical Indicators Calculation")

data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()
macd_obj = ta.trend.MACD(close=data["Close"])
data["MACD"] = macd_obj.macd()
data["MACD_Signal"] = macd_obj.macd_signal()
bb_obj = ta.volatility.BollingerBands(close=data["Close"])
data["BB_High"] = bb_obj.bollinger_hband()
data["BB_Low"] = bb_obj.bollinger_lband()

st.write("Sample Indicator Data:")
st.dataframe(data[["RSI", "MACD", "MACD_Signal", "BB_High", "BB_Low"]].head())

# --- Define Prediction Score ---
st.header("Prediction Score Calculation")

def prediction_score(row):
    score = 0
    # RSI: oversold (<30) bullish, overbought (>70) bearish
    if row["RSI"] < 30:
        score += 0.33
    elif row["RSI"] > 70:
        score -= 0.33

    # MACD: bullish if MACD > signal, bearish otherwise
    if row["MACD"] > row["MACD_Signal"]:
        score += 0.33
    elif row["MACD"] < row["MACD_Signal"]:
        score -= 0.33

    # Bollinger: bullish if close < lower band, bearish if close > upper band
    if row["Close"] < row["BB_Low"]:
        score += 0.33
    elif row["Close"] > row["BB_High"]:
        score -= 0.33

    return score

data["Prediction_Score"] = data.apply(prediction_score, axis=1)

st.write("Sample Prediction Scores:")
st.dataframe(data[["Close", "Prediction_Score"]].head())

# --- Backtest: Compare to Next-Day Movement ---
st.header("Backtest: Next-Day Comparison")

data["Next_Close"] = data["Close"].shift(-1)
data["Actual_Change_%"] = ((data["Next_Close"] - data["Close"]) / data["Close"]) * 100
data["Direction_Match"] = np.sign(data["Prediction_Score"]) == np.sign(data["Actual_Change_%"])

accuracy = data["Direction_Match"].mean() * 100
st.write(f"**Prediction Accuracy:** {accuracy:.2f}%")

results_df = data[["Close", "Prediction_Score", "Next_Close", "Actual_Change_%", "Direction_Match"]].dropna()
st.dataframe(results_df)

# --- Plot Results ---
st.header("Chart: QQQ Close vs. Prediction Score")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data["Close"], label="QQQ Close", marker="o")
ax.plot(data.index, data["Prediction_Score"], label="Prediction Score", linestyle="--", marker="x")
ax.set_xlabel("Date")
ax.set_ylabel("Price / Score")
ax.legend()
ax.grid(True)
st.pyplot(fig)
