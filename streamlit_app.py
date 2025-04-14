import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    # Here we open our CSV file
    data = pd.read_csv("Download Data - FUND_US_XNAS_QQQ.csv", parse_dates=["Date"])
    # We use the "Date" column as the index (like having the date on the left side)
    data.set_index("Date", inplace=True)
    # Convert our columns to numbers if they are not already (this is very important)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data.dropna(inplace=True)  # Remove any rows that don't have complete data
    return data

data = load_data()
if data is None or data.empty:
    st.error("No data found. Please check your CSV file!")
    st.stop()

st.subheader("Raw Data Preview")
st.dataframe(data.head())
import ta

st.header("Calculating Technical Indicators")

# RSI: It tells if the price is too high (overbought) or too low (oversold)
data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()

# MACD: It shows the trend by comparing short-term and long-term averages
macd_obj = ta.trend.MACD(close=data["Close"])
data["MACD"] = macd_obj.macd()
data["MACD_Signal"] = macd_obj.macd_signal()

# Bollinger Bands: It gives an idea of the price range, like the upper and lower boundaries
bb_obj = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
data["BB_High"] = bb_obj.bollinger_hband()
data["BB_Low"] = bb_obj.bollinger_lband()

# Show a quick preview of these new columns
st.dataframe(data[["RSI", "MACD", "MACD_Signal", "BB_High", "BB_Low"]].head())
st.header("Prediction Score Calculation")

def prediction_score(row):
    score = 0
    # Check the RSI: if low, add a bit; if high, subtract a bit.
    if row["RSI"] < 30:
        score += 0.33
    elif row["RSI"] > 70:
        score -= 0.33
    
    # Check the MACD: if MACD is above its signal, add; if below, subtract.
    if row["MACD"] > row["MACD_Signal"]:
        score += 0.33
    elif row["MACD"] < row["MACD_Signal"]:
        score -= 0.33
    
    # Check Bollinger Bands: if the close is below the lower band, add; if above the upper band, subtract.
    if row["Close"] < row["BB_Low"]:
        score += 0.33
    elif row["Close"] > row["BB_High"]:
        score -= 0.33
    
    return score

data["Prediction_Score"] = data.apply(prediction_score, axis=1)
st.dataframe(data[["Close", "Prediction_Score"]].head())
st.header("Backtest Comparison")

# Shift the Close column by one row to see the next day’s close price
data["Next_Close"] = data["Close"].shift(-1)

# Calculate the percentage change between today and tomorrow
data["Actual_Change_%"] = ((data["Next_Close"] - data["Close"]) / data["Close"]) * 100

# See if our prediction direction (positive or negative) matches the actual change direction
data["Direction_Match"] = np.sign(data["Prediction_Score"]) == np.sign(data["Actual_Change_%"])

accuracy = data["Direction_Match"].mean() * 100
st.write(f"Prediction Accuracy: {accuracy:.2f}%")
st.dataframe(data[["Close", "Prediction_Score", "Next_Close", "Actual_Change_%", "Direction_Match"]].dropna())
st.header("Chart: QQQ Close vs. Prediction Score")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data["Close"], label="QQQ Close", marker="o")
ax.plot(data.index, data["Prediction_Score"], label="Prediction Score", linestyle="--", marker="x")
ax.set_xlabel("Date")
ax.set_ylabel("Price / Score")
ax.legend()
ax.grid(True)
st.pyplot(fig)
