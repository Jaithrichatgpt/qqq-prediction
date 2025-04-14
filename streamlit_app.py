import streamlit as st
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

st.title("QQQ Daily Prediction Backtest from CSV (03-12-2025 to 04-11-2025)")

@st.cache_data
def load_data():
    """
    Load QQQ daily data from a CSV file.
    The CSV must have columns: Date, Open, High, Low, Close, Volume.
    Adjust the file name/path as necessary.
    """
    try:
        # Read the CSV file; update the filename if needed
        data = pd.read_csv("qqq_03_12_2025_to_04_11_2025.csv", parse_dates=["Date"])
        data.set_index("Date", inplace=True)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

# Load the data
data = load_data()

if data is None or data.empty:
    st.error("❌ No data found. Please check that your CSV file is in the correct location and properly formatted.")
    st.stop()

# Preview the loaded data
st.subheader("Data Preview")
st.dataframe(data.head())

# Ensure that the required columns exist; adjust if necessary.
expected_cols = ["Open", "High", "Low", "Close", "Volume"]
if not all(col in data.columns for col in expected_cols):
    st.error(f"Missing one or more required columns: {expected_cols}")
    st.stop()

# --- Calculate Technical Indicators ---
st.header("Technical Indicators Calculation")

# Calculate the 14-day RSI
data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()

# Calculate MACD (default parameters: fast=12, slow=26, signal=9)
macd_obj = ta.trend.MACD(close=data["Close"])
data["MACD"] = macd_obj.macd()
data["MACD_Signal"] = macd_obj.macd_signal()

# Calculate Bollinger Bands (default: window=20, std_dev=2)
bb_obj = ta.volatility.BollingerBands(close=data["Close"])
data["BB_High"] = bb_obj.bollinger_hband()
data["BB_Low"] = bb_obj.bollinger_lband()

st.write("Sample of Indicator Data:")
st.dataframe(data[["RSI", "MACD", "MACD_Signal", "BB_High", "BB_Low"]].head())

# --- Define the Prediction Score Function ---
st.header("Prediction Score Calculation")

def prediction_score(row):
    """
    Compute a composite prediction score based on three factors:
    1. RSI: 
       - If RSI < 30: add 0.33 (bullish)
       - If RSI > 70: subtract 0.33 (bearish)
    2. MACD: 
       - If MACD > MACD_Signal: add 0.33
       - If MACD < MACD_Signal: subtract 0.33
    3. Bollinger Bands: 
       - If Close < BB_Low: add 0.33
       - If Close > BB_High: subtract 0.33
    The resulting score ranges from -1 (strong bearish) to +1 (strong bullish).
    """
    score = 0
    # RSI factor
    if row["RSI"] < 30:
        score += 0.33
    elif row["RSI"] > 70:
        score -= 0.33
    # MACD factor
    if row["MACD"] > row["MACD_Signal"]:
        score += 0.33
    elif row["MACD"] < row["MACD_Signal"]:
        score -= 0.33
    # Bollinger Bands factor
    if row["Close"] < row["BB_Low"]:
        score += 0.33
    elif row["Close"] > row["BB_High"]:
        score -= 0.33
    return score

data["Prediction_Score"] = data.apply(prediction_score, axis=1)

st.write("Sample Prediction Scores:")
st.dataframe(data[["Close", "Prediction_Score"]].head())

# --- Evaluate Prediction Performance ---
st.header("Backtest: Prediction vs. Actual Movement")

# Compare prediction to the next day's close
data["Next_Close"] = data["Close"].shift(-1)
data["Actual_Change_%"] = ((data["Next_Close"] - data["Close"]) / data["Close"]) * 100

# Determine if the direction of the prediction matches the actual movement
data["Direction_Match"] = np.sign(data["Prediction_Score"]) == np.sign(data["Actual_Change_%"])

# Calculate overall prediction accuracy (percentage of rows where direction matches)
accuracy = data["Direction_Match"].mean() * 100

st.write(f"**Prediction Accuracy:** {accuracy:.2f}%")
results_df = data[["Close", "Prediction_Score", "Next_Close", "Actual_Change_%", "Direction_Match"]].dropna()
st.dataframe(results_df)

# --- Plotting the Results ---
st.header("Chart: QQQ Close Price vs. Prediction Score")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data["Close"], label="QQQ Close Price", marker="o")
ax.plot(data.index, data["Prediction_Score"], label="Prediction Score", marker="x", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Price / Score")
ax.legend()
ax.grid(True)
st.pyplot(fig)
