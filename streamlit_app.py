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
