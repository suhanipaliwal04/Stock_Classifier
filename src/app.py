# import streamlit as st
# import pandas as pd
# import yfinance as yf
# import joblib

# # Load your trained model
# import joblib

# # Load trained model and feature names
# model_data = joblib.load("best_stock_model.pkl")

# # If the saved file contains both model and feature list
# if isinstance(model_data, dict):
#     model = model_data.get("best_model", list(model_data.values())[0])
#     expected_features = model_data.get("feature_names", None)
# else:
#     # If it's just a single model object
#     model = model_data
#     expected_features = None
import streamlit as st
import pandas as pd
import yfinance as yf
import joblib

# Load saved model data
model_data = joblib.load("best_stock_model.pkl")

if isinstance(model_data, dict):
    model = model_data["best_model"]
    expected_features = model_data.get("feature_names", [])
    accuracy = model_data.get("accuracy", None)
else:
    model = model_data
    expected_features = []
    accuracy = None

st.title("📈 Stock Price Movement Prediction")
st.write("This app predicts **whether a stock will go UP or DOWN tomorrow** based on historical data and indicators.")
if accuracy:
    st.caption(f"Model Accuracy: {accuracy:.2%}")
# Input ticker
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA):", "AAPL")

# if st.button("Predict Next Day Movement"):
#     st.write(f"Fetching data for {ticker}...")
#     df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)

#     # Flatten MultiIndex if needed
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [col[0] for col in df.columns]

#     # Compute same features used during training
#     df['Return'] = df['Adj Close'].pct_change()
#     df['Return_lag_1'] = df['Return'].shift(1)
#     df['Return_lag_2'] = df['Return'].shift(2)
#     df['Return_lag_3'] = df['Return'].shift(3)
#     df['MA5'] = df['Adj Close'].rolling(window=5).mean()
#     df['MA10'] = df['Adj Close'].rolling(window=10).mean()
#     df['MA20'] = df['Adj Close'].rolling(window=20).mean()
#     df['MA_ratio_5_20'] = df['MA5'] / df['MA20']
#     df['vol_5'] = df['Return'].rolling(window=5).std()
#     df['vol_10'] = df['Return'].rolling(window=10).std()
#     df['momentum_5'] = df['Adj Close'] - df['Adj Close'].shift(5)
#     df['RSI'] = 100 - (100 / (1 + (df['Return'].rolling(14).mean() / df['Return'].rolling(14).std())))
#     df['MACD'] = df['Adj Close'].ewm(span=12, adjust=False).mean() - df['Adj Close'].ewm(span=26, adjust=False).mean()
#     df = df.dropna()
#     # Select last available data point for prediction
#     # X_latest = df.iloc[-1:].drop(columns=['Adj Close', 'Close', 'Open', 'High', 'Low', 'Volume'])
    
#     X_latest = df[expected_features].iloc[-1:].copy()
# if st.button("Predict Next Day Movement"):
#     st.write(f"Fetching data for {ticker}...")
#     df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)

#     # Flatten MultiIndex if needed
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [col[0] for col in df.columns]

#     # Compute same indicators
#     df['Return'] = df['Adj Close'].pct_change()
#     df['Return_lag_1'] = df['Return'].shift(1)
#     df['MA5'] = df['Adj Close'].rolling(window=5).mean()
#     df['MA10'] = df['Adj Close'].rolling(window=10).mean()
#     df['MA20'] = df['Adj Close'].rolling(window=20).mean()
#     df['MA_ratio_5_20'] = df['MA5'] / df['MA20']
#     df['vol_5'] = df['Return'].rolling(window=5).std()
#     df['momentum_5'] = df['Adj Close'] - df['Adj Close'].shift(5)
#     df = df.dropna().reset_index(drop=True)

#     # Only keep columns used during training
#     available_features = [f for f in expected_features if f in df.columns]
#     X_latest = df[available_features].iloc[-1:].copy()

#     if len(available_features) == 0:
#         st.error("Feature mismatch! The model's expected features are not found in the dataset.")
#     else:
#         pred = model.predict(X_latest)[0]

#         if pred == 1:
#             st.success(f"📈 Prediction: {ticker} will likely go **UP** tomorrow.")
#         else:
#             st.error(f"📉 Prediction: {ticker} will likely go **DOWN** tomorrow.")

#         st.line_chart(df['Adj Close'])

#     pred = model.predict(X_latest)[0]

#     if pred == 1:
#         st.success(f"📈 Prediction: {ticker} will likely go **UP** tomorrow.")
#     else:
#         st.error(f"📉 Prediction: {ticker} will likely go **DOWN** tomorrow.")

#     st.line_chart(df['Adj Close'])
# if st.button("Predict Next Day Movement"):
#     st.write(f"Fetching data for {ticker}...")
#     df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)

#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [col[0] for col in df.columns]

#     # Compute indicators (same as training)
#     df['Return'] = df['Adj Close'].pct_change()
#     df['Return_lag_1'] = df['Return'].shift(1)
#     df['MA5'] = df['Adj Close'].rolling(window=5).mean()
#     df['MA10'] = df['Adj Close'].rolling(window=10).mean()
#     df['MA20'] = df['Adj Close'].rolling(window=20).mean()
#     df['MA_ratio_5_20'] = df['MA5'] / df['MA20']
#     df['vol_5'] = df['Return'].rolling(window=5).std()
#     df['momentum_5'] = df['Adj Close'] - df['Adj Close'].shift(5)
#     df = df.dropna().reset_index(drop=True)

#     # --- Align exactly with model's expected feature order ---
#     expected_features = model_data.get("feature_names", [])
#     X_latest = pd.DataFrame(columns=expected_features)

#     for col in expected_features:
#         if col in df.columns:
#             X_latest[col] = df[col].iloc[-1]
#         else:
#             # fill missing feature with 0 (safe default)
#             X_latest[col] = 0

#     # Ensure correct shape (1 row)
#     X_latest = X_latest[expected_features].astype(float)

#     # --- Make prediction ---
#     try:
#         pred = model.predict(X_latest)[0]
#         if pred == 1:
#             st.success(f"📈 Prediction: {ticker} will likely go **UP** tomorrow.")
#         else:
#             st.error(f"📉 Prediction: {ticker} will likely go **DOWN** tomorrow.")
#         st.line_chart(df['Adj Close'])
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
# if st.button("Predict Next Day Movement"):
#     st.write(f"Fetching data for {ticker}...")
#     df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)

#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [col[0] for col in df.columns]

#     # Compute same indicators as in training
#     df['Return'] = df['Adj Close'].pct_change()
#     df['Return_lag_1'] = df['Return'].shift(1)
#     df['Return_lag_2'] = df['Return'].shift(2)
#     df['Return_lag_3'] = df['Return'].shift(3)
#     df['Return_lag_5'] = df['Return'].shift(5)
#     df['MA5'] = df['Adj Close'].rolling(window=5).mean()
#     df['MA10'] = df['Adj Close'].rolling(window=10).mean()
#     df['MA20'] = df['Adj Close'].rolling(window=20).mean()
#     df['MA_ratio_5_20'] = df['MA5'] / df['MA20']
#     df['vol_5'] = df['Return'].rolling(window=5).std()
#     df['vol_10'] = df['Return'].rolling(window=10).std()
#     df['momentum_5'] = df['Adj Close'] - df['Adj Close'].shift(5)

#     df = df.dropna().reset_index(drop=True)

#     # --- Verify that there is at least one valid row ---
#     if len(df) == 0:
#         st.error("⚠️ Not enough data to compute indicators. Try increasing the period to 1y or 2y.")
#         st.stop()

#     # --- Align with model features exactly ---
#     expected_features = model_data.get("feature_names", [])
#     latest_row = df.iloc[[-1]]  # last valid row only
#     X_latest = pd.DataFrame(columns=expected_features)

#     for col in expected_features:
#         if col in latest_row.columns:
#             X_latest[col] = latest_row[col].values
#         else:
#             X_latest[col] = [0.0]  # fill missing with zero

#     X_latest = X_latest[expected_features].astype(float)

#     # --- Make prediction ---
#     try:
#         pred = model.predict(X_latest)[0]
#         if pred == 1:
#             st.success(f"📈 Prediction: {ticker} will likely go **UP** tomorrow.")
#         else:
#             st.error(f"📉 Prediction: {ticker} will likely go **DOWN** tomorrow.")

#         st.line_chart(df['Adj Close'])
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objects as go

# ======================
# Load Model
# ======================
import os

model_path = os.path.join(os.path.dirname(__file__), "best_stock_model.pkl")
model_data = joblib.load(model_path)

if isinstance(model_data, dict):
    model = model_data["best_model"]
    expected_features = model_data.get("feature_names", [])
    accuracy = model_data.get("accuracy", None)
else:
    model = model_data
    expected_features = []
    accuracy = None

# ======================
# Streamlit Page Config
# ======================
st.set_page_config(page_title="Stock Trend Predictor", page_icon="📈", layout="wide")

# ======================
# Custom CSS
# ======================
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f3f9ff 0%, #ffffff 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        color: #0077b6;
        font-size: 42px;
        font-weight: 800;
        margin-top: -30px;
        margin-bottom: 10px;
    }
    .subtext {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 40px;
    }
    .card {
        border-radius: 16px;
        background-color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        padding: 25px;
        margin-top: 15px;
        text-align: center;
    }
    .up {
        color: #2ecc71;
        font-weight: bold;
        font-size: 26px;
    }
    .down {
        color: #e74c3c;
        font-weight: bold;
        font-size: 26px;
    }
    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# Title Section
# ======================
st.markdown("<h1 class='main-title'>📊 Stock Market Trend Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Predict if your selected stock will go <b>Up 📈</b> or <b>Down 📉</b> tomorrow using Machine Learning.</p>", unsafe_allow_html=True)

# ======================
# Sidebar
# ======================
st.sidebar.header("🔍 Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS):", "AAPL")
period = st.sidebar.selectbox("Select data period:", ["6mo", "1y", "2y"])
st.sidebar.caption("💡 Tip: Use Yahoo Finance symbols (e.g., `RELIANCE.NS` for NSE India)")

# ======================
# Model Accuracy Info
# ======================
if accuracy:
    st.sidebar.success(f"✅ Model Accuracy: {accuracy:.2%}")

# ======================
# Prediction Section
# ======================
if st.button("🚀 Predict Next Day Movement", use_container_width=True):
    st.info(f"Fetching and analyzing data for **{ticker}** ...")

    # ---- Fetch data ----
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)

    if df.empty:
        st.error("⚠️ Failed to fetch data. Please check the symbol and try again.")
        st.stop()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # ---- Compute Indicators ----
    df['Return'] = df['Adj Close'].pct_change()
    df['Return_lag_1'] = df['Return'].shift(1)
    df['Return_lag_2'] = df['Return'].shift(2)
    df['Return_lag_3'] = df['Return'].shift(3)
    df['Return_lag_5'] = df['Return'].shift(5)
    df['MA5'] = df['Adj Close'].rolling(window=5).mean()
    df['MA10'] = df['Adj Close'].rolling(window=10).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df['MA_ratio_5_20'] = df['MA5'] / df['MA20']
    df['vol_5'] = df['Return'].rolling(window=5).std()
    df['vol_10'] = df['Return'].rolling(window=10).std()
    df['momentum_5'] = df['Adj Close'] - df['Adj Close'].shift(5)
    df = df.dropna().reset_index(drop=True)

    if len(df) == 0:
        st.error("⚠️ Not enough data to compute indicators. Try a longer period.")
        st.stop()

    # ---- Align with model ----
    latest_row = df.iloc[[-1]]
    X_latest = pd.DataFrame(columns=expected_features)
    for col in expected_features:
        X_latest[col] = latest_row[col].values if col in latest_row.columns else [0.0]
    X_latest = X_latest[expected_features].astype(float)

    # ---- Predict ----
    try:
        pred = model.predict(X_latest)[0]
        label = "UP 📈" if pred == 1 else "DOWN 📉"
        color_class = "up" if pred == 1 else "down"

        st.markdown(f"<div class='card'><h3>Prediction for {ticker}</h3><div class='{color_class}'>{label}</div></div>", unsafe_allow_html=True)

        # ---- Chart ----
        st.subheader("📉 Recent Price Trend")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Adj Close'],
            name='Market Data'
        ))
        fig.update_layout(
            title=f"{ticker} - Price Chart ({period})",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ======================
# Footer
# ======================
st.markdown("<p class='footer'>💡 Developed by <b>Pranil Bankar</b> | Powered by Machine Learning & Streamlit</p>", unsafe_allow_html=True)
