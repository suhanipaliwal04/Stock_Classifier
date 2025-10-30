
import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from datetime import date, timedelta

# -----------------------
# Helper functions
# -----------------------
def compute_technical_indicators(df):
    """Compute features used in training. Operates in-place and returns df."""
    df = df.copy()
    # Adjusted Close must exist
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    # Returns and lags
    df['Return'] = df['Adj Close'].pct_change()
    for lag in [1, 2, 3, 5]:
        df[f'Return_lag_{lag}'] = df['Return'].shift(lag)

    # Moving averages
    df['MA5'] = df['Adj Close'].rolling(5).mean()
    df['MA10'] = df['Adj Close'].rolling(10).mean()
    df['MA20'] = df['Adj Close'].rolling(20).mean()
    df['MA_ratio_5_20'] = df['MA5'] / df['MA20']

    # Volatility
    df['vol_5'] = df['Return'].rolling(5).std()
    df['vol_10'] = df['Return'].rolling(10).std()

    # Momentum
    df['momentum_5'] = df['Adj Close'] - df['Adj Close'].shift(5)

    # RSI
    delta = df['Adj Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # Day of week
    df['dow'] = df.index.dayofweek

    return df

def build_X_latest(df, feature_names):
    """Create a one-row DataFrame with features in feature_names from df (latest row)."""
    latest = df.iloc[-1:]
    # if feature list empty, infer common features:
    if not feature_names:
        feature_names = [
            'Return_lag_1','Return_lag_2','Return_lag_3','Return_lag_5',
            'MA5','MA10','MA20','MA_ratio_5_20','vol_5','vol_10',
            'momentum_5','RSI','MACD','dow'
        ]
    # build row with fallback zeros if missing
    row = {}
    for f in feature_names:
        if f in latest.columns:
            row[f] = latest[f].values[0]
        else:
            # fallback: 0 or reasonable default for dow
            if f == 'dow':
                row[f] = int(latest.index[-1].dayofweek)
            else:
                row[f] = 0.0
    X = pd.DataFrame([row], columns=feature_names)
    return X

def load_model(path):
    """Load joblib model safely. Return tuple (model, scaler, feature_names, accuracy)."""
    data = joblib.load(path)
    if isinstance(data, dict):
        model = data.get('best_model') or data.get('model') or data.get('clf')
        scaler = data.get('scaler')
        feature_names = data.get('feature_names') or data.get('features') or []
        accuracy = data.get('accuracy') or data.get('acc')
        return model, scaler, feature_names, accuracy
    else:
        return data, None, [], None

def pretty_metric(value, fmt="${:,.2f}", na_text="—"):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return na_text
        return fmt.format(value)
    except:
        return str(value)

# -----------------------
# Load trained model
# -----------------------
MODEL_FILENAME = "best_stock_model.pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

try:
    model, scaler, expected_features, model_accuracy = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Make sure {MODEL_FILENAME} exists in the app folder.")
    st.stop()
except Exception as e:
    st.error(f"Failed loading model: {e}")
    st.stop()

# -----------------------
# Page config & CSS
# -----------------------
st.set_page_config(page_title="Stock Trend Predictor", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .app-title { font-size: 36px; font-weight:800; color: #0b5394; text-align:center; margin-bottom:6px}
    .app-sub { text-align:center; color:#4b4b4b; margin-bottom:18px }
    .card { background: #ffffff; border-radius:12px; padding:18px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
    .up { color: #16a085; font-weight:700; font-size:22px }
    .down { color: #c0392b; font-weight:700; font-size:22px }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<div class='app-title'>📊 Stock Trend Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='app-sub'>Enter a ticker, fetch recent data, and predict if it will go <b>UP</b> or <b>DOWN</b> tomorrow</div>", unsafe_allow_html=True)

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("🔎 Settings")
    ticker = st.text_input("Ticker (Yahoo symbol)", value="AAPL")
    period = st.selectbox("Data period", options=["6mo", "1y", "2y", "5y"], index=1)
    interval = st.selectbox("Interval", options=["1d", "1wk"], index=0)
    show_indicators = st.checkbox("Show technical indicators panel", value=True)
    st.markdown("---")
    if model_accuracy:
        st.success(f"Saved model metric (train F1 or acc): {model_accuracy:.3f}")
    st.caption("Tip: Use exchange suffixes for non-US tickers (e.g., INFY.NS).")

# -----------------------
# Predict button
# -----------------------
cols = st.columns([3,1])
with cols[0]:
    do_predict = st.button("🚀 Predict Next Day Movement", use_container_width=True)
with cols[1]:
    st.write(" ")

if do_predict:
    with st.spinner("Fetching data and computing indicators..."):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    if df is None or df.empty:
        st.error("No data returned. Check ticker symbol and try again.")
        st.stop()

    # flatten possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # compute indicators
    df = compute_technical_indicators(df)
    df = df.dropna().reset_index()

    if df.shape[0] < 10:
        st.error("Not enough data after indicator computation. Try a longer period.")
        st.stop()

    # Build X_latest according to expected_features
    X_latest = build_X_latest(df.set_index('Date'), expected_features)

    # Apply scaler if available
    if scaler is not None:
        try:
            X_in = scaler.transform(X_latest)
        except Exception:
            # fallback: try converting to numeric and then transform
            X_in = scaler.transform(X_latest.fillna(0).astype(float))
    else:
        X_in = X_latest.fillna(0).astype(float).values

    # Prediction & probability
    try:
        pred = model.predict(X_in)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_in)[0]
            prob_up = prob[1] if len(prob) > 1 else None
        elif hasattr(model, "decision_function"):
            # map decision_function to probability-like via sigmoid
            score = model.decision_function(X_in)[0]
            prob_up = 1 / (1 + np.exp(-score))
        else:
            prob_up = None

        label = "UP 📈" if pred == 1 else "DOWN 📉"
        cls_css = "up" if pred == 1 else "down"

        # Top row: Prediction card + key metrics
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2,1,1,1])
        with c1:
            st.markdown(f"<h3 style='margin:0'>Prediction: <span class='{cls_css}'>{label}</span></h3>", unsafe_allow_html=True)
            if prob_up is not None:
                st.write(f"Confidence (prob. UP): **{prob_up:.2%}**")
        with c2:
            last_close = df['Adj Close'].iloc[-1]
            st.metric("Last Close", f"${last_close:.2f}")
        with c3:
            ma5 = df['MA5'].iloc[-1]
            st.metric("MA5", f"${ma5:.2f}")
        with c4:
            ma20 = df['MA20'].iloc[-1]
            st.metric("MA20", f"${ma20:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Price chart
        st.subheader(f"{ticker} Price Chart ({period})")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Adj Close'], name='Price'
        ))
        # add MAs
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA5'], mode='lines', name='MA5', line=dict(color='green', width=1)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1)))
        fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_white', height=520)
        st.plotly_chart(fig, use_container_width=True)

        # Indicators panel
        if show_indicators:
            st.subheader("Technical Indicators (latest values)")
            ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
            with ind_col1:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            with ind_col2:
                st.metric("MACD", f"{df['MACD'].iloc[-1]:.4f}")
            with ind_col3:
                st.metric("5-day Vol", f"{df['vol_5'].iloc[-1]:.4f}")
            with ind_col4:
                st.metric("Momentum(5)", f"{df['momentum_5'].iloc[-1]:.2f}")

        # Optional: feature importance if available (RandomForest/XGBoost)
        fi = None
        try:
            if hasattr(model, 'feature_importances_') and expected_features:
                fi = pd.Series(model.feature_importances_, index=expected_features).sort_values(ascending=False)
            elif hasattr(model, 'coef_') and expected_features:
                coef = np.abs(model.coef_).ravel()
                fi = pd.Series(coef, index=expected_features).sort_values(ascending=False)
        except Exception:
            fi = None

        if fi is not None:
            st.subheader("Feature Importance")
            st.bar_chart(fi.head(12))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("<div style='text-align:center;color:gray'>💡Developed by <b>Pranil Bankar</b> — Stock Up/Down Classifier</div>", unsafe_allow_html=True)

