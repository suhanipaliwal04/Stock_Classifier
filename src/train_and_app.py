# train_and_app.py
import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Data fetching
try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError("Please install yfinance: pip install yfinance") from e

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans  # included for experimentation (unsupervised)
# Extra algorithms:
try:
    import xgboost as xgb
except:
    xgb = None
try:
    import lightgbm as lgb
except:
    lgb = None

# For plotting (EDA)
import matplotlib.pyplot as plt

# -------------------------
# 1) Fetch historical data
# -------------------------
def fetch_stock(ticker="AAPL", period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.dropna()
    return df

# -------------------------
# 2) Feature engineering
# -------------------------
def add_technical_indicators(df):
    df = df.copy()
    # Returns
    df['Return'] = df['Adj Close'].pct_change()
    # Lag features
    for lag in [1,2,3,5]:
        df[f'Return_lag_{lag}'] = df['Return'].shift(lag)
    # Moving averages
    df['MA5'] = df['Adj Close'].rolling(5).mean()
    df['MA10'] = df['Adj Close'].rolling(10).mean()
    df['MA20'] = df['Adj Close'].rolling(20).mean()
    df['MA_ratio_5_20'] = df['MA5'] / df['MA20']
    # Volatility
    df['vol_5'] = df['Return'].rolling(5).std()
    df['vol_10'] = df['Return'].rolling(10).std()
    # Momentum: simple difference
    df['momentum_5'] = df['Adj Close'] - df['Adj Close'].shift(5)
    # RSI (simple implementation)
    delta = df['Adj Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
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

# -------------------------
# 3) Create target
# -------------------------
def create_target(df):
    print("Columns in df:", df.columns)
    # Create a new column for next day's adjusted close
    df['NextClose'] = df['Adj Close'].shift(-1)
    # Drop last row since it has no NextClose value
    df = df.dropna(subset=['NextClose'])
    # Align both columns before comparison to avoid ValueError
    df = df.reset_index(drop=True)
    # Create binary target: 1 if next day's price is higher, else 0
    df['Target'] = (df['NextClose'] > df['Adj Close']).astype(int)
    return df

# -------------------------
# 4) Prepare dataset
# -------------------------
def prepare_dataset(ticker="AAPL", period="5y"):
    df = fetch_stock(ticker=ticker, period=period)
    df = add_technical_indicators(df)
    df = create_target(df)
    df = df.dropna()
    features = [
        'Return_lag_1','Return_lag_2','Return_lag_3','Return_lag_5',
        'MA5','MA10','MA20','MA_ratio_5_20','vol_5','vol_10',
        'momentum_5','RSI','MACD','dow'
    ]
    X = df[features]
    y = df['Target']
    return df, X, y

# -------------------------
# 5) Train + compare models
# -------------------------
def train_and_compare(X, y, results_path="results.csv", model_out="best_model.joblib"):
    # Split time-aware: use shuffle=False if you want time-split; here we use stratified random split for classification practice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    models['LogisticRegression'] = LogisticRegression(max_iter=1000)
    models['DecisionTree'] = DecisionTreeClassifier(random_state=42)
    models['KNN'] = KNeighborsClassifier()
    models['SVM'] = SVC(probability=True, random_state=42)
    models['RandomForest'] = RandomForestClassifier(random_state=42)
    models['MLP'] = MLPClassifier(max_iter=500, random_state=42)
    if xgb is not None:
        models['XGBoost'] = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, verbosity=0)
    if lgb is not None:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42)

    # Hyperparameter grids (small for brevity; expand for final submission)
    param_grids = {
        'LogisticRegression': {'C':[0.01,0.1,1,10]},
        'DecisionTree': {'max_depth':[3,5,8,None], 'min_samples_split':[2,5,10]},
        'KNN': {'n_neighbors':[3,5,7]},
        'SVM': {'C':[0.1,1,10], 'kernel':['rbf','linear']},
        'RandomForest': {'n_estimators':[50,100], 'max_depth':[5,10,None]},
        'MLP': {'hidden_layer_sizes':[(50,),(100,)], 'alpha':[0.0001,0.001]}
    }
    if xgb is not None:
        param_grids['XGBoost'] = {'n_estimators':[50,100], 'max_depth':[3,5], 'learning_rate':[0.01,0.1]}
    if lgb is not None:
        param_grids['LightGBM'] = {'n_estimators':[50,100], 'num_leaves':[31,50]}

    results = []
    best_score = -1
    best_model = None
    best_model_name = None

    for name, model in models.items():
        print(f"Training {name} ...")
        grid = GridSearchCV(model, param_grids.get(name, {}), cv=3, scoring='f1', n_jobs=-1) if param_grids.get(name) else None
        if grid is not None:
            grid.fit(X_train_scaled, y_train)
            clf = grid.best_estimator_
            print(f"Best params for {name}: {grid.best_params_}")
        else:
            clf = model
            clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test_scaled)[:,1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = roc_auc_score(y_test, clf.decision_function(X_test_scaled))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({
            'model': name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc
        })
        if f1 > best_score:
            best_score = f1
            best_model = clf
            best_model_name = name

    results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
    results_df.to_csv(results_path, index=False)
    # Save best model + scaler
    # joblib.dump({'model':best_model, 'scaler':scaler, 'features':X.columns.tolist()}, model_out)
    joblib.dump({
    "best_model": best_model,
    "feature_names": list(X.columns)
    }, "best_stock_model.pkl")
    print("Best model:", best_model_name, "saved to", model_out)
    return results_df, model_out

# -------------------------
# 6) Example run (training)
# -------------------------
if __name__ == "__main__" and os.getenv("RUN_STAGE","train")=="train":
    ticker = "AAPL"      # change as needed
    df, X, y = prepare_dataset(ticker=ticker, period="5y")
    print("Data prepared. Rows:", len(df))
    results_df, saved_path = train_and_compare(X, y, results_path="results.csv", model_out="best_model.joblib")
    print(results_df)
