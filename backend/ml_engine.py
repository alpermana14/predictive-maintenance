import os
import pandas as pd
import numpy as np
import mysql.connector
import lightgbm as lgb
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error

# Import your custom IDK function
# Ensure IDK_square_sliding.py is in the same folder as this file
try:
    from IDK_square_sliding import IDK_square_sliding
except ImportError:
    # Fallback mock if file is missing during testing
    print("[WARN] IDK module not found. Using mock anomaly detection.")
    def IDK_square_sliding(X, t, psi1, width, psi2):
        return np.random.rand(len(X), 1)

# ===================== ENV & CONSTANTS =====================
load_dotenv()

TABLE_NAME = "conveyor"
# Motor current is not monitored: the fitted Rogowski-coil sensor (1000 A range)
# cannot measure the 1-2 A drawn by the 450 W conveyor motor.
# z_peak / x_peak are peak acceleration in g (not velocity).
TARGETS = ["temperature", "z_rms", "x_rms", "z_peak", "x_peak", "noise"]
UNITS = {"temperature": "°C", "z_rms": "mm/s", "x_rms": "mm/s",
         "z_peak": "g", "x_peak": "g", "noise": "dB"}
FREQ = "30min"  #Change 30T to 30min
LAG_STEPS = 48
TEST_DAYS = 7
FORECAST_HORIZON = 12

# s-IDK^2 settings (Section 4.3 of the paper; CAEE decision D16): psi = 2 is the smallest
# value of the search space in [24], omega = 20 windows = one 10-hour operating cycle, and
# the scored population is the last 144 points (3 days at 30-minute resolution).
IDK_PSI = 2
IDK_WIDTH = 20
IDK_T = 100
IDK_POPULATION = 144

# ---------------------------------------------------------------------------
# LightGBM configuration.
#
# "tuned" is the equal-budget Optuna configuration reported in Table 2 of the
# paper and is the default, so the running system uses the hyperparameters the
# paper reports (CAEE revision, 14 Sep 2026). "deployed" is the earlier generic
# setting (learning rate 0.05, up to 500 rounds with early stopping), kept only
# so that older timing runs (E10) can be reproduced.
#
# Only the hyperparameters change. Feature construction is identical: lag
# features of all channels (Eq. 8) and one-step-ahead absolute targets (Eq. 9).
#
#   PM_LGB_CONFIG=tuned      (default, Table 2)
#   PM_LGB_CONFIG=deployed   (earlier generic setting)
# ---------------------------------------------------------------------------
LGB_CONFIG = os.environ.get("PM_LGB_CONFIG", "tuned").lower()

_LGB_DEPLOYED = {
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "seed": 42,
    },
    "num_boost_round": 500,
    "early_stopping_rounds": 20,
}

# Table 2 values of the CAEE revision (six measured channels, 15 Optuna TPE trials,
# validation 6-13 July 2026; experiments/e01b_forecasting_modelcomp.py). n_estimators
# becomes num_boost_round and early stopping is disabled, so all 158 rounds run.
# Note: LightGBM applies "subsample" only when bagging_freq > 0, which is not set here
# (nor in the tuning run), so bagging is inactive in both.
_LGB_TUNED = {
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.028181,
        "num_leaves": 209,
        "min_child_samples": 70,
        "subsample": 0.776061,
        "colsample_bytree": 0.648815,
        "reg_lambda": 0.095655,
        "verbosity": -1,
        "seed": 42,
    },
    "num_boost_round": 158,
    "early_stopping_rounds": None,
}

_LGB_SETTINGS = {"deployed": _LGB_DEPLOYED, "tuned": _LGB_TUNED}
if LGB_CONFIG not in _LGB_SETTINGS:
    raise ValueError(
        f"PM_LGB_CONFIG must be 'deployed' or 'tuned', got {LGB_CONFIG!r}")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# ===================== DATA LOADING =====================
def apply_imputation(df, target_columns):
    """
    Fills missing sensor values using Linear Interpolation.
    Mitigates errors in LGBM training and IDK scoring.
    """
    if df.empty:
        return df

    for col in target_columns:
        if col in df.columns:
            # Create a flag: True if the data is missing (NaN), False if it is real
            # We name it [sensor_name]_flag
            df[f"{col}_error_flag"] = df[col].isna()

    # 1. Linear Interpolation
    # This fills gaps of any size by drawing a line between known points.
    # We use limit_direction='both' to handle gaps at the start of the series.
    df_imputed = df.interpolate(method='linear', limit_direction='both')

    # 2. Safety Fallback: Forward/Backward Fill
    # If the sensor was missing at the very first or very last row, 
    # interpolation has no 'anchor' point. ffill/bfill fixes this.
    df_imputed = df_imputed.ffill().bfill()

    # 3. Final Fallback: Constant Zero/Median
    # In the rare case a column is ENTIRELY null, fill with 0 to prevent ML crash.
    df_imputed = df_imputed.fillna(0)

    return df_imputed

def load_conveyor_data():
    """Fetches and cleans data from MySQL."""
    print("[INFO] Connecting to MySQL...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        # Limit query for performance if needed, or select all
        query = f"SELECT * FROM {TABLE_NAME} WHERE conveyor_id > 1079" 
        df = pd.read_sql(query, conn)
        conn.close()

        # Timestamp conversion
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["datetime"] = df["datetime"].dt.tz_convert("Asia/Singapore")
        df["datetime"] = df["datetime"].dt.tz_localize(None)

        df = df.sort_values("datetime").reset_index(drop=True)
        
        # Resample to 30min intervals
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.round("30min")
        #df = df.groupby("datetime").first().resample("30min").ffill() #make the Nan value filled with last value
        df = df.groupby("datetime").first().resample("30min").asfreq() # After grouping/resampling, gaps appear as NaNs
        
        # --- NEW FIX: FORCE COLUMNS TO BE NUMBERS ---
        # This prevents the "Cannot interpolate with str dtype" error
        for col in TARGETS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # --------------------------------------------

        # --- SCENARIO 2 FIX: DROP TEXT COLUMNS BEFORE MATH ---
        # We must remove text before apply_imputation runs
        cols_to_drop = ["conveyor_id", "category", "status"] 
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        df = apply_imputation(df, TARGETS)

        # Clean columns
        cols_to_drop = ["conveyor_id", "category"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).round(2)
        
        print(f"[INFO] Data Loaded: {len(df)} records")
        return df
        
    except Exception as e:
        print(f"[ERROR] DB Error: {e}")
        # Return empty structure to prevent crash
        return pd.DataFrame(columns=TARGETS)

# ===================== FEATURE ENGINEERING =====================
def make_lag_features(df, lag_steps):
    X = df.copy()
    for lag in range(1, lag_steps + 1):
        X = pd.concat([X, df.shift(lag).add_suffix(f"_lag{lag}")], axis=1)
    return X.dropna()


def build_supervised(df):
    """Lag features of all channels (Eq. 8) and one-step-ahead absolute targets (Eq. 9).

    Row t holds lags 1..LAG_STEPS of every channel as inputs and the value of each
    channel at t as its target. Multi-step forecasts are produced recursively.
    """
    data = make_lag_features(df[TARGETS].copy(), LAG_STEPS)
    return data.drop(columns=TARGETS), data[TARGETS]

# ===================== MODEL TRAINING =====================
def train_models(X_train, y_train, X_val=None, y_val=None):
    """One LightGBM model per channel. Early stopping is used only when the
    configuration defines it and a validation split is given."""
    models = {}
    setting = _LGB_SETTINGS[LGB_CONFIG]
    params = dict(setting["params"])
    n_rounds = setting["num_boost_round"]
    stop_rounds = setting["early_stopping_rounds"]
    use_val = bool(stop_rounds) and X_val is not None and len(X_val) > 0

    for tgt in TARGETS:
        train_set = lgb.Dataset(X_train, y_train[tgt])
        kwargs = {}
        if use_val:
            kwargs = {"valid_sets": [lgb.Dataset(X_val, y_val[tgt])],
                      "callbacks": [lgb.early_stopping(stop_rounds, verbose=False)]}
        models[tgt] = lgb.train(params, train_set, num_boost_round=n_rounds, **kwargs)

    return models

# ===================== FORECASTING LOGIC =====================
def generate_forecast(models, df, X_cols):
    """Recursive 12-step forecast: each predicted step becomes lag 1 of the next."""
    buffer = df[TARGETS].iloc[-LAG_STEPS:].reset_index(drop=True)
    future_index = pd.date_range(
        start=df.index[-1] + pd.Timedelta(FREQ),
        periods=FORECAST_HORIZON,
        freq=FREQ
    )

    forecast_dict = {tgt: [] for tgt in TARGETS}
    for _ in range(FORECAST_HORIZON):
        # iloc[-1] is the most recent value (lag 1), iloc[-lag] goes back in time.
        row = {f"{col}_lag{lag}": float(buffer.iloc[-lag][col])
               for col in TARGETS for lag in range(1, LAG_STEPS + 1)}
        X_pred = pd.DataFrame([row]).reindex(columns=X_cols)
        preds_step = {tgt: float(models[tgt].predict(X_pred)[0]) for tgt in TARGETS}
        for tgt in TARGETS:
            forecast_dict[tgt].append(preds_step[tgt])
        buffer = pd.concat([buffer, pd.DataFrame([preds_step])], ignore_index=True).iloc[-LAG_STEPS:]

    return pd.DataFrame(forecast_dict, index=future_index)

# ===================== ANOMALY DETECTION (IDK) =====================
def detect_anomalies(df):
    """Runs IDK sliding window on all targets."""
    idk_scores = {}

    for sig in TARGETS:
        # Prepare data shape for IDK: the last IDK_POPULATION points (3 days)
        X = np.array(df[sig]).reshape(-1, 1)
        X = X[-min(len(X), IDK_POPULATION):]

        # Run IDK; one similarity score per window, aligned to the window end
        scores = IDK_square_sliding(
            X,
            t=IDK_T,
            psi1=IDK_PSI,
            width=IDK_WIDTH,
            psi2=IDK_PSI
        )
        idk_scores[sig] = scores.flatten()
        
    return idk_scores

# ===================== MAIN PIPELINE =====================
def run_pipeline(df):
    """Orchestrator function called by main.py."""
    print("[INFO] Feature Engineering...")
    X, y = build_supervised(df)

    print("[INFO] Training Models...")
    if _LGB_SETTINGS[LGB_CONFIG]["early_stopping_rounds"]:
        # Early stopping on the most recent 10% of rows.
        split_idx = int(len(X) * 0.9)
        models = train_models(X.iloc[:split_idx], y.iloc[:split_idx],
                              X.iloc[split_idx:], y.iloc[split_idx:])
    else:
        # Fixed number of boosting rounds: use every available row.
        models = train_models(X, y)

    print("[INFO] Generating Forecast...")
    forecast_df = generate_forecast(models, df, X.columns)

    print("[INFO] Detecting Anomalies...")
    anomalies = detect_anomalies(df)

    # Calculate Feature Importance (Optional, simplified)
    importance = {}
    for tgt in TARGETS:
        imp = models[tgt].feature_importance()
        names = models[tgt].feature_name()
        # Return top 10 as list of dicts
        sorted_idx = np.argsort(imp)[::-1][:10]
        importance[tgt] = [{"feature": names[i], "importance": float(imp[i])} for i in sorted_idx]

    return df, forecast_df, anomalies, importance, models