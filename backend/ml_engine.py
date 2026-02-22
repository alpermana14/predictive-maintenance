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
TARGETS = ["current", "temperature", "z_rms", "x_rms", "z_peak", "x_peak", "noise"]
FREQ = "30min"  #Change 30T to 30min
LAG_STEPS = 48
TEST_DAYS = 7
FORECAST_HORIZON = 12

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

# ===================== MODEL TRAINING =====================
def train_models(X_train, y_train, X_val, y_val):
    models = {}
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "seed": 42,
    }

    for tgt in TARGETS:
        train_set = lgb.Dataset(X_train, y_train[tgt])
        val_set = lgb.Dataset(X_val, y_val[tgt])
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(20)]
        )
        models[tgt] = model
    
    return models

# ===================== FORECASTING LOGIC =====================
def generate_forecast(models, df, X_cols):
    """Recursive forecasting loop."""
    buffer = df.iloc[-LAG_STEPS:].copy()
    future_index = pd.date_range(
        start=df.index[-1] + pd.Timedelta(FREQ),
        periods=FORECAST_HORIZON, 
        freq=FREQ
    )
    
    forecast_dict = {tgt: [] for tgt in TARGETS}

    for step in range(FORECAST_HORIZON):
        input_row = {}
        # Reconstruct lag features from buffer
        # Note: X_cols contains names like 'current_lag1', 'temperature_lag2'
        base_cols = [c.split("_lag")[0] for c in X_cols if "_lag" in c]
        base_cols = list(set(base_cols)) # unique

        for col in base_cols:
            if col in buffer.columns:
                for lag in range(1, LAG_STEPS + 1):
                    lag_col = f"{col}_lag{lag}"
                    if lag_col in X_cols:
                         # Get value from buffer (lag steps back)
                         # iloc[-1] is most recent, iloc[-lag] goes back
                         input_row[lag_col] = buffer.iloc[-lag][col]

        # Create single-row DataFrame for prediction
        X_pred = pd.DataFrame([input_row]).reindex(columns=X_cols).fillna(0)
        
        # Predict all targets for this step
        preds_step = {}
        for tgt in TARGETS:
            val = float(models[tgt].predict(X_pred)[0])
            preds_step[tgt] = val
            forecast_dict[tgt].append(val)

        # Update buffer with predicted values for next recursion
        new_row = buffer.iloc[-1].copy()
        for tgt in TARGETS:
            new_row[tgt] = preds_step[tgt]
        
        buffer = pd.concat([buffer, new_row.to_frame().T])
        buffer = buffer.iloc[-LAG_STEPS:] # Keep buffer size constant

    # Convert to DataFrame
    forecast_df = pd.DataFrame(forecast_dict, index=future_index)
    return forecast_df

# ===================== ANOMALY DETECTION (IDK) =====================
def detect_anomalies(df):
    """Runs IDK sliding window on all targets."""
    idk_scores = {}
    N = 144  # Lookback window (approx 30 days of 30min data)
    
    for sig in TARGETS:
        # Prepare data shape for IDK
        X = np.array(df[sig]).reshape(-1, 1)
        X = X[-min(len(X), N):]
        
        # Run IDK
        scores = IDK_square_sliding(
            X,
            t=100,
            psi1=4,
            width=20,
            psi2=4
        )
        idk_scores[sig] = scores.flatten()
        
    return idk_scores

# ===================== MAIN PIPELINE =====================
def run_pipeline(df):
    """Orchestrator function called by main.py."""
    print("[INFO] Feature Engineering...")
    df_numeric = df[TARGETS].copy()
    data = make_lag_features(df_numeric, LAG_STEPS)
    X = data.drop(columns=TARGETS)
    y = data[TARGETS]

    # Split (Simple 80/20 for retraining)
    split_idx = int(len(X) * 0.9) 
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print("[INFO] Training Models...")
    models = train_models(X_train, y_train, X_val, y_val)

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