import os
import requests
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin



print(f"🚀 load_data.py triggered at {datetime.now().isoformat()}")

# ======================================================
# 1️⃣ Config
# ======================================================
symbol = "TRX-USD"
market = "cadli"
start_date = "2024-12-30"
end_date = datetime.now().strftime("%Y-%m-%d")

url = (
    f"https://data-api.coindesk.com/index/cc/v1/historical/days"
    f"?market={market}&instrument={symbol}"
    f"&limit=400&aggregate=1&fill=true&apply_mapping=true&response_format=JSON"
)

os.makedirs("data", exist_ok=True)
print(f"📡 Fetching TRON OHLC data from Coindesk ({start_date} → {end_date})...")

# ======================================================
# 2️⃣ Fetch data from API
# ======================================================
response = requests.get(url)
response.raise_for_status()
data = response.json().get("Data", [])

if not data:
    raise Exception("❌ No TRON data returned from Coindesk")

df = pd.DataFrame(data)
df.columns = df.columns.str.lower()

if "timestamp" not in df.columns:
    raise Exception(f"❌ No 'timestamp' column found: {df.columns.tolist()}")

# ======================================================
# 3️⃣ Clean and format
# ======================================================
df["timeOpen"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
df = df.rename(columns={
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume"
})
df = df[["timeOpen", "open", "high", "low", "close", "volume"]].dropna()
df = df[(df["timeOpen"] >= start_date) & (df["timeOpen"] <= end_date)].reset_index(drop=True)

# Save raw data
raw_path = "data/tron_ohlc_raw_2025.csv"
df.to_csv(raw_path, index=False)
print(f"✅ Saved raw TRON OHLC data → {raw_path} ({len(df)} rows)")

# ======================================================
# 4️⃣ Lag Feature Transformer
# ======================================================
class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Create lag and lead features for time series.
    - open_t-2, close_t-2, volume_t-1
    - high_t+1 (lead target)
    """
    def __init__(self, close_lag=2, volume_lag=1, sort_col='timeOpen'):
        self.close_lag = close_lag
        self.volume_lag = volume_lag
        self.sort_col = sort_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Sort
        if self.sort_col in X.columns:
            X = X.sort_values(by=self.sort_col).reset_index(drop=True)

        # Create lag features
        X["open_t-2"] = X["open"].shift(self.close_lag)
        X["close_t-2"] = X["close"].shift(self.close_lag)
        X["volume_t-1"] = X["volume"].shift(self.volume_lag)

        # Create lead target
        if "high" in X.columns:
            X["high_t1"] = X["high"].shift(-1)

        # Drop NaN
        X = X.dropna().reset_index(drop=True)

        # Drop original columns to avoid leakage
        X = X.drop(columns=["open", "close", "volume", "high"], errors="ignore")

        return X

# ======================================================
# 5️⃣ Transform data and split X/y
# ======================================================
transformer = LagFeatureTransformer()
transformed = transformer.fit_transform(df)

# y = next-day high (lead target)
y_new = transformed.pop("high_t1")
X_new = transformed

# Save both
X_path = "data/tron_X_train_new_2025.csv"
y_path = "data/tron_y_train_new_2025.csv"
X_new.to_csv(X_path, index=False)
y_new.to_csv(y_path, index=False, header = False)

print(f"✅ Saved engineered features → {X_path}")
print(f"✅ Saved target variable → {y_path}")
print(f"📈 Shape: X={X_new.shape}, y={y_new.shape}")

print("🎯 Data preparation complete!")
