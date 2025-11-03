import os
import requests
import pandas as pd
import numpy as np
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
df = df[["timeOpen", "open", "high", "low", "close", "volume"]].dropna()
df = df[(df["timeOpen"] >= start_date) & (df["timeOpen"] <= end_date)].reset_index(drop=True)

# ======================================================
# 4️⃣ Add technical indicators
# ======================================================
def add_technical_indicators(df):
    df = df.copy()

    # MA9
    df['MA9'] = df['close'].rolling(window=9, min_periods=1).mean()

    # Bollinger Bands (20 & 50)
    for period in [20, 50]:
        df[f'BB{period}_MA'] = df['close'].rolling(window=period, min_periods=1).mean()
        df[f'BB{period}_STD'] = df['close'].rolling(window=period, min_periods=1).std()
        df[f'BB{period}_upper'] = df[f'BB{period}_MA'] + 2 * df[f'BB{period}_STD']
        df[f'BB{period}_lower'] = df[f'BB{period}_MA'] - 2 * df[f'BB{period}_STD']

    # RSI (14)
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic RSI (14)
    lowest_rsi = df['RSI'].rolling(window=14, min_periods=1).min()
    highest_rsi = df['RSI'].rolling(window=14, min_periods=1).max()
    df['stoch_rsi'] = (df['RSI'] - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-9)

    # MACD (12, 26, 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14, min_periods=1).mean()

    # OBV
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Drop BB STD columns
    df = df.drop(columns=['BB20_STD', 'BB50_STD'], errors='ignore')
    return df

df = add_technical_indicators(df)

# ======================================================
# 5️⃣ Lag Feature Transformer (follow notebook logic)
# ======================================================
class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Create lag features following notebook logic:
    - close_lag = 2 (for prices, indicators)
    - volume_lag = 1 (for volume, ATR)
    """
    def __init__(self, close_lag=2, volume_lag=1, sort_col='timeOpen'):
        self.close_lag = close_lag
        self.volume_lag = volume_lag
        self.sort_col = sort_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.sort_col in X.columns:
            X = X.sort_values(by=self.sort_col).reset_index(drop=True)

        # 🎯 Lead target (next-day high)
        X["high_t1"] = X["high"].shift(-1)

        # ✅ Create lag features
        X['open_t-2'] = X['open'].shift(self.close_lag)
        X['close_t-2'] = X['close'].shift(self.close_lag)
        X['volume_t-1'] = X['volume'].shift(self.volume_lag)
        X['low_t-2'] = X['low'].shift(self.close_lag)
    

        # Indicators
        X['MA9-2'] = X['MA9'].shift(self.close_lag)

        X['BB20_MA-2'] = X['BB20_MA'].shift(self.close_lag)
        X['BB20_upper-2'] = X['BB20_upper'].shift(self.close_lag)
        X['BB20_lower-2'] = X['BB20_lower'].shift(self.close_lag)

        X['BB50_MA-2'] = X['BB50_MA'].shift(self.close_lag)
        X['BB50_upper-2'] = X['BB50_upper'].shift(self.close_lag)
        X['BB50_lower-2'] = X['BB50_lower'].shift(self.close_lag)

        X['stoch_rsi-2'] = X['stoch_rsi'].shift(self.close_lag)
        X['MACD-2'] = X['MACD'].shift(self.close_lag)
        X['MACD_signal-2'] = X['MACD_signal'].shift(self.close_lag)
        X['MACD_hist-2'] = X['MACD_hist'].shift(self.close_lag)
        X['RSI-2'] = X['RSI'].shift(self.close_lag)
        X['ATR-1'] = X['ATR'].shift(self.volume_lag)
        X['OBV-2'] = X['OBV'].shift(self.close_lag)

        # 🧹 Drop original columns (to prevent leakage)
        drop_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'MA9', 'BB20_MA', 'BB20_upper', 'BB20_lower',
            'BB50_MA', 'BB50_upper', 'BB50_lower',
            'stoch_rsi', 'MACD', 'MACD_signal', 'MACD_hist',
            'RSI', 'ATR', 'OBV'
        ]
        X = X.drop(columns=drop_cols, errors='ignore')

        # 🧽 Clean NaN rows
        X = X.dropna().reset_index(drop=True)
        return X

# ======================================================
# 6️⃣ Transform and split
# ======================================================
transformer = LagFeatureTransformer(close_lag=2, volume_lag=1)
transformed = transformer.fit_transform(df)

# Split target
y_new = transformed.pop("high_t1")
X_new = transformed

# ======================================================
# 7️⃣ Save
# ======================================================
raw_path = "data/tron_ohlc_raw_2025.csv"
X_path = "data/tron_X_train_new_2025.csv"
y_path = "data/tron_y_train_new_2025.csv"

df.to_csv(raw_path, index=False)
X_new.to_csv(X_path, index=False)
y_new.to_csv(y_path, index=False, header=False)

print(f"✅ Saved raw data with indicators → {raw_path}")
print(f"✅ Saved lag features → {X_path}")
print(f"✅ Saved target variable → {y_path}")
print(f"📈 Shape: X={X_new.shape}, y={y_new.shape}")
print("🎯 Data preparation complete!")
