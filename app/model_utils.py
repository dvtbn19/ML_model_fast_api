# ======================================================
# app/model_utils.py (Fixed Feature Mismatch)
# ======================================================

import pandas as pd
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import sys

# 👇 Import custom transformers so joblib can recognize them
from app.custom_transformers import YearExtractor, StandardScalerTransformer
import app.custom_transformers

# 👇 Alias to avoid "Can't get attribute ..." when loading joblib model
sys.modules["__main__"] = app.custom_transformers


# ======================================================
# PATH CONFIGURATION
# ======================================================
MODEL_PATH = Path("models/best_model.joblib")
X_NEW_PATH = Path("data/tron_X_train_new_2025.csv")
Y_NEW_PATH = Path("data/tron_y_train_new_2025.csv")
RAW_PATH = Path("data/tron_ohlc_raw_2025.csv")


# ======================================================
# 1️⃣ Load model
# ======================================================
def load_model():
    """Load the trained model pipeline or create a new Ridge model."""
    if MODEL_PATH.exists():
        print(f"✅ Loaded model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    else:
        print("⚠️ Model not found — creating a new Ridge model...")
        return Ridge(alpha=0.5)


# ======================================================
# 2️⃣ Fine-tune model with new data
# ======================================================
def fine_tune_model():
    """
    Retrain (fine-tune) the model using the latest engineered dataset.
    """
    if not X_NEW_PATH.exists() or not Y_NEW_PATH.exists():
        raise FileNotFoundError("❌ Missing feature or target file. Please run data/load_data.py first.")

    print("🔄 Fine-tuning model with latest data...")

    # Load processed data
    X_new = pd.read_csv(X_NEW_PATH)
    y_new = pd.read_csv(Y_NEW_PATH, header=None).squeeze("columns")

    # 🔍 Drop unnecessary columns if any remain
    drop_cols = ["high_t+1", "high_t1", "high"]
    X_new = X_new.drop(columns=[c for c in drop_cols if c in X_new.columns], errors="ignore")

    # Load or create model
    model = load_model()

    # Fit model
    model.fit(X_new, y_new)

    # Evaluate training RMSE
    preds = model.predict(X_new)
    rmse = mean_squared_error(y_new, preds, squared=False)
    print(f"📉 Fine-tuned RMSE on new data: {rmse:.6f}")

    # Save updated model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_path = Path(f"models/best_model_{timestamp}.joblib")
    joblib.dump(model, new_model_path)
    print(f"✅ Updated model saved to: {new_model_path}")

    return model, new_model_path, {"RMSE_train": rmse}


# ======================================================
# 3️⃣ Predict next-day HIGH
# ======================================================
def predict_next_high():
    """
    Predict next-day HIGH (T+1) using the latest model and the last available data point.
    """
    if not X_NEW_PATH.exists():
        raise FileNotFoundError("❌ Feature data not found. Please run load_data.py first.")

    # Load processed features and model
    X_all = pd.read_csv(X_NEW_PATH)
    model = load_model()

    # ✅ Select the last available row
    X_pred = X_all.tail(1).copy()

    # ✅ Remove columns that were not part of training
    drop_cols = ["low", "high", "high_t+1", "high_t1"]
    X_pred = X_pred.drop(columns=[c for c in drop_cols if c in X_pred.columns], errors="ignore")

    # Identify feature alignment
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        missing = [f for f in model_features if f not in X_pred.columns]
        extra = [f for f in X_pred.columns if f not in model_features]
        if missing:
            print(f"⚠️ Warning: Missing features in prediction data → {missing}")
        if extra:
            print(f"⚠️ Dropping extra features not used in training → {extra}")
            X_pred = X_pred.drop(columns=extra, errors="ignore")

    # Predict next-day high
    predicted_high = float(model.predict(X_pred)[0])

    # Get dates
    last_date = pd.read_csv(RAW_PATH)["timeOpen"].iloc[-1]
    next_day = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"📅 Using data up to {last_date} to forecast high for {next_day}")
    print(f"🔮 Predicted next-day high: {predicted_high:.5f}")

    return {
        "predicted_high": round(predicted_high, 5),
        "last_data_date": str(pd.to_datetime(last_date).date()),
        "predicted_for": next_day,
        "status": "success"
    }
