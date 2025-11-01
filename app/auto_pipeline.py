# ======================================================
# app/auto_pipeline.py (Final Refactored)
# ======================================================

import subprocess
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

from app.model_utils import fine_tune_model, predict_next_high


def run_daily_pipeline(trigger_source: str = "scheduled"):
    """
    Automated end-to-end pipeline for TRON high price forecasting.

    Steps:
      1️⃣ Fetch the latest OHLC data via data/load_data.py
      2️⃣ Fine-tune the existing model with the updated data
      3️⃣ Predict the next-day HIGH price
      4️⃣ Log all results (model path, metrics, prediction) to data/prediction_log.csv
    """
    data_file = Path("data/tron_ohlc_raw_2025.csv")
    prediction_log = Path("data/prediction_log.csv")

    print(f"\n🕛 [Pipeline] Started at {datetime.now().isoformat()} (triggered by: {trigger_source})")

    try:
        # Step 1️⃣: Fetch latest data
        print("🌐 [Step 1] Fetching latest data via load_data.py ...")
        subprocess.run(["python", "data/load_data.py"], check=True)

        # Wait briefly to ensure files are written completely
        print("⏳ Waiting for file write to complete...")
        time.sleep(3)

        # Step 2️⃣: Fine-tune model
        print("🔁 [Step 2] Fine-tuning model with updated data...")
        model, model_path, metrics = fine_tune_model()

        # Step 3️⃣: Predict next-day HIGH
        print("🔮 [Step 3] Predicting next-day HIGH ...")
        prediction_result = predict_next_high()

        # Step 4️⃣: Attach metadata
        prediction_result["timestamp"] = datetime.now().isoformat()
        prediction_result["model_path"] = str(model_path)
        prediction_result["trigger_source"] = trigger_source
        prediction_result["train_rmse"] = metrics.get("RMSE_train", None)

        # Step 5️⃣: Save prediction log
        df_new = pd.DataFrame([prediction_result])
        if prediction_log.exists():
            df_old = pd.read_csv(prediction_log)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv(prediction_log, index=False)
        print(f"✅ [Step 4] Prediction logged → {prediction_log.resolve()}")

        # Step 6️⃣: Summary
        print(
            f"📊 Summary → Predicted HIGH (T+1): {prediction_result['predicted_high']:.5f} | "
            f"Train RMSE={prediction_result.get('train_rmse')}"
        )

    except subprocess.CalledProcessError:
        print("❌ [Pipeline] Failed to run load_data.py (data fetch error).")
    except Exception as e:
        print(f"❌ [Pipeline] Error during daily pipeline execution: {e}")


# Optional: allow manual execution from CLI
if __name__ == "__main__":
    run_daily_pipeline(trigger_source="manual")
