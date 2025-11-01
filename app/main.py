# ======================================================
# app/main.py  ✅ FINAL FIXED VERSION
# ======================================================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.model_utils import fine_tune_model, predict_next_high
from app.auto_pipeline import run_daily_pipeline  # ✅ correct import


# ======================================================
# APP CONFIGURATION
# ======================================================
app = FastAPI(
    title="TRON Forecast API",
    description="""
🚀 **TRON Price Forecast API**
This FastAPI service automatically fetches TRON (TRX-USD) data,
re-trains the predictive model, and forecasts the **next-day HIGH price**.

### 🔧 Features
- Auto-updates data and re-trains model daily
- Runs one full pipeline immediately when the API starts
- Allows manual retraining & instant prediction via REST API

### 🔗 Endpoints
- `/` → Overview of the project
- `/health/` → Health check
- `/predict/tron` → Predict next-day HIGH price
- `/retrain/` → Manually re-train model
""",
    version="2.0.0",
)


# ======================================================
# 1️⃣ ROOT ENDPOINT
# ======================================================
@app.get("/")
def root():
    """Display API overview."""
    return {
        "project": "TRON Forecast API",
        "description": "Predict next-day HIGH price for TRON using automated daily retraining and forecasting.",
        "endpoints": {
            "/": "Overview",
            "/health/": "Health check",
            "/predict/tron": "Predict next-day HIGH price",
            "/retrain/": "Manually fine-tune model",
        },
        "scheduler": "Runs daily at 00:00:10 Australia/Sydney",
        "github_repo": "https://github.com/dvtbn19/at3_fast_api",
        "last_update": datetime.now().isoformat(),
    }


# ======================================================
# 2️⃣ HEALTH CHECK
# ======================================================
@app.get("/health/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "✅ OK", "timestamp": datetime.now().isoformat()}


# ======================================================
# 3️⃣ PREDICTION ENDPOINT
# ======================================================
@app.get("/predict/{token}")
def predict(token: str):
    """Predict next-day high for TRON."""
    if token.lower() != "tron":
        raise HTTPException(status_code=400, detail="Only 'tron' is supported currently.")
    return predict_next_high()


# ======================================================
# 4️⃣ MANUAL RETRAIN ENDPOINT
# ======================================================
@app.post("/retrain/")
def retrain(background_tasks: BackgroundTasks):
    """Manually trigger model fine-tuning with latest available data."""
    def retrain_job():
        print("🔁 Manual retrain triggered via API...")
        fine_tune_model()

    background_tasks.add_task(retrain_job)
    return {"status": "🟢 Retrain started in background", "timestamp": datetime.now().isoformat()}


# ======================================================
# 5️⃣ DAILY SCHEDULER SETUP (auto retrain + predict)
# ======================================================
scheduler = AsyncIOScheduler(timezone=ZoneInfo("Australia/Sydney"))


@app.on_event("startup")
def on_startup():
    """Start scheduler and run full pipeline immediately on startup."""
    data_path = Path("data/tron_ohlc_raw_2025.csv")

    print(f"🚀 Starting TRON Forecast API at {datetime.now().isoformat()}")
    print(f"📂 Monitoring data file: {data_path.resolve()}")

    # ✅ Run pipeline immediately once when server starts
    print("🕛 Running full pipeline immediately on startup...")
    run_daily_pipeline(trigger_source="startup")

    # ✅ Schedule daily pipeline at 00:00:10 Australia/Sydney
    scheduler.add_job(
        run_daily_pipeline,
        CronTrigger(hour=11, minute=15, second=10),
        coalesce=True,
        max_instances=1,
        misfire_grace_time=60,
    )

    # 🧪 Schedule a one-time test run after 30 seconds (for testing)
    scheduler.add_job(
        run_daily_pipeline,
        run_date=datetime.now(ZoneInfo("Australia/Sydney")) + timedelta(seconds=30),
        kwargs={"trigger_source": "test_run"},
    )

    scheduler.start()
    print("⏰ Scheduler started: Daily retrain & predict at 00:00:10 (Australia/Sydney)")
    print("🧪 Test job scheduled to run in 30 seconds (for development only).")
    print("✅ Background scheduler initialized successfully.\n")


@app.on_event("shutdown")
def on_shutdown():
    """Stop scheduler gracefully when the API shuts down."""
    scheduler.shutdown()
    print("🛑 Scheduler stopped gracefully.")
