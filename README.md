### 1. Clone the repository
```
git clone https://github.com/dvtbn19/at3_fast_api
cd at3_fast_api

```
```
poetry install
poetry run pip install -r requirements.txt
```
### 2. To fun FastAPI app
```
poetry run uvicorn app.main:app --reload
```
### 3.  To access live production Render
```
access: https://at3-fast-api-4b7s.onrender.com/
Check health: https://at3-fast-api-4b7s.onrender.com/health
Predict: https://at3-fast-api-4b7s.onrender.com/predict/TRON
```


## FastAPI repo Organization

```
📦 AMLA_AT2 (FastAPI Weather Prediction API)
├── app/                          # Application package
│   ├── __pycache__/              # Python cache files (tự sinh)
│   └── main.py                   # FastAPI entrypoint
│
├── data/                         # Data utilities
│   └── load_data.py          # Script to fetch raw data
│
├── models/                       # Trained ML models
│   ├── best_model.joblib         # The best trained model
│
├── .cache.sqlite                 # Local cache (tự sinh)
├── .python-version               # Python version info (for pyenv/Poetry)
├── Dockerfile                    # Docker build instructions
├── poetry.lock                   # Poetry lockfile
├── pyproject.toml                # Poetry project configuration
└── requirements.txt              # Package dependencies 
```

--------
