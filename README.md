# AI Portfolio Management System
## Run
    pip install -r requirements.txt
    streamlit run app.py
Uses live Yahoo data; falls back to synthetic data if offline.

## Structure
    app.py                  Frontend (Streamlit UI)
    backend/service.py      Facade: run_analysis() - single entry point
    backend/data_loader.py  Prices (yfinance / synthetic)
    backend/risk.py         Vol, Sharpe, VaR, CVaR, drawdown
    backend/models.py       XGBoost forecaster + KMeans risk tiers
    backend/optimizer.py    Markowitz optimiser + frontier
