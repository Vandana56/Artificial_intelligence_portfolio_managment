"""Service facade: the ONLY module the frontend imports. Orchestrates all backend parts."""
import numpy as np
import pandas as pd
from . import data_loader, risk, models, optimizer


def run_analysis(tickers, weights=None, years=3, rf=0.04, use_live=True):
    prices, source = data_loader.get_prices(tickers, years, use_live)
    rets = prices.pct_change().dropna()
    w = np.ones(len(tickers)) / len(tickers) if weights is None else np.asarray(weights) / np.sum(weights)

    per_asset = pd.DataFrame({t: risk.risk_metrics(rets[t], rf) for t in tickers}).T
    per_asset = per_asset.join(models.cluster_risk_tiers(per_asset))   # Risk Score + Risk Tier
    port_ret = risk.portfolio_returns(rets, w)
    port = risk.risk_metrics(port_ret, rf)
    forecasts = {t: models.train_direction_model(prices[t]) for t in tickers}

    return {
        "source": source, "prices": prices, "returns": rets, "weights": pd.Series(w, index=tickers),
        "per_asset": per_asset, "portfolio": port,
        "portfolio_curve": (1 + port_ret).cumprod() * 100,
        "forecasts": forecasts, "model_name": models.MODEL_NAME,
        "max_sharpe": optimizer.optimize(rets, rf, "sharpe"),
        "min_vol": optimizer.optimize(rets, rf, "vol"),
        "frontier": optimizer.efficient_frontier(rets, rf),
        "corr": rets.corr(),
    }
