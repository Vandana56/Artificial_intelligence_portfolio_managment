"""Risk analytics: volatility, Sharpe, VaR, CVaR, drawdown."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns).cumprod()
    return float((curve / curve.cummax() - 1).min())


def risk_metrics(returns: pd.Series, rf=0.04, conf=0.95) -> dict:
    ann_ret = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    var = np.percentile(returns, (1 - conf) * 100)
    cvar = returns[returns <= var].mean()
    return {
        "Annual Return": float(ann_ret),
        "Volatility": float(ann_vol),
        "Sharpe": float((ann_ret - rf) / ann_vol) if ann_vol else 0.0,
        "VaR 95% (1d)": float(var),
        "CVaR 95% (1d)": float(cvar),
        "Max Drawdown": max_drawdown(returns),
    }


def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    return returns.dot(np.asarray(weights))
