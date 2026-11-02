"""Mean-variance (Markowitz) optimisation + efficient frontier."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TD = 252


def _stats(w, mu, cov, rf):
    r = float(w @ mu); v = float(np.sqrt(w @ cov @ w))
    return r, v, (r - rf) / v if v else 0.0


def optimize(returns: pd.DataFrame, rf=0.04, objective="sharpe"):
    mu, cov = returns.mean().values * TD, returns.cov().values * TD
    n = len(mu)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bnds = [(0, 1)] * n
    f = (lambda w: -_stats(w, mu, cov, rf)[2]) if objective == "sharpe" else (lambda w: _stats(w, mu, cov, rf)[1])
    res = minimize(f, np.ones(n) / n, bounds=bnds, constraints=cons, method="SLSQP")
    r, v, s = _stats(res.x, mu, cov, rf)
    return {"weights": pd.Series(res.x, index=returns.columns), "return": r, "vol": v, "sharpe": s}


def random_frontier(returns: pd.DataFrame, rf=0.04, n=2000, seed=0) -> pd.DataFrame:
    mu, cov = returns.mean().values * TD, returns.cov().values * TD
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(len(mu)))
        r, v, s = _stats(w, mu, cov, rf)
        rows.append((v, r, s))
    return pd.DataFrame(rows, columns=["Volatility", "Return", "Sharpe"])
