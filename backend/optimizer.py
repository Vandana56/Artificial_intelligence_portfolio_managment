"""Mean-variance (Markowitz) optimisation and the true efficient frontier."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TD = 252


def _inputs(returns: pd.DataFrame):
    mu = returns.mean().values * TD
    cov = returns.cov().values * TD + 1e-10 * np.eye(returns.shape[1])  # tiny ridge: stability
    return mu, cov


def _perf(w, mu, cov, rf):
    r = float(w @ mu)
    v = float(np.sqrt(w @ cov @ w))
    return r, v, (r - rf) / v if v else 0.0


def _solve(fun, n, extra_cons=(), x0=None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra_cons]
    x0 = np.ones(n) / n if x0 is None else x0
    return minimize(fun, x0, bounds=[(0, 1)] * n, constraints=cons, method="SLSQP",
                    options={"maxiter": 300, "ftol": 1e-10})


def optimize(returns: pd.DataFrame, rf=0.04, objective="sharpe") -> dict:
    mu, cov = _inputs(returns)
    n = len(mu)
    if objective == "sharpe":
        fun = lambda w: -_perf(w, mu, cov, rf)[2]
    else:
        fun = lambda w: float(w @ cov @ w)          # variance: smoother than volatility
    res = _solve(fun, n)
    w = np.clip(res.x, 0, 1) if res.success else np.ones(n) / n
    w = w / w.sum()
    r, v, s = _perf(w, mu, cov, rf)
    return {"weights": pd.Series(w, index=returns.columns), "return": r, "vol": v, "sharpe": s}


def efficient_frontier(returns: pd.DataFrame, rf=0.04, n_random=5000, n_points=40, seed=0) -> dict:
    """Random long-only portfolios (vectorised cloud) + the true frontier, found by
    minimising variance at each target return from the min-vol point to the best asset."""
    mu, cov = _inputs(returns)
    n = len(mu)

    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(n), size=n_random)                 # (n_random, n)
    ret = W @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    cloud = pd.DataFrame({"Volatility": vol, "Return": ret, "Sharpe": (ret - rf) / vol})

    mv = optimize(returns, rf, "vol")
    targets = np.linspace(mv["return"], mu.max(), n_points)
    pts, x0 = [], mv["weights"].values
    for t in targets:
        res = _solve(lambda w: float(w @ cov @ w), n,
                     extra_cons=[{"type": "eq", "fun": lambda w, t=t: w @ mu - t}], x0=x0)
        if res.success:
            x0 = res.x                                           # warm start next target
            pts.append((float(np.sqrt(res.x @ cov @ res.x)), float(res.x @ mu)))
    frontier = pd.DataFrame(pts, columns=["Volatility", "Return"])
    return {"random": cloud, "frontier": frontier}
