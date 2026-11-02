"""Data layer: live prices via yfinance, with a synthetic fallback so the app always runs."""
import numpy as np
import pandas as pd


def _synthetic_prices(tickers, days=756, seed=42):
    rng = np.random.default_rng(seed)
    n = len(tickers)
    A = rng.normal(size=(n, n))
    corr = A @ A.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    vols = rng.uniform(0.15, 0.45, n) / np.sqrt(252)
    mus = rng.uniform(0.04, 0.18, n) / 252
    cov = corr * np.outer(vols, vols)
    rets = rng.multivariate_normal(mus, cov, size=days)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)


def get_prices(tickers, years=3, use_live=True):
    """Returns (prices DataFrame, source string)."""
    if use_live:
        try:
            import yfinance as yf
            df = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
            if isinstance(df, pd.Series):
                df = df.to_frame(tickers[0])
            df = df.dropna(how="all").ffill().dropna()
            if len(df) > 250 and df.shape[1] == len(tickers):
                return df[tickers], "Yahoo Finance (live)"
        except Exception:
            pass
    return _synthetic_prices(tickers, days=252 * years), "Synthetic demo data"
