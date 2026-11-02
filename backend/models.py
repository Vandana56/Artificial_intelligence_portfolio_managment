"""ML layer: XGBoost direction forecaster + scikit-learn KMeans risk-tier clustering."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    def _make_model():
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
    MODEL_NAME = "XGBoost"
except ImportError:  # fallback keeps the app usable
    from sklearn.ensemble import GradientBoostingClassifier
    def _make_model():
        return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05)
    MODEL_NAME = "GradientBoosting (xgboost not installed)"

HORIZON = 5  # predict 5-day-ahead direction


def build_features(price: pd.Series) -> pd.DataFrame:
    r = price.pct_change()
    f = pd.DataFrame(index=price.index)
    for k in (1, 5, 10, 20):
        f[f"ret_{k}"] = price.pct_change(k)
    f["vol_10"] = r.rolling(10).std()
    f["vol_20"] = r.rolling(20).std()
    f["ma_ratio"] = price / price.rolling(20).mean() - 1
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    return f


def train_direction_model(price: pd.Series) -> dict:
    X = build_features(price)
    y = (price.shift(-HORIZON) / price - 1 > 0).astype(int)
    data = X.join(y.rename("y")).dropna()
    # time-ordered split (no shuffling -> no look-ahead leakage)
    cut = int(len(data) * 0.8)
    tr, te = data.iloc[:cut], data.iloc[cut:]
    model = _make_model()
    model.fit(tr.drop(columns="y"), tr["y"])
    acc = accuracy_score(te["y"], model.predict(te.drop(columns="y")))
    p_up = float(model.predict_proba(X.dropna().iloc[[-1]])[0, 1])
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {"accuracy": float(acc), "prob_up": p_up, "importance": imp}


def cluster_risk_tiers(metrics: pd.DataFrame) -> pd.Series:
    """metrics: rows=tickers, cols include Volatility, Max Drawdown, VaR 95% (1d)."""
    cols = ["Volatility", "Max Drawdown", "VaR 95% (1d)"]
    n = len(metrics)
    k = min(3, n)
    if k < 2:
        return pd.Series(["Medium"], index=metrics.index)
    Z = StandardScaler().fit_transform(metrics[cols].abs())
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    order = pd.Series(metrics["Volatility"].values).groupby(labels).mean().sort_values().index
    names = ["Low", "Medium", "High"] if k == 3 else ["Low", "High"]
    mapping = {c: names[i] for i, c in enumerate(order)}
    return pd.Series([mapping[l] for l in labels], index=metrics.index)
