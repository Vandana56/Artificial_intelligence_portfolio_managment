"""ML layer: XGBoost direction forecaster (leak-free evaluation) + KMeans risk tiers."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    def _make_model():
        return XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    MODEL_NAME = "XGBoost"
except ImportError:  # fallback keeps the app usable
    from sklearn.ensemble import GradientBoostingClassifier

    def _make_model():
        return GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05,
                                          random_state=42)
    MODEL_NAME = "GradientBoosting (xgboost not installed)"

HORIZON = 5        # predict direction 5 trading days ahead
TEST_FRAC = 0.20   # final hold-out share
MIN_ROWS = 100     # minimum labelled rows needed to train/evaluate


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


def build_dataset(price: pd.Series):
    """Returns (labelled rows, latest feature row).

    The last HORIZON rows have an UNKNOWN future price, so their target stays NaN
    (never silently turned into class 0) and they are dropped from training/testing.
    """
    X = build_features(price)
    fwd = price.shift(-HORIZON) / price - 1
    y = (fwd > 0).astype(float).where(fwd.notna())   # NaN where the future is unknown
    labelled = X.join(y.rename("y")).dropna()        # drops warm-up rows AND unknown targets
    latest = X.dropna().iloc[[-1]]                   # today's features -> live prediction
    return labelled, latest


def _scores(y_true, y_pred, y_prob) -> dict:
    y_true = np.asarray(y_true)
    p1 = y_true.mean()
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "baseline": float(max(p1, 1 - p1)),   # accuracy of always guessing the majority class
    }


def train_direction_model(price: pd.Series) -> dict:
    data, latest = build_dataset(price)
    if len(data) < MIN_ROWS:
        raise ValueError("Not enough history to train the model; use more years.")
    feats = [c for c in data.columns if c != "y"]

    # ---- chronological hold-out with a HORIZON-row purge gap (no label overlap) ----
    cut = int(len(data) * (1 - TEST_FRAC))
    train = data.iloc[:cut - HORIZON]
    test = data.iloc[cut:]

    model = _make_model().fit(train[feats], train["y"].astype(int))
    pred = model.predict(test[feats])
    prob = model.predict_proba(test[feats])[:, 1]
    metrics = _scores(test["y"].astype(int), pred, prob)
    cm = confusion_matrix(test["y"].astype(int), pred, labels=[0, 1])

    # ---- walk-forward CV on the training part only (out-of-sample stability) ----
    accs, f1s = [], []
    dev = train
    if len(dev) >= 5 * 20:
        for tr, va in TimeSeriesSplit(n_splits=5, gap=HORIZON).split(dev):
            m = _make_model().fit(dev.iloc[tr][feats], dev.iloc[tr]["y"].astype(int))
            p = m.predict(dev.iloc[va][feats])
            yt = dev.iloc[va]["y"].astype(int)
            accs.append(accuracy_score(yt, p))
            f1s.append(f1_score(yt, p, zero_division=0))
    cv = {"acc_mean": float(np.mean(accs)) if accs else float("nan"),
          "acc_std": float(np.std(accs)) if accs else float("nan"),
          "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
          "folds": len(accs)}

    # ---- final model on ALL labelled rows -> today's probability ----
    final = _make_model().fit(data[feats], data["y"].astype(int))
    p_up = float(final.predict_proba(latest[feats])[0, 1])
    imp = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)

    return {"prob_up": p_up, "metrics": metrics, "confusion": cm, "cv": cv, "importance": imp,
            "n_train": len(train), "n_test": len(test), "horizon": HORIZON}


def cluster_risk_tiers(metrics: pd.DataFrame) -> pd.DataFrame:
    """KMeans on volatility, drawdown and VaR. Clusters are ranked by a composite score
    (mean z-score of ALL three metrics), not volatility alone."""
    cols = ["Volatility", "Max Drawdown", "VaR 95% (1d)"]
    n = len(metrics)
    if n < 2:
        return pd.DataFrame({"Risk Score": [0.0], "Risk Tier": ["Medium"]}, index=metrics.index)
    Z = StandardScaler().fit_transform(metrics[cols].abs())
    score = Z.mean(axis=1)                              # per-asset composite risk score
    k = min(3, n)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    order = pd.Series(score).groupby(labels).mean().sort_values().index
    names = ["Low", "Medium", "High"] if k == 3 else ["Low", "High"]
    mapping = {c: names[i] for i, c in enumerate(order)}
    return pd.DataFrame({"Risk Score": score, "Risk Tier": [mapping[l] for l in labels]},
                        index=metrics.index)
