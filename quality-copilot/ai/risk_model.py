"""Off-target risk model: will the wall thickness drift off target in the next few readings?

Target: at least one of the next HORIZON readings is more than K_SIGMA baseline standard
deviations from the baseline mean. The model never sees the future: every feature is
computed from readings and process settings up to the current time only.
Always compare the model with the simple rules in naive_scores().
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

HORIZON = 4          # next 4 readings = 2 hours at one reading per 30 minutes
K_SIGMA = 2.0
WINDOW = 8
BASELINE_READINGS = 480
SETTINGS = ["screw_rpm", "haul_speed_mpm", "melt_temp_c", "mfr"]
FEATURES = (["z", "roll_mean", "roll_std", "diff"]
            + [s + "_dev" for s in SETTINGS] + ["haul_chg", "shift_night"])


def build_dataset(wall, process, baseline_n=BASELINE_READINGS, horizon=HORIZON, k_sigma=K_SIGMA):
    """One line, wall thickness readings merged with process settings.

    wall: columns timestamp, value, shift. process: columns timestamp + SETTINGS.
    Returns one row per reading with FEATURES and target y (NaN where the future is unknown).
    """
    d = wall.merge(process[["timestamp"] + SETTINGS], on="timestamp", how="inner")
    d = d.sort_values("timestamp").reset_index(drop=True)
    if len(d) <= baseline_n:
        raise ValueError(f"Need more than {baseline_n} readings (got {len(d)})")
    base = d.iloc[:baseline_n]
    mu, sd = base["value"].mean(), base["value"].std(ddof=1)
    if sd == 0:
        raise ValueError("Baseline readings are all identical")
    z = (d["value"] - mu) / sd
    out = pd.DataFrame({"timestamp": d["timestamp"], "z": z})
    out["roll_mean"] = z.rolling(WINDOW, min_periods=3).mean()
    out["roll_std"] = z.rolling(WINDOW, min_periods=3).std()
    out["diff"] = z.diff()
    for s in SETTINGS:
        out[s + "_dev"] = d[s] - base[s].mean()
    out["haul_chg"] = d["haul_speed_mpm"].pct_change(4)
    out["shift_night"] = (d["shift"] == "Night").astype(int)
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    off = (z.abs() > k_sigma).astype(float)
    future = pd.concat([off.shift(-k) for k in range(1, horizon + 1)], axis=1)
    out["y"] = future.max(axis=1)
    out.loc[future.isna().any(axis=1), "y"] = np.nan    # future not fully known
    return out


def time_split(data, train_days=20, gap_hours=2):
    """Train on the first train_days, test on the rest. Never random.

    The last gap_hours before the cut are dropped from training so that their
    4-reading look-ahead cannot reach into the test period.
    """
    d = data.dropna(subset=["y"]).copy()
    ts = pd.to_datetime(d["timestamp"])
    cut = ts.min() + pd.Timedelta(days=train_days)
    train = d[ts < cut - pd.Timedelta(hours=gap_hours)]
    test = d[ts >= cut]
    return train, test


def train_model(train, seed=0):
    y = train["y"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training data needs both normal and off-target examples")
    pos = int(y.sum())
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        scale_pos_weight=(len(y) - pos) / pos, eval_metric="logloss", random_state=seed)
    model.fit(train[FEATURES], y)
    return model


def naive_scores(df):
    """Simple rules the model must beat: distance from normal, now and averaged."""
    return {"naive |z|": df["z"].abs().to_numpy(),
            "naive |rolling mean|": df["roll_mean"].abs().to_numpy()}


def score_report(y, scores, top_share=0.10):
    """AUC, average precision, and precision/recall when only the top 10 % are flagged."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    rep = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean())}
    if y.min() == y.max():
        rep.update(auc=float("nan"), average_precision=float("nan"),
                   precision_top=float("nan"), recall_top=float("nan"))
        return rep
    k = max(1, int(round(len(y) * top_share)))
    top = np.argsort(-scores)[:k]
    hits = int(y[top].sum())
    rep.update(auc=float(roc_auc_score(y, scores)),
               average_precision=float(average_precision_score(y, scores)),
               precision_top=hits / k, recall_top=hits / int(y.sum()))
    return rep


def feature_importance(model):
    """Which inputs the model leaned on (gain), largest first."""
    imp = pd.Series(model.feature_importances_, index=FEATURES)
    return imp.sort_values(ascending=False)
