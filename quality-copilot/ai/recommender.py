"""Settings recommender: which haul-off speed brings the wall thickness back to target?

Physics first: a mass balance links extruder output (kg/h), haul-off speed and wall thickness.
From each reading we back-calculate the output the extruder really delivered. The data then
teaches only what the physics cannot know: how output per screw rpm changes with melt
temperature and resin MFR. A slowly drifting 'efficiency' (hopper, screw wear, and so on)
is tracked from the latest readings. Optuna then searches for the setting that hits the aim.

This is learned from SIMULATED data. Never apply a recommendation to a real line without
a trial under supervision.
"""
import numpy as np
import optuna
import pandas as pd

OD_M = 0.110            # outside diameter, m (PE100 SDR11 110 mm)
DENSITY = 950.0         # melt-to-solid density used by the mass balance, kg/m3
ALPHA = 0.5             # weight of the newest reading in the efficiency tracker
COLUMNS = ["timestamp", "value", "screw_rpm", "haul_speed_mpm", "melt_temp_c", "mfr"]


def output_kgh(thickness_mm, haul_mpm, od_m=OD_M, density=DENSITY):
    """Extruder output (kg/h) needed to make this wall thickness at this haul-off speed."""
    t = np.asarray(thickness_mm, dtype=float) / 1000.0
    v = np.asarray(haul_mpm, dtype=float) / 60.0
    return 3600.0 * density * np.pi * t * (od_m - t) * v


def thickness_mm(output, haul_mpm, od_m=OD_M, density=DENSITY):
    """Wall thickness (mm) produced by this output at this haul-off speed."""
    m = np.asarray(output, dtype=float) / 3600.0
    v = np.asarray(haul_mpm, dtype=float) / 60.0
    disc = od_m ** 2 - 4.0 * m / (np.pi * density * v)
    return (od_m - np.sqrt(np.maximum(disc, 0.0))) / 2.0 * 1000.0


def prepare(wall, process):
    """Merge wall thickness readings with process settings (one line, time order)."""
    d = wall.merge(process, on="timestamp", how="inner").sort_values("timestamp")
    return d[COLUMNS].reset_index(drop=True)


def _log_output_per_rpm(d):
    return np.log(output_kgh(d["value"], d["haul_speed_mpm"]) / d["screw_rpm"])


def fit_process_model(d):
    """Learn how output per screw rpm depends on melt temperature and MFR (log-linear)."""
    if len(d) < 30:
        raise ValueError("Need at least 30 readings to fit the process model")
    melt_ref, mfr_ref = float(d["melt_temp_c"].mean()), float(d["mfr"].mean())
    X = np.column_stack([np.ones(len(d)), d["melt_temp_c"] - melt_ref, d["mfr"] - mfr_ref])
    coef, *_ = np.linalg.lstsq(X, _log_output_per_rpm(d).to_numpy(), rcond=None)
    return {"intercept": float(coef[0]), "melt_coef": float(coef[1]), "mfr_coef": float(coef[2]),
            "melt_ref": melt_ref, "mfr_ref": mfr_ref}


def _base_log_eff(model, melt, mfr):
    return (model["intercept"] + model["melt_coef"] * (np.asarray(melt, dtype=float) - model["melt_ref"])
            + model["mfr_coef"] * (np.asarray(mfr, dtype=float) - model["mfr_ref"]))


def residuals(model, d):
    """What the melt/MFR model cannot explain (the hidden, drifting efficiency)."""
    return _log_output_per_rpm(d) - _base_log_eff(model, d["melt_temp_c"], d["mfr"])


def ewma_state(resid, alpha=ALPHA):
    """Current efficiency estimate: exponentially weighted average, newest reading counts most."""
    r = pd.Series(np.asarray(resid, dtype=float))
    if r.empty:
        return 0.0
    return float(r.ewm(alpha=alpha, adjust=False).mean().iloc[-1])


def predict_thickness(model, state, rpm, haul, melt, mfr):
    """Predicted wall thickness (mm) for these settings and this efficiency state."""
    out = np.asarray(rpm, dtype=float) * np.exp(_base_log_eff(model, melt, mfr) + state)
    return thickness_mm(out, haul)


def backtest(model, d, alpha=ALPHA):
    """One-step-ahead predictions: each reading is predicted from the readings before it."""
    e = residuals(model, d)
    state = e.ewm(alpha=alpha, adjust=False).mean().shift(1).fillna(0.0)
    out = pd.DataFrame({"timestamp": d["timestamp"], "actual": d["value"]})
    out["model"] = predict_thickness(model, state, d["screw_rpm"], d["haul_speed_mpm"],
                                     d["melt_temp_c"], d["mfr"])
    out["persistence"] = d["value"].shift(1)
    out["haul_changed"] = np.log(d["haul_speed_mpm"]).diff().abs() > 0.005
    return out


def fit_and_validate(frames, train_days=20, alpha=ALPHA):
    """Fit on the first train_days of all lines, test on the rest (never random).

    frames: dict line -> prepared data. Returns the model and an honest test report
    comparing it with the simplest alternative: 'the next reading equals the last one'.
    """
    start = min(pd.to_datetime(f["timestamp"]).min() for f in frames.values())
    cut = start + pd.Timedelta(days=train_days)
    train = pd.concat([f[pd.to_datetime(f["timestamp"]) < cut] for f in frames.values()])
    model = fit_process_model(train)
    parts = []
    for f in frames.values():
        b = backtest(model, f, alpha)
        parts.append(b[pd.to_datetime(b["timestamp"]) >= cut].dropna())
    test = pd.concat(parts)
    err_m = (test["actual"] - test["model"]).abs()
    err_p = (test["actual"] - test["persistence"]).abs()
    ch = test["haul_changed"]
    report = {
        "n_test": int(len(test)), "n_haul_changes": int(ch.sum()),
        "mae_model": float(err_m.mean()), "mae_persistence": float(err_p.mean()),
        "mae_model_haul_change": float(err_m[ch].mean()) if ch.any() else float("nan"),
        "mae_persistence_haul_change": float(err_p[ch].mean()) if ch.any() else float("nan"),
        "sigma_mm": float((test["actual"] - test["model"]).std(ddof=1)),
    }
    return model, report


def margin_sigmas(pred, lsl, usl, sigma):
    """How many prediction sigmas separate the predicted thickness from the nearest spec limit."""
    return float(min(pred - lsl, usl - pred) / sigma)


def recommend_settings(model, state, melt, mfr, rpm, haul, target, lsl, usl, sigma_mm,
                       max_change_pct=5.0, allow_rpm=False, n_trials=300, seed=0):
    """Search (with Optuna) for the smallest setting change that brings thickness to target.

    By default only the haul-off speed changes, as operators usually do. Allowing the screw
    rpm to change costs extra, so it is used only when it clearly helps.
    """
    c = max_change_pct / 100.0
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        r = trial.suggest_float("rpm", rpm * (1 - c), rpm * (1 + c)) if allow_rpm else rpm
        h = trial.suggest_float("haul", haul * (1 - c), haul * (1 + c))
        t = float(predict_thickness(model, state, r, h, melt, mfr))
        miss = (t - target) / 0.05                      # 0.05 mm counts as one unit of miss
        change = 0.05 * ((h / haul - 1) * 100) ** 2 + 0.25 * ((r / rpm - 1) * 100) ** 2
        return miss ** 2 + change

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    new_haul = float(study.best_params["haul"])
    new_rpm = float(study.best_params["rpm"]) if allow_rpm else float(rpm)
    now = float(predict_thickness(model, state, rpm, haul, melt, mfr))
    new = float(predict_thickness(model, state, new_rpm, new_haul, melt, mfr))
    return {
        "rpm": new_rpm, "haul": new_haul,
        "haul_change_pct": (new_haul / haul - 1) * 100, "rpm_change_pct": (new_rpm / rpm - 1) * 100,
        "predicted_now": now, "predicted_new": new,
        "margin_sigmas": margin_sigmas(new, lsl, usl, sigma_mm),
        "reached_target": abs(new - target) <= 0.03,
        "at_limit": abs(new_haul / haul - 1) >= c * 0.999,
    }
