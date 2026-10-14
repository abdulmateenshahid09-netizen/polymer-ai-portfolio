"""Off-target risk model check: time-based split, compared with simple rules."""
import pandas as pd

from ai.risk_model import (build_dataset, feature_importance, naive_scores, score_report,
                           time_split, train_model, FEATURES, HORIZON, K_SIGMA)
from db import read_table


def main():
    m = read_table("measurements")
    m = m[m["characteristic"] == "wall_thickness"]
    p = read_table("process_log")

    parts = []
    for line, g in m.groupby("line"):
        d = build_dataset(g[["timestamp", "value", "shift"]], p[p["line"] == line])
        d["line"] = line
        parts.append(d)
    data = pd.concat(parts, ignore_index=True)
    train, test = time_split(data)

    print(f"Target: a reading more than {K_SIGMA:g} baseline sigma from normal "
          f"within the next {HORIZON} readings (2 hours)")
    print(f"Train: first 20 days, {len(train)} readings, {int(train['y'].sum())} positive")
    print(f"Test:  last 10 days,  {len(test)} readings, {int(test['y'].sum())} positive\n")

    model = train_model(train)
    rows = []
    for name, s in [("XGBoost model", model.predict_proba(test[FEATURES])[:, 1])] + \
                   list(naive_scores(test).items()):
        r = score_report(test["y"], s)
        rows.append({"method": name, "AUC": round(r["auc"], 3),
                     "avg precision": round(r["average_precision"], 3),
                     "precision (top 10%)": round(r["precision_top"], 3),
                     "recall (top 10%)": round(r["recall_top"], 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nBase rate (share of readings followed by an off-target reading): "
          f"{test['y'].mean():.3f}")
    print("AUC 0.5 = coin flip, 1.0 = perfect. Top 10% = only the riskiest tenth is flagged.")
    print("\nWhat the model leaned on:")
    print(feature_importance(model).round(3).head(5).to_string())


if __name__ == "__main__":
    main()
