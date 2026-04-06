"""
Run this ONCE before starting the API or dashboard.
It reads train.csv and computes all aggregate stats the dashboard needs.

Usage:
    python scripts/generate_aggregate_stats.py
"""

import pandas as pd
import json
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")

    train = pd.read_csv(train_path)
    print(f"Loaded train.csv — {len(train)} rows, columns: {list(train.columns)}")

    stats = {}

    # ── Normalise repeat_contact to numeric 0/1 regardless of stored dtype ──
    if "repeat_contact" in train.columns:
        rc = train["repeat_contact"]
        if rc.dtype == object or str(rc.dtype) == "string":
            # Handle "yes"/"no", "True"/"False", "1"/"0", "1.0"/"0.0"
            train["repeat_contact"] = (
                rc.astype(str).str.strip().str.lower()
                  .map({"1": 1, "1.0": 1, "yes": 1, "true": 1,
                        "0": 0, "0.0": 0, "no": 0,  "false": 0})
                  .fillna(0).astype(int)
            )
        else:
            train["repeat_contact"] = pd.to_numeric(rc, errors="coerce").fillna(0).astype(int)

    # ── Derive a duration bucket from call_duration_seconds if needed ────────
    if "call_duration" not in train.columns and "call_duration_seconds" in train.columns:
        secs = train["call_duration_seconds"]
        train["call_duration"] = pd.cut(
            secs,
            bins=[0, 240, 390, float("inf")],
            labels=["short", "medium", "long"],
        ).astype(str)

    # 1. Average CSAT per issue type
    if "issue_type" in train.columns and "csat_score" in train.columns:
        stats["avg_csat_by_issue"] = (
            train.groupby("issue_type")["csat_score"].mean().round(3).to_dict()
        )

    # 2. Average CSAT per emotional arc
    if "emotional_arc" in train.columns and "csat_score" in train.columns:
        stats["avg_csat_by_arc"] = (
            train.groupby("emotional_arc")["csat_score"].mean().round(3).to_dict()
        )

    # 3. Repeat contact rate per issue type
    if "repeat_contact" in train.columns and "issue_type" in train.columns:
        stats["repeat_contact_rate_by_issue"] = (
            train.groupby("issue_type")["repeat_contact"].mean().round(3).to_dict()
        )

    # 4. Emotional arc distribution (% of calls)
    if "emotional_arc" in train.columns:
        counts = train["emotional_arc"].value_counts()
        stats["arc_distribution"] = (counts / len(train) * 100).round(1).to_dict()

    # 5. CSAT distribution buckets
    if "csat_score" in train.columns:
        s = train["csat_score"]
        stats["csat_distribution"] = {
            "low_1_to_2.5":      round((s < 2.5).mean() * 100, 1),
            "medium_2.5_to_3.5": round(((s >= 2.5) & (s < 3.5)).mean() * 100, 1),
            "high_3.5_to_5":     round((s >= 3.5).mean() * 100, 1),
        }
        stats["overall_avg_csat"] = round(float(s.mean()), 3)

    stats["total_calls_in_training"] = len(train)

    # 6. Call duration distribution
    if "call_duration" in train.columns:
        stats["duration_distribution"] = (
            train["call_duration"].value_counts(normalize=True).mul(100).round(1).to_dict()
        )

    # Save
    out_path = os.path.join(BASE_DIR, "outputs", "metrics", "aggregate_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅  Saved to {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()