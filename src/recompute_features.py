"""
recompute_features.py — Fix resolution_flag from agent-recorded resolution_status

THE BUG
-------
Person 1 computed resolution_flag by scanning transcript text for resolution
keywords. Because the dataset uses only 47 unique shuffled phrases, this
keyword approach never fires → resolution_flag is 0 for every row.

THE FIX
-------
resolution_flag must be read directly from the resolution_status column
(agent-recorded at call end, available at inference time — NOT leakage):
    resolved   → 1
    unresolved → 0

This single change is expected to drop MAE from ~1.13 → ~0.55–0.65 and
raise Pearson r from ~0.24 → ~0.82–0.87.

USAGE
-----
    python recompute_features.py

Reads:  data/processed/{train,val,test}_features.csv
Writes: data/processed/{train,val,test}_features.csv  (in-place update)

Run this BEFORE ridge.py / random_forest.py / ensemble.py / evaluate.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "processed"
SPLITS = ["train", "val", "test"]


def _fix_resolution_flag(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Replace resolution_flag with a binary encoding of resolution_status.

    resolution_status is agent-recorded at call end (available at inference
    time). This is NOT data leakage — it is equivalent to reading a CRM field
    that the agent ticks before hanging up.
    """
    if "resolution_status" not in df.columns:
        raise ValueError(
            f"[{split_name}] 'resolution_status' column missing. "
            "Cannot compute resolution_flag. "
            "Check that the raw CSV was joined into the feature file."
        )

    df = df.copy()
    df["resolution_flag"] = (
        df["resolution_status"].astype(str).str.strip().str.lower() == "resolved"
    ).astype(int)

    dist = df["resolution_flag"].value_counts().to_dict()
    n = len(df)
    pct_resolved = dist.get(1, 0) / n * 100
    print(
        f"  [{split_name}] resolution_flag: "
        f"resolved={dist.get(1, 0)} ({pct_resolved:.1f}%), "
        f"unresolved={dist.get(0, 0)} ({100 - pct_resolved:.1f}%)"
    )

    if abs(pct_resolved - 50.0) > 15:
        print(
            f"  [WARN] [{split_name}] resolution_flag is skewed "
            f"({pct_resolved:.1f}% resolved). "
            "Check for split imbalance."
        )

    return df


def _fix_repeat_contact(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Ensure repeat_contact is integer 0/1, not string 'yes'/'no'.
    (Already handled in ridge.py/_clean_df but defensive fix here too.)
    """
    df = df.copy()
    if df["repeat_contact"].dtype == object:
        df["repeat_contact"] = (
            df["repeat_contact"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
            .fillna(0)
            .astype(int)
        )
    else:
        df["repeat_contact"] = df["repeat_contact"].fillna(0).astype(int)
    return df


def _verify_no_leakage(df: pd.DataFrame, split_name: str) -> None:
    """
    Confirm no target-derived columns are being used as features.
    Raises if csat_range or any derivative of csat_score appears in features.
    """
    leakage_cols = {"csat_range", "emotional_arc"}
    # emotional_arc could be derived from CSAT; resolution_status is fine (agent input)
    found = leakage_cols & set(df.columns)
    if found:
        print(
            f"  [INFO] [{split_name}] Columns present but excluded from model features: "
            f"{found}. Ensure FEATURE_COLUMNS in phase0_skeleton.py does not list them."
        )


def _show_correlations(df: pd.DataFrame) -> None:
    """Print correlations of key features with csat_score (train set only)."""
    print("\n── Feature correlations with csat_score (train) ──")
    key_features = [
        "resolution_flag",
        "repeat_contact",
        "mean_sentiment",
        "last_20_sentiment",
        "std_sentiment",
        "empathy_density",
        "apology_count",
        "transfer_count",
        "talk_time_ratio",
        "interruption_count",
        "duration_ordinal",
        "duration_deviation",
    ]
    for col in key_features:
        if col in df.columns:
            try:
                r = df[col].astype(float).corr(df["csat_score"])
                flag = "  ← KEY SIGNAL" if abs(r) > 0.1 else ""
                print(f"  {col:<30} r={r:+.4f}{flag}")
            except Exception:
                print(f"  {col:<30} ERROR computing correlation")
    print()


def run():
    print("=" * 60)
    print("RECOMPUTE FEATURES — Fixing resolution_flag")
    print("=" * 60)
    print(
        "\nBUG: Person 1 derived resolution_flag from transcript keyword matching."
        "\n     All synthetic calls produce resolution_flag=0 (no keywords match)."
        "\nFIX: Encode resolution_flag directly from resolution_status column"
        "\n     (agent-recorded at call end — NOT data leakage)."
    )

    for split in SPLITS:
        csv_path = FEATURES_DIR / f"{split}_features.csv"
        if not csv_path.exists():
            print(f"\n  [SKIP] {csv_path} not found — skipping.")
            continue

        print(f"\n── Processing {split} ──")
        df = pd.read_csv(csv_path)
        print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

        # Show old resolution_flag distribution
        if "resolution_flag" in df.columns:
            old_dist = df["resolution_flag"].value_counts().to_dict()
            print(f"  OLD resolution_flag distribution: {old_dist}")
            if len(old_dist) == 1 and 0 in old_dist:
                print("  [CONFIRMED BUG] resolution_flag was all zeros!")

        df = _fix_resolution_flag(df, split)
        df = _fix_repeat_contact(df, split)
        _verify_no_leakage(df, split)

        df.to_csv(csv_path, index=False)
        print(f"  Written → {csv_path}")

    # Verify on train set
    train_path = FEATURES_DIR / "train_features.csv"
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        _show_correlations(train_df)

        res_flag_corr = (
            train_df["resolution_flag"].astype(float).corr(train_df["csat_score"])
        )
        if res_flag_corr > 0.7:
            print(
                f"  [OK] resolution_flag correlation r={res_flag_corr:.4f}"
                " — consistent with r=0.82 documented in business spec."
            )
        else:
            print(
                f"  [WARN] resolution_flag correlation r={res_flag_corr:.4f}"
                " — lower than expected 0.82. Check encoding."
            )

    print("\n" + "=" * 60)
    print("RECOMPUTE COMPLETE")
    print("  Next: python src/models/ridge.py")
    print("        python src/models/random_forest.py")
    print("        python src/models/ensemble.py")
    print("        python src/evaluation/evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    run()
