"""
Phase 1 — Ridge Regression (Baseline + Ablation)
Person 2 (Models & Evaluation)

WHAT THIS FILE DOES
-------------------
1. Loads train/val feature CSVs from Person 1 (falls back to dummy data if
   they haven't arrived yet — so you can run and test this right now).

2. Trains Ridge regression with 5 alpha values and picks the best by
   validation MAE. Saves ridge_model.pkl.

3. Runs the ablation study: 5 passes of Ridge — once with all features,
   then once with each feature group (A/B/C/D) removed. The drop in
   validation MAE when a group is removed tells you how much that group
   contributes. This is a key table in your final report.

4. Saves all results to outputs/ so Person 4 can use them in the dashboard.

WHEN TO SWITCH FROM DUMMY → REAL DATA
--------------------------------------
Change USE_DUMMY_DATA = False once Person 1 delivers:
  data/features/train_features.csv
  data/features/val_features.csv
  data/features/test_features.csv  ← do NOT load this here. Phase 5 only.

RUN ORDER
---------
  python phase0_skeleton.py   ← already done
  python phase1_ridge.py      ← this file
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from phase0_skeleton import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    N_FEATURES,
    N_TRAIN, N_VAL,
    RANDOM_STATE,
    dummy_features,
    validate_feature_schema,
    evaluate,
    metrics_table,
)

np.random.seed(RANDOM_STATE)

# ── Paths ────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR   = ROOT / "models"
OUTPUTS_DIR  = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Toggle this once Person 1 delivers their CSVs ────────────
USE_DUMMY_DATA = True


# ─────────────────────────────────────────────────────────────
# FEATURE GROUP DEFINITIONS
# Used for ablation: each run removes one entire group.
# Update these lists the moment Person 1 confirms exact column names.
# ─────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "A_sentiment": [
        "mean_sentiment",
        "final_sentiment",
        "sentiment_std",
    ],
    "B_structure": [
        "talk_time_ratio",
        "avg_words_agent",
        "avg_words_customer",
        "interruption_count",
        "resolution_flag",
    ],
    "C_agent": [
        "empathy_density",
        "apology_count",
        "transfer_count",
    ],
    "D_metadata": [
        "duration_ordinal",
        "duration_deviation",
        "repeat_contact",
        "issue_type_billing",
        "issue_type_technical",
        "issue_type_account",
        "issue_type_payment",
        "issue_type_network",
        "issue_type_delivery",
        "issue_type_returns",
        "issue_type_complaint",
        "issue_type_upgrade",
        "issue_type_cancellation",
    ],
}

# Alpha values to search — log-spaced from 0.01 to 100
ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray]:
    """
    Load train and val splits.

    Returns (X_train, y_train, X_val, y_val, col_names, scaler)
      - X_*     : float arrays, already scaled
      - y_*     : float arrays, raw CSAT 1–5
      - col_names: list of feature column names in the order they appear in X
      - scaler  : fitted StandardScaler (needed by Phase 6 predict function)

    NOTE: test split is NOT loaded here. Only Phase 5 touches it.
    """
    if USE_DUMMY_DATA:
        print("  [INFO] Using dummy data. Set USE_DUMMY_DATA=False once")
        print("         Person 1 delivers data/features/train_features.csv")
        print()

        X_train_raw, y_train = dummy_features(N_TRAIN, seed=RANDOM_STATE)
        X_val_raw,   y_val   = dummy_features(N_VAL,   seed=RANDOM_STATE + 1)
        col_names = FEATURE_COLUMNS

        # Fit scaler on train only, apply to val
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val   = scaler.transform(X_val_raw)

    else:
        # ── Real data path ───────────────────────────────────
        train_path = FEATURES_DIR / "train_features.csv"
        val_path   = FEATURES_DIR / "val_features.csv"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Expected {train_path}. Has Person 1 delivered their CSVs?"
            )

        train_df = pd.read_csv(train_path)
        val_df   = pd.read_csv(val_path)

        # Validate schema (will raise if something is wrong)
        print("  Validating schemas...")
        validate_feature_schema(train_df, "train")
        validate_feature_schema(val_df,   "val")

        col_names = FEATURE_COLUMNS   # use agreed order

        X_train_raw = train_df[col_names].values.astype(float)
        y_train     = train_df["csat_score"].values.astype(float)

        X_val_raw   = val_df[col_names].values.astype(float)
        y_val       = val_df["csat_score"].values.astype(float)

        # Scaler: fit on train ONLY
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val   = scaler.transform(X_val_raw)

        # Save scaler — Person 3 needs this for inference
        scaler_path = MODELS_DIR / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Saved scaler → {scaler_path}")

    return X_train, y_train, X_val, y_val, col_names, scaler


# ─────────────────────────────────────────────────────────────
# ALPHA SEARCH
# Try 5 alpha values, evaluate each on validation MAE,
# pick best, lock it. Never look at test here.
# ─────────────────────────────────────────────────────────────

def run_alpha_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple[float, dict]:
    """
    Train Ridge with each alpha in ALPHA_CANDIDATES, evaluate on val set.

    Returns:
        best_alpha : float — the alpha with lowest val MAE
        results    : dict  — {alpha: metrics_dict} for all candidates
                             (saved to outputs/ridge_alpha_search.csv)
    """
    print("\n── Alpha search ──────────────────────────────────────────")
    print(f"  Candidates: {ALPHA_CANDIDATES}")
    print()

    results = {}
    for alpha in ALPHA_CANDIDATES:
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_val), 1.0, 5.0)
        m = evaluate(y_val, preds)
        results[alpha] = m
        print(f"  alpha={alpha:<8}  MAE={m['mae']:.4f}  r={m['pearson_r']:.4f}")

    # Pick best by MAE (lower is better)
    best_alpha = min(results, key=lambda a: results[a]["mae"])
    print(f"\n  Best alpha = {best_alpha}  (val MAE = {results[best_alpha]['mae']:.4f})")

    # Save search results
    search_df = pd.DataFrame([
        {"alpha": a, **m} for a, m in results.items()
    ])
    search_path = OUTPUTS_DIR / "ridge_alpha_search.csv"
    search_df.to_csv(search_path, index=False)
    print(f"  Saved alpha search → {search_path}")

    return best_alpha, results


# ─────────────────────────────────────────────────────────────
# ABLATION STUDY
# 5 Ridge runs on validation data:
#   Run 0 : all features (baseline)
#   Run 1 : remove Group A (sentiment)
#   Run 2 : remove Group B (structure)
#   Run 3 : remove Group C (agent behaviour)
#   Run 4 : remove Group D (metadata)
#
# The delta in MAE (run_i minus baseline) is the group's contribution.
# A large delta = that group matters a lot.
# ─────────────────────────────────────────────────────────────

def run_ablation(
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    X_val:     np.ndarray,
    y_val:     np.ndarray,
    col_names: list[str],
    best_alpha: float,
) -> pd.DataFrame:
    """
    Ablation: train Ridge 5 times, each with one feature group removed.

    Args:
        X_train, y_train : training data (scaled)
        X_val, y_val     : validation data (scaled)
        col_names        : list of column names matching X columns
        best_alpha       : alpha locked from alpha search

    Returns:
        ablation_df : DataFrame with columns
                      [run, features_removed, n_features, mae, rmse, pearson_r, f1_binary, mae_delta]
    """
    print("\n── Ablation study ────────────────────────────────────────")
    print(f"  Using best alpha = {best_alpha}")

    ablation_rows = []

    # All runs: first is baseline (all features), then one group removed each
    runs = [("all_features", None)] + [
        (f"remove_{grp}", grp) for grp in FEATURE_GROUPS
    ]

    baseline_mae = None

    for run_name, remove_group in runs:
        # Determine which columns to keep
        if remove_group is None:
            keep_cols = col_names
        else:
            remove_cols = set(FEATURE_GROUPS[remove_group])
            keep_cols = [c for c in col_names if c not in remove_cols]

        # Get column indices
        col_idx = [col_names.index(c) for c in keep_cols if c in col_names]
        X_tr = X_train[:, col_idx]
        X_v  = X_val[:,   col_idx]

        model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
        model.fit(X_tr, y_train)
        preds = np.clip(model.predict(X_v), 1.0, 5.0)
        m = evaluate(y_val, preds)

        if baseline_mae is None:
            baseline_mae = m["mae"]
            delta = 0.0
        else:
            delta = m["mae"] - baseline_mae   # positive = worse = group was useful

        removed_label = remove_group if remove_group else "none"
        print(f"  {run_name:<28}  n_feat={len(keep_cols):<3}  "
              f"MAE={m['mae']:.4f}  delta={delta:+.4f}")

        ablation_rows.append({
            "run":             run_name,
            "features_removed": removed_label,
            "n_features":      len(keep_cols),
            "mae":             round(m["mae"],       4),
            "rmse":            round(m["rmse"],      4),
            "pearson_r":       round(m["pearson_r"], 4),
            "f1_binary":       round(m["f1_binary"], 4),
            "mae_delta":       round(delta,          4),
        })

    ablation_df = pd.DataFrame(ablation_rows)

    # Save
    ablation_path = OUTPUTS_DIR / "ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\n  Saved ablation results → {ablation_path}")

    # Print summary table
    print("\n  Ablation summary (sorted by contribution):")
    summary = ablation_df[ablation_df["features_removed"] != "none"].sort_values(
        "mae_delta", ascending=False
    )[["features_removed", "n_features", "mae", "mae_delta"]]
    print(summary.to_string(index=False))

    return ablation_df


# ─────────────────────────────────────────────────────────────
# MAIN PHASE 1 RUNNER
# ─────────────────────────────────────────────────────────────

def run_phase1():
    print("=" * 60)
    print("PHASE 1 — RIDGE REGRESSION")
    print("=" * 60)

    # 1. Load data
    print("\n── Loading data ──")
    X_train, y_train, X_val, y_val, col_names, scaler = load_data()
    print(f"  X_train: {X_train.shape}  y_train range: [{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  X_val:   {X_val.shape}    y_val range:   [{y_val.min():.1f}, {y_val.max():.1f}]")

    # 2. Alpha search
    best_alpha, alpha_results = run_alpha_search(X_train, y_train, X_val, y_val)

    # 3. Train final Ridge with best alpha
    print("\n── Training final Ridge ──")
    ridge = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    ridge.fit(X_train, y_train)
    val_preds = np.clip(ridge.predict(X_val), 1.0, 5.0)

    print(f"\n  Val prediction range: [{val_preds.min():.2f}, {val_preds.max():.2f}]")
    print(f"  True y range:         [{y_val.min():.2f}, {y_val.max():.2f}]")

    final_metrics = evaluate(y_val, val_preds, model_name="Ridge (best alpha, val)")

    # 4. Sanity checks
    print("\n── Sanity checks ──")
    pct_outside_range = np.mean((val_preds < 1.0) | (val_preds > 5.0)) * 100
    print(f"  Predictions clipped to [1,5]: {pct_outside_range:.1f}% were outside before clipping")
    print(f"  Pred std: {val_preds.std():.4f}  (true std: {y_val.std():.4f})")

    # Coefficient inspection — which features did Ridge weight most?
    coef_series = pd.Series(np.abs(ridge.coef_), index=col_names).sort_values(ascending=False)
    print("\n  Top 5 Ridge coefficients (absolute value):")
    for feat, coef in coef_series.head(5).items():
        print(f"    {feat:<35} {coef:.4f}")

    # 5. Save model
    model_path = MODELS_DIR / "ridge_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(ridge, f)
    print(f"\n  Saved Ridge model → {model_path}")

    # 6. Ablation
    ablation_df = run_ablation(X_train, y_train, X_val, y_val, col_names, best_alpha)

    # 7. Save val predictions for ensemble later (Phase 4 needs them)
    preds_path = OUTPUTS_DIR / "ridge_val_preds.npy"
    np.save(preds_path, val_preds)
    print(f"  Saved val predictions → {preds_path}  (Phase 4 needs this)")

    # 8. Summary
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print(f"  Best alpha:    {best_alpha}")
    print(f"  Val MAE:       {final_metrics['mae']:.4f}")
    print(f"  Val Pearson r: {final_metrics['pearson_r']:.4f}")
    print(f"  Val F1:        {final_metrics['f1_binary']:.4f}")
    print()
    print("  Files written:")
    print(f"    models/ridge_model.pkl")
    print(f"    outputs/ridge_alpha_search.csv")
    print(f"    outputs/ablation_results.csv")
    print(f"    outputs/ridge_val_preds.npy")
    print()
    print("  Next: run phase2_random_forest.py")
    print("=" * 60)

    return ridge, best_alpha, final_metrics, ablation_df


if __name__ == "__main__":
    run_phase1()
