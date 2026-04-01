"""
Phase 1 — Ridge Regression (Baseline + Ablation)
Person 2 (Models & Evaluation)

Trains Ridge with 5 alpha values, picks best by val MAE, runs ablation study.

DO NOT load test_features.csv here — that is Phase 5 only.

Outputs:
  models/ridge_model.pkl
  models/scaler.pkl
  outputs/ridge_alpha_search.csv
  outputs/ridge_val_preds.npy
  outputs/ablation_results.csv
"""

import sys
import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from phase0_skeleton import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    N_FEATURES,
    N_TRAIN,
    N_VAL,
    RANDOM_STATE,
    dummy_features,
    validate_feature_schema,
    evaluate,
    metrics_table,
)

np.random.seed(RANDOM_STATE)

FEATURES_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

USE_DUMMY_DATA = False

# Feature groups — must match exact column names in FEATURE_COLUMNS
FEATURE_GROUPS = {
    "A_sentiment": [
        "mean_sentiment",
        "last_20_sentiment",
        "std_sentiment",
    ],
    "B_structure": [
        "talk_time_ratio",
        "avg_agent_words",
        "avg_customer_words",
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
        "intent_billing",
        "intent_technical",
        "intent_account",
        "intent_payment",
        "intent_network",
        "intent_delivery",
        "intent_refund",
        "intent_complaint",
        "intent_subscription",
        "intent_login",
    ],
}

ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate .1 columns, fix string-encoded categoricals."""
    # Drop duplicate columns
    mask = ~df.columns.duplicated(keep="first")
    df = df.loc[:, mask].copy()
    dot1_cols = [c for c in df.columns if c.endswith(".1")]
    if dot1_cols:
        df.drop(columns=dot1_cols, inplace=True)

    # repeat_contact: yes/no → 1/0
    if "repeat_contact" in df.columns:
        df["repeat_contact"] = (
            df["repeat_contact"]
            .astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
            .fillna(0).astype(float)
        )

    # duration_ordinal: short/medium/long → -1/0/1 (may be NaN if Person 1 skipped)
    if "duration_ordinal" in df.columns:
        df["duration_ordinal"] = (
            df["duration_ordinal"]
            .astype(str).str.strip().str.lower()
            .map({
                "short": -1, "medium": 0, "long": 1,
                "-1": -1, "0": 0, "1": 1,
                "-1.0": -1, "0.0": 0, "1.0": 1,
                "nan": 0,
            })
            .fillna(0).astype(float)
        )

    # duration_deviation: coerce to numeric
    if "duration_deviation" in df.columns:
        df["duration_deviation"] = pd.to_numeric(
            df["duration_deviation"], errors="coerce"
        ).fillna(0.0)

    # Coerce all feature columns to numeric, fill NaN with 0
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_data():
    """
    Load train/val splits. Returns (X_train, y_train, X_val, y_val, col_names, scaler).
    X_* are scaled. Scaler is fit on train only.
    """
    if USE_DUMMY_DATA:
        print("  [INFO] Using dummy data.")
        X_train_raw, y_train = dummy_features(N_TRAIN, seed=RANDOM_STATE)
        X_val_raw, y_val = dummy_features(N_VAL, seed=RANDOM_STATE + 1)
        col_names = FEATURE_COLUMNS
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        return X_train, y_train, X_val, y_val, col_names, scaler

    train_path = FEATURES_DIR / "train_features.csv"
    val_path = FEATURES_DIR / "val_features.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path}. Has Person 1 delivered their CSVs?"
        )

    train_df = _clean_df(pd.read_csv(train_path))
    val_df = _clean_df(pd.read_csv(val_path))

    # Validate schema
    print("  Validating schemas...")
    validate_feature_schema(train_df, "train")
    validate_feature_schema(val_df, "val")

    col_names = FEATURE_COLUMNS
    X_train_raw = train_df[col_names].values.astype(float)
    y_train = train_df["csat_score"].values.astype(float)
    X_val_raw = val_df[col_names].values.astype(float)
    y_val = val_df["csat_score"].values.astype(float)

    # Scaler: fit on train ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # Save scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler → {scaler_path}")

    return X_train, y_train, X_val, y_val, col_names, scaler


def run_alpha_search(X_train, y_train, X_val, y_val):
    """Try each alpha, return (best_alpha, results_dict)."""
    print("\n── Alpha search ──────────────────────────────────────────")
    print(f"  Candidates: {ALPHA_CANDIDATES}")

    results = {}
    for alpha in ALPHA_CANDIDATES:
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_val), 1.0, 5.0)
        m = evaluate(y_val, preds)
        results[alpha] = m
        print(f"  alpha={alpha:<8}  MAE={m['mae']:.4f}  r={m['pearson_r']:.4f}")

    best_alpha = min(results, key=lambda a: results[a]["mae"])
    print(f"\n  Best alpha = {best_alpha}  (val MAE = {results[best_alpha]['mae']:.4f})")

    search_df = pd.DataFrame([{"alpha": a, **m} for a, m in results.items()])
    search_path = OUTPUTS_DIR / "ridge_alpha_search.csv"
    search_df.to_csv(search_path, index=False)
    print(f"  Saved alpha search → {search_path}")

    return best_alpha, results


def run_ablation(X_train, y_train, X_val, y_val, col_names, best_alpha):
    """
    5 Ridge runs: baseline + one per feature group removed.
    Returns DataFrame with exactly 5 rows and required schema columns.
    """
    print("\n── Ablation study ────────────────────────────────────────")
    print(f"  Using best alpha = {best_alpha}")

    ablation_rows = []
    runs = [("all_features", None)] + [(f"remove_{grp}", grp) for grp in FEATURE_GROUPS]
    baseline_mae = None

    for run_name, remove_group in runs:
        if remove_group is None:
            keep_cols = col_names
        else:
            remove_cols = set(FEATURE_GROUPS[remove_group])
            keep_cols = [c for c in col_names if c not in remove_cols]

        col_idx = [col_names.index(c) for c in keep_cols if c in col_names]
        X_tr = X_train[:, col_idx]
        X_v = X_val[:, col_idx]

        model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
        model.fit(X_tr, y_train)
        preds = np.clip(model.predict(X_v), 1.0, 5.0)
        m = evaluate(y_val, preds)

        if baseline_mae is None:
            baseline_mae = m["mae"]
            delta = 0.0
        else:
            delta = m["mae"] - baseline_mae

        removed_label = remove_group if remove_group else "none"
        print(
            f"  {run_name:<28}  n_feat={len(keep_cols):<3}  "
            f"MAE={m['mae']:.4f}  delta={delta:+.4f}"
        )

        ablation_rows.append({
            "run": run_name,
            "features_removed": removed_label,
            "n_features": len(keep_cols),
            "mae": round(m["mae"], 4),
            "rmse": round(m["rmse"], 4),
            "pearson_r": round(m["pearson_r"], 4),
            "f1_binary": round(m["f1_binary"], 4),
            "mae_delta": round(delta, 4),
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = OUTPUTS_DIR / "ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\n  Saved ablation results → {ablation_path}")

    return ablation_df


def run_phase1():
    print("=" * 60)
    print("PHASE 1 — RIDGE REGRESSION")
    print("=" * 60)

    print("\n── Loading data ──")
    X_train, y_train, X_val, y_val, col_names, scaler = load_data()
    print(f"  X_train: {X_train.shape}  y_train range: [{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  X_val:   {X_val.shape}    y_val range:   [{y_val.min():.1f}, {y_val.max():.1f}]")

    best_alpha, alpha_results = run_alpha_search(X_train, y_train, X_val, y_val)

    print("\n── Training final Ridge ──")
    ridge = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    ridge.fit(X_train, y_train)
    val_preds = np.clip(ridge.predict(X_val), 1.0, 5.0)

    final_metrics = evaluate(y_val, val_preds, model_name="Ridge (best alpha, val)")

    print(f"\n  Pred std: {val_preds.std():.4f}  (true std: {y_val.std():.4f})")

    # Save model
    model_path = MODELS_DIR / "ridge_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(ridge, f)
    print(f"\n  Saved Ridge model → {model_path}")

    # Ablation
    ablation_df = run_ablation(X_train, y_train, X_val, y_val, col_names, best_alpha)

    # Save val predictions for Phase 4
    preds_path = OUTPUTS_DIR / "ridge_val_preds.npy"
    np.save(preds_path, val_preds)
    print(f"  Saved val predictions → {preds_path}")

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print(f"  Best alpha:    {best_alpha}")
    print(f"  Val MAE:       {final_metrics['mae']:.4f}")
    print(f"  Val Pearson r: {final_metrics['pearson_r']:.4f}")
    print(f"  Files: models/ridge_model.pkl, models/scaler.pkl,")
    print(f"         outputs/ridge_alpha_search.csv, outputs/ridge_val_preds.npy,")
    print(f"         outputs/ablation_results.csv")
    print("=" * 60)

    return ridge, best_alpha, final_metrics, ablation_df


if __name__ == "__main__":
    run_phase1()
