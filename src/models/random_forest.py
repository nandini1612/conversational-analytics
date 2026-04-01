"""
Phase 2 — Random Forest Regressor
Person 2 (Models & Evaluation)

Trains RF with 9 hyperparameter combos, picks best by val MAE.
RF does NOT use scaled inputs — tree models are scale-invariant.

Outputs:
  models/rf_model.pkl
  models_saved/feature_importances.json   ← Person 3 (SHAP) + Person 4 (dashboard)
  models_saved/rf_search.csv
  models_saved/rf_val_preds.npy

DO NOT load test_features.csv here — that is Phase 5 only.
"""

import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from phase0_skeleton import (
    FEATURE_COLUMNS,
    N_TRAIN,
    N_VAL,
    RANDOM_STATE,
    dummy_features,
    validate_feature_schema,
    evaluate,
)

np.random.seed(RANDOM_STATE)

FEATURES_DIR  = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
OUTPUTS_DIR   = ROOT / "outputs" / "predictions"
METRICS_DIR   = ROOT / "outputs" / "metrics"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

USE_DUMMY_DATA = False

# Full 9-combo grid: n_estimators × max_depth × min_samples_leaf
PARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
    for n in [100, 200, 500]
    for d in [None, 10, 20]
    for l in [1, 5, 10]
]
# That's 3×3×3 = 27 combos; we use all 27 (≥9 required)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate .1 columns, fix string-encoded categoricals."""
    mask = ~df.columns.duplicated(keep="first")
    df = df.loc[:, mask].copy()
    dot1_cols = [c for c in df.columns if c.endswith(".1")]
    if dot1_cols:
        df.drop(columns=dot1_cols, inplace=True)

    if "repeat_contact" in df.columns:
        df["repeat_contact"] = (
            df["repeat_contact"]
            .astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
            .fillna(0).astype(float)
        )

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

    if "duration_deviation" in df.columns:
        df["duration_deviation"] = pd.to_numeric(
            df["duration_deviation"], errors="coerce"
        ).fillna(0.0)

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_data():
    """Load train/val splits. RF uses raw (unscaled) features."""
    if USE_DUMMY_DATA:
        print("  [INFO] Using dummy data.")
        X_train_raw, y_train = dummy_features(N_TRAIN, seed=RANDOM_STATE)
        X_val_raw, y_val = dummy_features(N_VAL, seed=RANDOM_STATE + 1)
        return X_train_raw, y_train, X_val_raw, y_val, FEATURE_COLUMNS

    train_path = FEATURES_DIR / "train_features.csv"
    val_path = FEATURES_DIR / "val_features.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path}. Has Person 1 delivered their CSVs?"
        )

    train_df = _clean_df(pd.read_csv(train_path))
    val_df = _clean_df(pd.read_csv(val_path))

    validate_feature_schema(train_df, "train")
    validate_feature_schema(val_df, "val")

    col_names = FEATURE_COLUMNS
    X_train = train_df[col_names].values.astype(float)
    y_train = train_df["csat_score"].values.astype(float)
    X_val = val_df[col_names].values.astype(float)
    y_val = val_df["csat_score"].values.astype(float)

    return X_train, y_train, X_val, y_val, col_names


def run_phase2():
    print("=" * 60)
    print("PHASE 2 — RANDOM FOREST")
    print("=" * 60)

    X_train, y_train, X_val, y_val, col_names = load_data()
    print(f"\n  X_train: {X_train.shape}  X_val: {X_val.shape}")
    print(f"  Grid size: {len(PARAM_GRID)} combinations")

    # Grid search
    print("\n── Hyperparameter search ──")
    best_mae, best_params, best_model = float("inf"), None, None
    search_rows = []

    for params in PARAM_GRID:
        rf = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = np.clip(rf.predict(X_val), 1.0, 5.0)
        m = evaluate(y_val, preds)
        label = f"n={params['n_estimators']} d={str(params['max_depth']):<4} l={params['min_samples_leaf']}"
        print(f"  {label:<35}  MAE={m['mae']:.4f}  r={m['pearson_r']:.4f}")
        search_rows.append({"params": str(params), **m})
        if m["mae"] < best_mae:
            best_mae, best_params, best_model = m["mae"], params, rf

    print(f"\n  Best params: {best_params}  (val MAE={best_mae:.4f})")

    # Mean-compression warning
    best_preds = np.clip(best_model.predict(X_val), 1.0, 5.0)
    print(f"\n  RF pred range: [{best_preds.min():.2f}, {best_preds.max():.2f}]")
    print(f"  RF pred std:   {best_preds.std():.4f}  (true std: {y_val.std():.4f})")
    if best_preds.std() < y_val.std() * 0.5:
        print("  [WARN] RF compressing predictions toward mean — document in report.")

    final_m = evaluate(y_val, best_preds, model_name="Random Forest (best, val)")

    # Feature importances — sorted descending
    importances = dict(zip(col_names, best_model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 feature importances:")
    for feat, imp in sorted_imp[:10]:
        print(f"    {feat:<35} {imp:.4f}")

    # Save model to models/rf_model.pkl
    model_path = MODELS_DIR / "rf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n  Saved → {model_path}")

    # Save feature importances JSON
    imp_path = METRICS_DIR / "feature_importances.json"
    with open(imp_path, "w") as f:
        json.dump({"importances": importances, "sorted": sorted_imp}, f, indent=2)
    print(f"  Saved → {imp_path}  ← hand off to Person 3 + 4")

    # Save search log
    search_path = METRICS_DIR / "rf_search.csv"
    pd.DataFrame(search_rows).to_csv(search_path, index=False)
    print(f"  Saved → {search_path}")

    # Save val predictions for Phase 4 ensemble
    preds_path = OUTPUTS_DIR / "rf_val_preds.npy"
    np.save(preds_path, best_preds)
    print(f"  Saved val preds → {preds_path}")

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print(f"  Best params: {best_params}")
    print(f"  Val MAE:     {final_m['mae']:.4f}")
    print("  Files: models/rf_model.pkl, outputs/metrics/feature_importances.json,")
    print("         outputs/metrics/rf_search.csv, outputs/predictions/rf_val_preds.npy")
    print("  → Hand off rf_model.pkl + feature_importances.json to Person 3 + 4")
    print("=" * 60)

    return best_model, best_params, final_m


if __name__ == "__main__":
    run_phase2()
