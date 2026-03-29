"""
Phase 2 — Random Forest Regressor
Person 2 (Models & Evaluation)

Trains RF with ~9 hyperparameter combos, picks best by val MAE,
saves rf_model.pkl and feature_importances.json.

HANDOFF: rf_model.pkl + feature_importances.json → Person 3 (SHAP) + Person 4 (dashboard)
         This is your highest-priority deliverable. Hand it off the moment this runs.

Switch USE_DUMMY_DATA = False once Person 1 delivers feature CSVs.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

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

ROOT = Path(__file__).parent
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

USE_DUMMY_DATA = True  # ← flip to False when Person 1 delivers

# Hyperparameter grid — 9 combos
PARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
    for n in [100, 200, 500]
    for d in [None, 10]
    for l in [1, 5]
][:9]  # cap at 9


def load_data():
    if USE_DUMMY_DATA:
        print("  [INFO] Using dummy data.")
        X_train_raw, y_train = dummy_features(N_TRAIN, seed=RANDOM_STATE)
        X_val_raw, y_val = dummy_features(N_VAL, seed=RANDOM_STATE + 1)
        col_names = FEATURE_COLUMNS
        # RF does not need scaled inputs — use raw
        return X_train_raw, y_train, X_val_raw, y_val, col_names
    else:
        import pandas as pd

        train_df = pd.read_csv(FEATURES_DIR / "train_features.csv")
        val_df = pd.read_csv(FEATURES_DIR / "val_features.csv")
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

    # Grid search
    print("\n── Hyperparameter search ──")
    best_mae, best_params, best_model = float("inf"), None, None
    search_rows = []

    for params in PARAM_GRID:
        rf = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = np.clip(rf.predict(X_val), 1.0, 5.0)
        m = evaluate(y_val, preds)
        label = f"n={params['n_estimators']} d={params['max_depth']} l={params['min_samples_leaf']}"
        print(f"  {label:<30}  MAE={m['mae']:.4f}  r={m['pearson_r']:.4f}")
        search_rows.append({"params": str(params), **m})
        if m["mae"] < best_mae:
            best_mae, best_params, best_model = m["mae"], params, rf

    print(f"\n  Best params: {best_params}  (val MAE={best_mae:.4f})")

    # Prediction spread check (mean-compression warning)
    best_preds = np.clip(best_model.predict(X_val), 1.0, 5.0)
    print(f"\n  RF pred range: [{best_preds.min():.2f}, {best_preds.max():.2f}]")
    print(f"  RF pred std:   {best_preds.std():.4f}  (true std: {y_val.std():.4f})")
    if best_preds.std() < y_val.std() * 0.5:
        print("  [WARN] RF compressing predictions toward mean — document in report.")

    final_m = evaluate(y_val, best_preds, model_name="Random Forest (best, val)")

    # Feature importances
    importances = dict(zip(col_names, best_model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 feature importances:")
    for feat, imp in sorted_imp[:10]:
        print(f"    {feat:<35} {imp:.4f}")

    # Save model
    model_path = MODELS_DIR / "rf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n  Saved → {model_path}")

    # Save feature importances JSON (Person 3 + Person 4 need this)
    imp_path = OUTPUTS_DIR / "feature_importances.json"
    with open(imp_path, "w") as f:
        json.dump({"importances": importances, "sorted": sorted_imp}, f, indent=2)
    print(f"  Saved → {imp_path}  ← hand off to Person 3 + 4 now")

    # Save search log
    pd.DataFrame(search_rows).to_csv(OUTPUTS_DIR / "rf_search.csv", index=False)

    # Save val predictions for Phase 4 ensemble
    np.save(OUTPUTS_DIR / "rf_val_preds.npy", best_preds)
    print(f"  Saved val preds → outputs/rf_val_preds.npy  (Phase 4 needs this)")

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE — hand off rf_model.pkl + feature_importances.json")
    print("=" * 60)

    return best_model, best_params, final_m


if __name__ == "__main__":
    run_phase2()
