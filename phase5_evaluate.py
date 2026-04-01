"""
Phase 5 — Final Test Set Evaluation
Person 2 (Models & Evaluation)

THIS IS THE ONLY TIME YOU OPEN TEST LABELS.
Run this exactly once, after all models are trained and ensemble weights locked.
Never use these results to go back and retune anything.

Produces:
  outputs/test_metrics_table.csv  → Person 4 dashboard
  outputs/ablation_test.csv       → final report
  outputs/calibration_data.json   → Person 4 scatter plot
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

from phase0_skeleton import (
    FEATURE_COLUMNS,
    N_TEST,
    RANDOM_STATE,
    dummy_features,
    evaluate,
    metrics_table,
)

np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "models_saved"

USE_DUMMY_DATA = False  # ← flip False when real data arrives
BERT_READY = False  # ← flip True after Colab BERT run


def get_test_data():
    """Load test features and labels. Called ONCE in this file only."""
    if USE_DUMMY_DATA:
        print("  [INFO] Using dummy test data.")
        X_test_raw, y_test = dummy_features(N_TEST, seed=RANDOM_STATE + 2)
        return X_test_raw, y_test, FEATURE_COLUMNS
    else:
        test_df = pd.read_csv(FEATURES_DIR / "test_features.csv")
        col_names = FEATURE_COLUMNS
        X_test = test_df[col_names].values.astype(float)
        y_test = test_df["csat_score"].values.astype(float)
        return X_test, y_test, col_names


def load_models_and_weights():
    """Load all saved artefacts."""
    artefacts = {}

    ridge_path = MODELS_DIR / "ridge_model.pkl"
    rf_path = MODELS_DIR / "rf_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    weights_path = OUTPUTS_DIR / "ensemble_weights.json"

    if not ridge_path.exists():
        raise FileNotFoundError("ridge_model.pkl missing — run phase1 first")
    if not rf_path.exists():
        raise FileNotFoundError("rf_model.pkl missing — run phase2 first")
    if not weights_path.exists():
        raise FileNotFoundError("ensemble_weights.json missing — run phase4 first")

    with open(ridge_path, "rb") as f:
        artefacts["ridge"] = pickle.load(f)
    with open(rf_path, "rb") as f:
        artefacts["rf"] = pickle.load(f)

    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            artefacts["scaler"] = pickle.load(f)
    else:
        artefacts["scaler"] = None  # dummy mode — no scaler saved

    with open(weights_path) as f:
        artefacts["weights"] = json.load(f)

    return artefacts


def run_phase5():
    print("=" * 60)
    print("PHASE 5 — FINAL TEST SET EVALUATION")
    print("  !! Test labels opened here for the first and only time !!")
    print("=" * 60)

    # Load test data
    print("\n── Loading test data ──")
    X_test_raw, y_test, col_names = get_test_data()
    print(
        f"  X_test: {X_test_raw.shape}  y_test range: [{y_test.min():.1f}, {y_test.max():.1f}]"
    )

    # Load models
    print("\n── Loading saved models ──")
    art = load_models_and_weights()
    print(f"  Loaded: ridge_model, rf_model, ensemble_weights={art['weights']}")

    # Scale for Ridge (RF uses raw)
    if art["scaler"] is not None:
        X_test_scaled = art["scaler"].transform(X_test_raw)
    else:
        # Dummy path: fit a scaler on test itself (acceptable only for dummy runs)
        from sklearn.preprocessing import StandardScaler

        X_test_scaled = StandardScaler().fit_transform(X_test_raw)

    # Generate predictions
    ridge_preds = np.clip(art["ridge"].predict(X_test_scaled), 1.0, 5.0)
    rf_preds = np.clip(art["rf"].predict(X_test_raw), 1.0, 5.0)

    if BERT_READY:
        bert_preds = np.load(OUTPUTS_DIR / "bert_test_preds.npy")
        bert_preds = np.clip(bert_preds, 1.0, 5.0)
        bert_label = "DistilBERT"
    else:
        bert_preds = np.clip(np.random.default_rng(99).uniform(1, 5, N_TEST), 1.0, 5.0)
        bert_label = "DistilBERT (dummy)"

    w = art["weights"]
    ensemble_preds = np.clip(
        w["ridge"] * ridge_preds + w["rf"] * rf_preds + w["bert"] * bert_preds,
        1.0,
        5.0,
    )

    # Evaluate all models
    print("\n── Test set metrics ──")
    results = {
        "Ridge": evaluate(y_test, ridge_preds, model_name="Ridge"),
        "Random Forest": evaluate(y_test, rf_preds, model_name="Random Forest"),
        bert_label: evaluate(y_test, bert_preds, model_name=bert_label),
        "Ensemble": evaluate(y_test, ensemble_preds, model_name="Ensemble"),
    }

    table = metrics_table(results)
    print("\n── Final test metrics table (for Person 4 dashboard) ──")
    print(table.to_string())

    # Important finding: DistilBERT vs Ridge
    ridge_mae = results["Ridge"]["mae"]
    bert_key = bert_label
    bert_mae = results[bert_key]["mae"]
    if bert_mae > ridge_mae:
        print(
            f"\n  [FINDING] DistilBERT MAE ({bert_mae:.4f}) > Ridge MAE ({ridge_mae:.4f})"
        )
        print("  → Expected. Synthetic transcripts give BERT no real language signal.")
        print("    Document this explicitly in your report as a genuine finding.")

    # Prediction spread check
    print(
        f"\n  Ensemble pred range: [{ensemble_preds.min():.2f}, {ensemble_preds.max():.2f}]"
    )
    print(
        f"  Ensemble pred std:   {ensemble_preds.std():.4f}  (true std: {y_test.std():.4f})"
    )

    # Ablation on test set (re-run Ridge ablation using saved scaler)
    print("\n── Ablation on test set ──")
    from phase1_ridge import FEATURE_GROUPS

    ablation_rows = []
    baseline_mae = ridge_preds  # use full-feature Ridge predictions

    baseline_test_mae = evaluate(y_test, ridge_preds)["mae"]
    ablation_rows.append(
        {"run": "all_features", "mae": round(baseline_test_mae, 4), "mae_delta": 0.0}
    )

    for grp_name, grp_cols in FEATURE_GROUPS.items():
        keep_idx = [col_names.index(c) for c in col_names if c not in grp_cols]
        X_t = X_test_scaled[:, keep_idx]
        # Refit Ridge on reduced test is NOT correct — we should ideally save reduced models
        # For now: approximate by zeroing out removed features on the full model
        # This is acceptable for report purposes; note the approximation.
        X_zeroed = X_test_scaled.copy()
        for c in grp_cols:
            if c in col_names:
                X_zeroed[:, col_names.index(c)] = 0.0
        preds = np.clip(art["ridge"].predict(X_zeroed), 1.0, 5.0)
        m = evaluate(y_test, preds)
        delta = m["mae"] - baseline_test_mae
        print(f"  remove_{grp_name:<20}  MAE={m['mae']:.4f}  delta={delta:+.4f}")
        ablation_rows.append(
            {
                "run": f"remove_{grp_name}",
                "mae": round(m["mae"], 4),
                "mae_delta": round(delta, 4),
            }
        )

    pd.DataFrame(ablation_rows).to_csv(OUTPUTS_DIR / "ablation_test.csv", index=False)

    # Calibration data for Person 4 scatter plot
    calibration = {
        "y_true": y_test.tolist(),
        "ensemble": ensemble_preds.tolist(),
        "ridge": ridge_preds.tolist(),
        "rf": rf_preds.tolist(),
        "dataset_mean": float(y_test.mean()),
    }
    cal_path = OUTPUTS_DIR / "calibration_data.json"
    with open(cal_path, "w") as f:
        json.dump(calibration, f)
    print(f"\n  Saved calibration data → {cal_path}")

    # Save final metrics table
    table_path = OUTPUTS_DIR / "test_metrics_table.csv"
    table.to_csv(table_path)
    print(f"  Saved metrics table   → {table_path}  ← hand off to Person 4")

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE — do not open test labels again.")
    print("  Next: run phase6_predict_fn.py")
    print("=" * 60)

    return results, ensemble_preds


if __name__ == "__main__":
    run_phase5()
