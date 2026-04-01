"""
Phase 5 — Final Test Set Evaluation
Person 2 (Models & Evaluation)

!! THIS IS THE ONLY FILE THAT OPENS test_features.csv !!
!! Run exactly once. Do not re-run to adjust models or weights. !!

Produces:
  reports/test_metrics_table.csv   → Person 4 dashboard
  reports/calibration_data.json    → Person 4 scatter plot
  reports/ablation_test.csv        → final report

RUN ORDER:
  python src/models/ridge.py
  python src/models/random_forest.py
  [notebooks/bert.ipynb on Colab]
  python src/models/ensemble.py
  python src/evaluation/evaluate.py   ← this file (LAST)
"""

import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from phase0_skeleton import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    evaluate,
    metrics_table,
)

np.random.seed(RANDOM_STATE)

FEATURES_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_SAVED_DIR = ROOT / "models_saved"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

BERT_READY = True  # set True after Colab BERT run


def _clean_test_df(df: pd.DataFrame) -> pd.DataFrame:
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


def get_test_data():
    """
    Load test features and labels.
    Called ONCE in this file only — never import test_features.csv elsewhere.
    """
    test_path = FEATURES_DIR / "test_features.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Expected {test_path}. Has Person 1 delivered test_features.csv?"
        )

    test_df = _clean_test_df(pd.read_csv(test_path))
    X_test = test_df[FEATURE_COLUMNS].values.astype(float)
    y_test = test_df["csat_score"].values.astype(float)
    return X_test, y_test, FEATURE_COLUMNS


def load_models_and_weights():
    """Load all saved artefacts. Raises FileNotFoundError with clear messages."""
    ridge_path = MODELS_DIR / "ridge_model.pkl"
    rf_path = MODELS_DIR / "rf_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    weights_path = REPORTS_DIR / "ensemble_weights.json"

    for path, phase in [
        (ridge_path, "src/models/ridge.py"),
        (rf_path, "src/models/random_forest.py"),
        (weights_path, "src/models/ensemble.py"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path.name} missing — run {phase} first")

    artefacts = {}
    with open(ridge_path, "rb") as f:
        artefacts["ridge"] = pickle.load(f)
    with open(rf_path, "rb") as f:
        artefacts["rf"] = pickle.load(f)
    with open(weights_path) as f:
        artefacts["weights"] = json.load(f)

    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            artefacts["scaler"] = pickle.load(f)
    else:
        artefacts["scaler"] = None

    return artefacts


def run_phase5():
    print("=" * 60)
    print("PHASE 5 — FINAL TEST SET EVALUATION")
    print("  !! Test labels opened here for the first and only time !!")
    print("=" * 60)

    # Load test data (once)
    print("\n── Loading test data ──")
    X_test_raw, y_test, col_names = get_test_data()
    print(f"  X_test: {X_test_raw.shape}  y_test range: [{y_test.min():.1f}, {y_test.max():.1f}]")

    # Load models
    print("\n── Loading saved models ──")
    art = load_models_and_weights()
    print(f"  Loaded: ridge_model, rf_model, ensemble_weights={art['weights']}")

    # Scale for Ridge
    if art["scaler"] is not None:
        X_test_scaled = art["scaler"].transform(X_test_raw)
    else:
        from sklearn.preprocessing import StandardScaler
        X_test_scaled = StandardScaler().fit_transform(X_test_raw)

    # Predictions
    ridge_preds = np.clip(art["ridge"].predict(X_test_scaled), 1.0, 5.0)
    rf_preds = np.clip(art["rf"].predict(X_test_raw), 1.0, 5.0)

    if BERT_READY:
        bert_test_path = MODELS_SAVED_DIR / "bert_test_preds.npy"
        if not bert_test_path.exists():
            raise FileNotFoundError(
                "bert_test_preds.npy missing but BERT_READY=True. "
                "Run bert.ipynb on test set first."
            )
        bert_preds = np.clip(np.load(bert_test_path), 1.0, 5.0)
        bert_label = "DistilBERT"
    else:
        bert_preds = np.clip(
            np.random.default_rng(99).uniform(1, 5, len(y_test)), 1.0, 5.0
        )
        bert_label = "DistilBERT (dummy)"

    w = art["weights"]
    ensemble_preds = np.clip(
        w["ridge"] * ridge_preds + w["rf"] * rf_preds + w["bert"] * bert_preds,
        1.0, 5.0,
    )

    # Naive baseline — always predict training mean
    train_df = pd.read_csv(FEATURES_DIR / "train_features.csv")
    train_mean = float(train_df["csat_score"].mean())
    baseline_preds = np.full_like(y_test, train_mean)

    # Metrics
    print("\n── Test set metrics ──")
    results = {
        "Naive (mean)": evaluate(y_test, baseline_preds, model_name="Naive (mean)"),
        "Ridge": evaluate(y_test, ridge_preds, model_name="Ridge"),
        "Random Forest": evaluate(y_test, rf_preds, model_name="Random Forest"),
        bert_label: evaluate(y_test, bert_preds, model_name=bert_label),
        "Ensemble": evaluate(y_test, ensemble_preds, model_name="Ensemble"),
    }

    # FINDING: DistilBERT vs Ridge
    ridge_mae = results["Ridge"]["mae"]
    bert_mae = results[bert_label]["mae"]
    if bert_mae > ridge_mae:
        print(f"\n  [FINDING] DistilBERT MAE ({bert_mae:.4f}) > Ridge MAE ({ridge_mae:.4f})")
        print("  → Expected. Synthetic transcripts give BERT no real language signal.")
        print("    Document this explicitly in the final report.")

    # Ensemble spread
    print(f"\n  Ensemble pred range: [{ensemble_preds.min():.2f}, {ensemble_preds.max():.2f}]")
    print(f"  Ensemble pred std:   {ensemble_preds.std():.4f}  (true std: {y_test.std():.4f})")

    # Metrics table
    table = metrics_table(results)
    print("\n── Final test metrics table ──")
    print(table.to_string())

    # Save metrics table (Person 4 needs this)
    table_path = REPORTS_DIR / "test_metrics_table.csv"
    table.reset_index().to_csv(table_path, index=False)
    print(f"\n  Saved → {table_path}  ← hand off to Person 4")

    # Calibration data (Person 4 scatter plot)
    calibration = {
        "y_true": y_test.tolist(),
        "ensemble": ensemble_preds.tolist(),
        "ridge": ridge_preds.tolist(),
        "rf": rf_preds.tolist(),
        "dataset_mean": float(y_test.mean()),
    }
    cal_path = REPORTS_DIR / "calibration_data.json"
    with open(cal_path, "w") as f:
        json.dump(calibration, f)
    print(f"  Saved → {cal_path}  ← hand off to Person 4")

    # Ablation on test set
    print("\n── Ablation on test set ──")
    from ridge import FEATURE_GROUPS

    ablation_rows = []
    baseline_test_mae = results["Ridge"]["mae"]
    ablation_rows.append({
        "run": "all_features",
        "features_removed": "none",
        "n_features": len(col_names),
        "mae": round(baseline_test_mae, 4),
        "mae_delta": 0.0,
    })

    for grp_name, grp_cols in FEATURE_GROUPS.items():
        X_zeroed = X_test_scaled.copy()
        for c in grp_cols:
            if c in col_names:
                X_zeroed[:, col_names.index(c)] = 0.0
        preds = np.clip(art["ridge"].predict(X_zeroed), 1.0, 5.0)
        m = evaluate(y_test, preds)
        delta = m["mae"] - baseline_test_mae
        print(f"  remove_{grp_name:<20}  MAE={m['mae']:.4f}  delta={delta:+.4f}")
        ablation_rows.append({
            "run": f"remove_{grp_name}",
            "features_removed": grp_name,
            "n_features": len(col_names) - len(grp_cols),
            "mae": round(m["mae"], 4),
            "mae_delta": round(delta, 4),
        })

    ablation_path = REPORTS_DIR / "ablation_test.csv"
    pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)
    print(f"\n  Saved → {ablation_path}")

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE — do not open test labels again.")
    print("  Next: python src/predict.py")
    print("=" * 60)

    return results, ensemble_preds


if __name__ == "__main__":
    run_phase5()
