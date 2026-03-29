"""
Phase 4 — Ensemble
Person 2 (Models & Evaluation)

Combines Ridge + RF + DistilBERT predictions with equal weights (1/3 each).
If one model clearly dominates on validation, you can adjust weights here
and update ensemble_weights.json before Phase 5.

Saves ensemble_weights.json → Person 3 needs this for the /predict endpoint.

Run after phase3_bert.ipynb (Colab) has saved bert_val_preds.npy locally.
If BERT preds aren't ready yet, set BERT_READY = False to run a 2-model
ensemble for now and update when they arrive.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

from phase0_skeleton import (
    N_VAL,
    RANDOM_STATE,
    dummy_features,
    evaluate,
    metrics_table,
)

np.random.seed(RANDOM_STATE)

OUTPUTS_DIR = Path(__file__).parent / "outputs"
BERT_READY = False  # ← set True once phase3_bert.ipynb has run


def load_val_preds() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load val predictions saved by Phase 1, 2, and 3. Returns (y_val, ridge, rf, bert)."""

    # Ground truth — dummy or real depending on what Phase 1/2 used
    _, y_val = dummy_features(N_VAL, seed=RANDOM_STATE + 1)

    ridge_path = OUTPUTS_DIR / "ridge_val_preds.npy"
    rf_path = OUTPUTS_DIR / "rf_val_preds.npy"
    bert_path = OUTPUTS_DIR / "bert_val_preds.npy"

    if not ridge_path.exists():
        raise FileNotFoundError(
            "ridge_val_preds.npy missing — run phase1_ridge.py first"
        )
    if not rf_path.exists():
        raise FileNotFoundError(
            "rf_val_preds.npy missing — run phase2_random_forest.py first"
        )

    ridge_preds = np.load(ridge_path)
    rf_preds = np.load(rf_path)

    if BERT_READY:
        if not bert_path.exists():
            raise FileNotFoundError("bert_val_preds.npy missing but BERT_READY=True")
        bert_preds = np.load(bert_path)
    else:
        print("  [INFO] BERT_READY=False — using dummy BERT preds (uniform random).")
        print(
            "         Set BERT_READY=True after phase3_bert.ipynb produces bert_val_preds.npy"
        )
        bert_preds = np.clip(np.random.default_rng(99).uniform(1, 5, N_VAL), 1.0, 5.0)

    return y_val, ridge_preds, rf_preds, bert_preds


def pick_best_weights(
    y_val: np.ndarray,
    ridge_preds: np.ndarray,
    rf_preds: np.ndarray,
    bert_preds: np.ndarray,
) -> dict:
    """
    Try equal weights and a few alternatives; pick by val MAE.
    Returns the winning weight dict.
    """
    candidates = [
        {"ridge": 1 / 3, "rf": 1 / 3, "bert": 1 / 3},  # equal
        {"ridge": 0.5, "rf": 0.3, "bert": 0.2},  # Ridge-heavy
        {"ridge": 0.4, "rf": 0.4, "bert": 0.2},  # RF-heavy
        {
            "ridge": 0.5,
            "rf": 0.5,
            "bert": 0.0,
        },  # no BERT (good fallback if BERT is poor)
    ]

    print("\n── Weight search ──")
    best_mae, best_weights = float("inf"), candidates[0]
    for w in candidates:
        ens = np.clip(
            w["ridge"] * ridge_preds + w["rf"] * rf_preds + w["bert"] * bert_preds,
            1.0,
            5.0,
        )
        m = evaluate(y_val, ens)
        label = f"r={w['ridge']:.1f} rf={w['rf']:.1f} b={w['bert']:.1f}"
        print(f"  {label}  →  MAE={m['mae']:.4f}  r={m['pearson_r']:.4f}")
        if m["mae"] < best_mae:
            best_mae, best_weights = m["mae"], w

    print(f"\n  Best weights: {best_weights}  (val MAE={best_mae:.4f})")
    return best_weights


def run_phase4():
    print("=" * 60)
    print("PHASE 4 — ENSEMBLE")
    print("=" * 60)

    y_val, ridge_preds, rf_preds, bert_preds = load_val_preds()

    # Individual model metrics for comparison
    print("\n── Individual model val metrics ──")
    r_m = evaluate(y_val, ridge_preds, model_name="Ridge")
    rf_m = evaluate(y_val, rf_preds, model_name="Random Forest")
    b_m = evaluate(
        y_val,
        bert_preds,
        model_name="DistilBERT (dummy)" if not BERT_READY else "DistilBERT",
    )

    # Weight search
    best_weights = pick_best_weights(y_val, ridge_preds, rf_preds, bert_preds)

    # Final ensemble predictions
    ensemble_preds = np.clip(
        best_weights["ridge"] * ridge_preds
        + best_weights["rf"] * rf_preds
        + best_weights["bert"] * bert_preds,
        1.0,
        5.0,
    )
    ens_m = evaluate(y_val, ensemble_preds, model_name="Ensemble (val)")

    # Simple confidence interval: ±1 std of the three model predictions per call
    stacked = np.stack([ridge_preds, rf_preds, bert_preds], axis=1)
    ci_std = stacked.std(axis=1)
    print(f"\n  Mean CI width (±1 std): {ci_std.mean():.4f}")
    print(f"  This is the confidence_interval field in the predict() output")

    # Summary table
    print("\n── Val metrics table (preview of Phase 5 format) ──")
    table = metrics_table(
        {
            "Ridge": r_m,
            "Random Forest": rf_m,
            "DistilBERT": b_m,
            "Ensemble": ens_m,
        }
    )
    print(table.to_string())

    # Save ensemble weights JSON — Person 3 needs this
    weights_path = Path(__file__).parent / "outputs" / "ensemble_weights.json"
    with open(weights_path, "w") as f:
        json.dump(best_weights, f, indent=2)
    print(f"\n  Saved → {weights_path}  ← hand off to Person 3")

    # Save ensemble val preds for Phase 5
    np.save(OUTPUTS_DIR / "ensemble_val_preds.npy", ensemble_preds)

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print(f"  Weights: {best_weights}")
    print(f"  Val MAE: {ens_m['mae']:.4f}")
    print("  Next: run phase5_evaluate.py")
    print("=" * 60)

    return best_weights, ens_m


if __name__ == "__main__":
    run_phase4()
