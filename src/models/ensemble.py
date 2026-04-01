"""
Phase 4 — Ensemble Weight Selection
Person 2 (Models & Evaluation)

Combines Ridge + RF + DistilBERT predictions using a weighted sum.
Evaluates at least 4 weight configurations on the validation set and
selects the one with the lowest validation MAE.

If BERT val preds are unavailable (BERT_READY=False), falls back to
a two-model Ridge+RF ensemble with renormalised weights.

Outputs:
  reports/ensemble_weights.json   ← Person 3 needs this for /predict endpoint
  reports/ensemble_val_preds.npy

RUN ORDER:
  python src/models/ridge.py
  python src/models/random_forest.py
  [notebooks/bert.ipynb on Colab]  ← optional
  python src/models/ensemble.py    ← this file
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from phase0_skeleton import (
    N_VAL,
    RANDOM_STATE,
    dummy_features,
    evaluate,
    metrics_table,
)

np.random.seed(RANDOM_STATE)

OUTPUTS_DIR = ROOT / "outputs"          # ridge_val_preds.npy lives here
MODELS_SAVED_DIR = ROOT / "models_saved"  # rf_val_preds.npy lives here
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Set True once notebooks/bert.ipynb has run and saved bert_val_preds.npy
BERT_READY = True


def load_val_preds():
    """
    Load val predictions from Phase 1 (ridge) and Phase 2 (rf).
    Also loads y_val from val_features.csv.
    """
    ridge_path = OUTPUTS_DIR / "ridge_val_preds.npy"
    rf_path = MODELS_SAVED_DIR / "rf_val_preds.npy"
    bert_path = MODELS_SAVED_DIR / "bert_val_preds.npy"

    if not ridge_path.exists():
        raise FileNotFoundError(
            f"ridge_val_preds.npy missing at {ridge_path} — run src/models/ridge.py first"
        )
    if not rf_path.exists():
        raise FileNotFoundError(
            f"rf_val_preds.npy missing at {rf_path} — run src/models/random_forest.py first"
        )

    ridge_preds = np.load(ridge_path)
    rf_preds = np.load(rf_path)

    # Load y_val from CSV
    val_csv = ROOT / "data" / "processed" / "val_features.csv"
    if val_csv.exists():
        import pandas as pd
        val_df = pd.read_csv(val_csv)
        # Fix repeat_contact
        if "repeat_contact" in val_df.columns:
            val_df["repeat_contact"] = (
                val_df["repeat_contact"]
                .astype(str).str.strip().str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0})
                .fillna(0).astype(float)
            )
        y_val = val_df["csat_score"].values.astype(float)
    else:
        print("  [WARN] val_features.csv not found — using dummy y_val")
        _, y_val = dummy_features(N_VAL, seed=RANDOM_STATE + 1)

    # Align lengths (in case of minor mismatch)
    n = min(len(y_val), len(ridge_preds), len(rf_preds))
    y_val = y_val[:n]
    ridge_preds = ridge_preds[:n]
    rf_preds = rf_preds[:n]

    if BERT_READY:
        if not bert_path.exists():
            raise FileNotFoundError(
                f"bert_val_preds.npy missing but BERT_READY=True. "
                f"Run notebooks/bert.ipynb on Colab first."
            )
        bert_preds = np.load(bert_path)[:n]
        print(f"  Loaded BERT val preds from {bert_path}")
    else:
        print("  [INFO] BERT_READY=False — using dummy BERT preds (uniform random).")
        print("         Set BERT_READY=True after bert.ipynb produces bert_val_preds.npy")
        bert_preds = np.clip(
            np.random.default_rng(99).uniform(1, 5, n), 1.0, 5.0
        )

    return y_val, ridge_preds, rf_preds, bert_preds


def pick_best_weights(y_val, ridge_preds, rf_preds, bert_preds):
    """
    Evaluate at least 4 weight configurations on val set.
    Returns the config with the strictly lowest val MAE.
    Weights always sum to 1.0.
    """
    candidates = [
        {"ridge": 1/3, "rf": 1/3, "bert": 1/3},       # equal weights
        {"ridge": 0.5, "rf": 0.3, "bert": 0.2},        # ridge-heavy
        {"ridge": 0.4, "rf": 0.4, "bert": 0.2},        # balanced
        {"ridge": 0.3, "rf": 0.5, "bert": 0.2},        # rf-heavy
        {"ridge": 0.5, "rf": 0.5, "bert": 0.0},        # no BERT (fallback)
        {"ridge": 0.6, "rf": 0.4, "bert": 0.0},        # ridge-dominant no BERT
    ]

    # Verify all weights sum to 1.0
    for c in candidates:
        assert abs(sum(c.values()) - 1.0) < 1e-6, f"Weights don't sum to 1: {c}"

    print("\n── Weight search ──")
    best_mae, best_weights = float("inf"), candidates[0]

    for w in candidates:
        ens = np.clip(
            w["ridge"] * ridge_preds + w["rf"] * rf_preds + w["bert"] * bert_preds,
            1.0, 5.0,
        )
        m = evaluate(y_val, ens)
        label = f"r={w['ridge']:.2f} rf={w['rf']:.2f} b={w['bert']:.2f}"
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

    print("\n── Individual model val metrics ──")
    r_m = evaluate(y_val, ridge_preds, model_name="Ridge")
    rf_m = evaluate(y_val, rf_preds, model_name="Random Forest")
    b_m = evaluate(
        y_val, bert_preds,
        model_name="DistilBERT (dummy)" if not BERT_READY else "DistilBERT",
    )

    best_weights = pick_best_weights(y_val, ridge_preds, rf_preds, bert_preds)

    ensemble_preds = np.clip(
        best_weights["ridge"] * ridge_preds
        + best_weights["rf"] * rf_preds
        + best_weights["bert"] * bert_preds,
        1.0, 5.0,
    )
    ens_m = evaluate(y_val, ensemble_preds, model_name="Ensemble (val)")

    # CI width
    stacked = np.stack([ridge_preds, rf_preds, bert_preds], axis=1)
    ci_std = stacked.std(axis=1)
    print(f"\n  Mean CI width (±1 std): {ci_std.mean():.4f}")

    print("\n── Val metrics table ──")
    table = metrics_table({
        "Ridge": r_m,
        "Random Forest": rf_m,
        "DistilBERT": b_m,
        "Ensemble": ens_m,
    })
    print(table.to_string())

    # Save ensemble weights → reports/ensemble_weights.json
    weights_path = REPORTS_DIR / "ensemble_weights.json"
    with open(weights_path, "w") as f:
        json.dump(best_weights, f, indent=2)
    print(f"\n  Saved → {weights_path}  ← hand off to Person 3")

    # Save ensemble val preds
    np.save(REPORTS_DIR / "ensemble_val_preds.npy", ensemble_preds)

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print(f"  Weights: {best_weights}")
    print(f"  Val MAE: {ens_m['mae']:.4f}")
    print("  Next: python src/evaluation/evaluate.py")
    print("=" * 60)

    return best_weights, ens_m


if __name__ == "__main__":
    run_phase4()
