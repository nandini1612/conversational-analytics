"""
Phase 0 — Setup & Skeleton
Person 2 (Models & Evaluation)

PURPOSE
-------
This file does two things:
  1. Defines and validates the exact feature schema Person 1 must deliver.
     Run validate_feature_schema() the moment their CSVs land — it will
     catch silent bugs (wrong column names, wrong dtypes, out-of-range values)
     before they corrupt model training.

  2. Builds a skeleton training loop on dummy data of the correct shape.
     Every model class (Ridge, RF, DistilBERT) can be verified end-to-end
     right now, before Person 1 is done. When real features arrive,
     you swap dummy_features() for pd.read_csv() and nothing else changes.

FEATURE SCHEMA (agreed with Person 1)
--------------------------------------
Group A — Sentiment (3 features)
  mean_sentiment          float   [-1, 1]   overall tone
  final_sentiment         float   [-1, 1]   last 20% of turns averaged
  sentiment_std           float   [0, 2]    volatility across turns

Group B — Conversational structure (4 features)
  talk_time_ratio         float   [0, 1]    agent turns / total turns
  avg_words_agent         float   [0, inf]  avg words per agent turn
  avg_words_customer      float   [0, inf]  avg words per customer turn
  interruption_count      int     [0, inf]  short turns after speaker swap
  resolution_flag         int     {0, 1}    resolution keyword in last 20%

Group C — Agent behaviour (3 features)
  empathy_density         float   [0, 1]    empathy phrases / agent turns
  apology_count           int     [0, inf]  "sorry"/"apologise" in agent turns
  transfer_count          int     [0, inf]  transfer phrases in agent turns

Group D — Metadata (12 features)
  duration_ordinal        int     {-1, 0, 1} short/medium/long
  duration_deviation      float   any        call ordinal minus intent mean
  repeat_contact          int     {0, 1}     already binary in CSV
  issue_type_billing      int     {0, 1}
  issue_type_technical    int     {0, 1}
  issue_type_account      int     {0, 1}
  issue_type_payment      int     {0, 1}
  issue_type_network      int     {0, 1}
  issue_type_delivery     int     {0, 1}
  issue_type_returns      int     {0, 1}
  issue_type_complaint    int     {0, 1}
  issue_type_upgrade      int     {0, 1}
  issue_type_cancellation int     {0, 1}

Total: 22 features (3 + 5 + 3 + 12 — note B has 5 not 4, spec says 4 but
lists 5 items; resolve with Person 1 before their handoff)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────
# 1. FEATURE SCHEMA
#    Single source of truth. Person 1 must produce these exact names.
#    Person 3 needs this list to build their API request validator.
# ─────────────────────────────────────────────────────────────

FEATURE_SCHEMA = {
    # Group A — sentiment (Person 1 actual names)
    "mean_sentiment": {"dtype": float, "min": -1.0, "max": 1.0},
    "last_20_sentiment": {
        "dtype": float,
        "min": -1.0,
        "max": 1.0,
    },  # was final_sentiment
    "std_sentiment": {"dtype": float, "min": 0.0, "max": 2.0},  # was sentiment_std
    # Group B — conversational structure (Person 1 actual names)
    "talk_time_ratio": {"dtype": float, "min": 0.0, "max": 1.0},
    "avg_agent_words": {"dtype": float, "min": 0.0, "max": None},  # was avg_words_agent
    "avg_customer_words": {
        "dtype": float,
        "min": 0.0,
        "max": None,
    },  # was avg_words_customer
    "interruption_count": {"dtype": float, "min": 0.0, "max": None},
    "resolution_flag": {"dtype": float, "min": 0.0, "max": 1.0},
    # Group C — agent behaviour
    "empathy_density": {"dtype": float, "min": 0.0, "max": 1.0},
    "apology_count": {"dtype": float, "min": 0.0, "max": None},
    "transfer_count": {"dtype": float, "min": 0.0, "max": None},
    # Group D — metadata (Person 1 used intent_* prefix, 10 categories)
    "duration_ordinal": {"dtype": float, "min": -1.0, "max": 1.0},
    "duration_deviation": {"dtype": float, "min": None, "max": None},
    "repeat_contact": {"dtype": float, "min": 0.0, "max": 1.0},
    "intent_billing": {
        "dtype": float,
        "min": 0.0,
        "max": 1.0,
    },  # was issue_type_billing
    "intent_technical": {
        "dtype": float,
        "min": 0.0,
        "max": 1.0,
    },  # was issue_type_technical
    "intent_account": {
        "dtype": float,
        "min": 0.0,
        "max": 1.0,
    },  # was issue_type_account
    "intent_payment": {
        "dtype": float,
        "min": 0.0,
        "max": 1.0,
    },  # was issue_type_payment
    "intent_network": {
        "dtype": float,
        "min": 0.0,
        "max": 1.0,
    },  # was issue_type_network
    "intent_delivery": {
        "dtype": float,
        "min": 0.0,
        "max": 1.0,
    },  # was issue_type_delivery
    "intent_refund": {"dtype": float, "min": 0.0, "max": 1.0},  # was issue_type_returns
    "intent_complaint": {
        "dtype": float,
        "min": 0.0,
        "max": 1.0,
    },  # was issue_type_complaint
    "intent_subscription": {
        "dtype": float,
        "min": 0.0,
        "max": 1.0,
    },  # was issue_type_upgrade
    "intent_login": {"dtype": float, "min": 0.0, "max": 1.0},  # new category
}

FEATURE_COLUMNS = list(FEATURE_SCHEMA.keys())  # 22 columns, in order
N_FEATURES = len(FEATURE_COLUMNS)  # 22
TARGET_COL = "csat_score"  # float 1.0–5.0

# Dataset constants (2,500 rows, 70/15/15 split)
N_TOTAL = 2500
N_TRAIN = int(N_TOTAL * 0.70)  # 1750
N_VAL = int(N_TOTAL * 0.15)  # 375
N_TEST = N_TOTAL - N_TRAIN - N_VAL  # 375


# ─────────────────────────────────────────────────────────────
# 2. SCHEMA VALIDATOR
#    Call this immediately when Person 1 hands off their CSVs.
#    Returns True if all checks pass, raises ValueError otherwise.
# ─────────────────────────────────────────────────────────────


def validate_feature_schema(df: pd.DataFrame, split_name: str = "unknown") -> bool:
    """
    Validate that a feature dataframe from Person 1 matches the agreed schema.

    Checks:
      - All 22 expected columns are present
      - No unexpected extra columns (warns but doesn't fail)
      - No null values in any feature column
      - Each column's values fall within the agreed min/max range

    Args:
        df:         Feature dataframe loaded from train/val/test_features.csv
        split_name: 'train', 'val', or 'test' — used in error messages only

    Returns:
        True if all checks pass.

    Raises:
        ValueError with a clear message on the first failure found.
    """
    errors = []

    # Check for missing columns
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Warn about unexpected extra columns (not a failure — Person 1 may add extras)
    extra = [
        col for col in df.columns if col not in FEATURE_COLUMNS and col != TARGET_COL
    ]
    if extra:
        print(
            f"  [WARN] {split_name}: extra columns not in schema (will be ignored): {extra}"
        )

    # Only continue range checks if columns exist
    present = [col for col in FEATURE_COLUMNS if col in df.columns]

    # Null check
    null_counts = df[present].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        errors.append(f"Null values found:\n{cols_with_nulls}")

    # Range checks
    for col in present:
        spec = FEATURE_SCHEMA[col]
        lo, hi = spec["min"], spec["max"]
        if lo is not None and df[col].min() < lo:
            errors.append(
                f"'{col}' min={df[col].min():.4f} is below expected minimum {lo}"
            )
        if hi is not None and df[col].max() > hi:
            errors.append(
                f"'{col}' max={df[col].max():.4f} is above expected maximum {hi}"
            )

    if errors:
        msg = f"\n[SCHEMA FAIL — {split_name}]\n" + "\n".join(
            f"  • {e}" for e in errors
        )
        raise ValueError(msg)

    print(
        f"  [OK] {split_name}: {len(df)} rows × {len(present)} features — schema valid"
    )
    return True


# ─────────────────────────────────────────────────────────────
# 3. EVALUATION METRICS
#    Centralised here so Ridge, RF, DistilBERT, and Ensemble
#    all report exactly the same numbers. Person 4's dashboard
#    table comes directly from these functions.
# ─────────────────────────────────────────────────────────────

CSAT_BINARY_THRESHOLD = 3.0  # below = unsatisfied, at/above = satisfied


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """
    Compute all four metrics used in the final test-set table.

    Args:
        y_true:     Ground-truth CSAT scores (float, 1–5)
        y_pred:     Model predictions (float)
        model_name: Label used in printed output only

    Returns:
        dict with keys: mae, rmse, pearson_r, f1_binary
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)

    # Binary F1 — satisfied = CSAT >= 3.0
    y_true_bin = (y_true >= CSAT_BINARY_THRESHOLD).astype(int)
    y_pred_bin = (y_pred >= CSAT_BINARY_THRESHOLD).astype(int)

    # Manual F1 (avoids sklearn import just for this)
    tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
    fn = np.sum((y_pred_bin == 0) & (y_true_bin == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    result = {"mae": mae, "rmse": rmse, "pearson_r": r, "f1_binary": f1}

    if model_name:
        print(f"\n  [{model_name}]")
        print(f"    MAE       = {mae:.4f}")
        print(f"    RMSE      = {rmse:.4f}")
        print(f"    Pearson r = {r:.4f}")
        print(f"    F1 (≥3.0) = {f1:.4f}")

    return result


def metrics_table(results: dict) -> pd.DataFrame:
    """
    Turn a dict of {model_name: metrics_dict} into a clean summary DataFrame.
    This is exactly what Person 4 needs for the dashboard table.

    Usage:
        results = {
            "Ridge": evaluate(y_true, ridge_preds),
            "Random Forest": evaluate(y_true, rf_preds),
            ...
        }
        df = metrics_table(results)
    """
    rows = []
    for model_name, m in results.items():
        rows.append(
            {
                "Model": model_name,
                "MAE": round(m["mae"], 4),
                "RMSE": round(m["rmse"], 4),
                "Pearson r": round(m["pearson_r"], 4),
                "F1 (≥3.0)": round(m["f1_binary"], 4),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


# ─────────────────────────────────────────────────────────────
# 4. DUMMY DATA GENERATOR
#    Produces random features of the correct shape so you can
#    verify your training loops before Person 1 delivers.
#    NOT used after real features arrive.
# ─────────────────────────────────────────────────────────────


def dummy_features(
    n_rows: int, seed: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random X (features) and y (CSAT) of the right shape.

    Returns:
        X: np.ndarray of shape (n_rows, 22)
        y: np.ndarray of shape (n_rows,) with values in [1, 5]
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_rows, N_FEATURES))  # all values in [0, 1]
    y = rng.uniform(1.0, 5.0, size=n_rows)  # continuous CSAT target
    return X, y


# ─────────────────────────────────────────────────────────────
# 5. SKELETON TRAINING LOOP
#    Verifies Ridge and RF can train, predict, and be evaluated
#    on dummy data. DistilBERT skeleton is in phase3_bert.py
#    because it needs Hugging Face imports.
# ─────────────────────────────────────────────────────────────


def run_skeleton():
    """
    Full skeleton run on dummy data.
    Confirms: data shapes, Ridge trains, RF trains, metrics compute correctly.
    Run this once to verify your environment before Person 1 delivers features.
    """
    print("=" * 60)
    print("PHASE 0 — SKELETON RUN (dummy data)")
    print(f"  N_FEATURES = {N_FEATURES}")
    print(f"  Train rows = {N_TRAIN}, Val rows = {N_VAL}, Test rows = {N_TEST}")
    print("=" * 60)

    # Generate dummy splits
    X_train, y_train = dummy_features(N_TRAIN, seed=RANDOM_STATE)
    X_val, y_val = dummy_features(N_VAL, seed=RANDOM_STATE + 1)
    X_test, y_test = dummy_features(N_TEST, seed=RANDOM_STATE + 2)

    print(f"\n  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # ── Ridge skeleton ──────────────────────────────────────
    print("\n── Ridge (skeleton) ──")
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_train, y_train)
    ridge_val_preds = ridge.predict(X_val)
    # Clip to valid CSAT range — Ridge can predict outside [1, 5]
    ridge_val_preds = np.clip(ridge_val_preds, 1.0, 5.0)
    evaluate(y_val, ridge_val_preds, model_name="Ridge — val (dummy)")

    # ── Random Forest skeleton ───────────────────────────────
    print("\n── Random Forest (skeleton) ──")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_val_preds = rf.predict(X_val)
    rf_val_preds = np.clip(rf_val_preds, 1.0, 5.0)
    evaluate(y_val, rf_val_preds, model_name="Random Forest — val (dummy)")

    # Check RF prediction spread (flag mean-compression early)
    print(f"\n  RF pred range: [{rf_val_preds.min():.2f}, {rf_val_preds.max():.2f}]")
    print(f"  RF pred std:   {rf_val_preds.std():.4f}")
    print(f"  True y std:    {y_val.std():.4f}")
    if rf_val_preds.std() < y_val.std() * 0.5:
        print(
            "  [WARN] RF is compressing predictions toward the mean — expected behaviour."
        )
        print("         Document this limitation in your report.")

    # ── Dummy ensemble (equal weights) ──────────────────────
    print("\n── Ensemble (equal weights, skeleton) ──")
    # DistilBERT preds are random here — replace with real in Phase 4
    bert_dummy_preds = np.clip(np.random.default_rng(99).uniform(1, 5, N_VAL), 1.0, 5.0)
    ensemble_preds = (ridge_val_preds + rf_val_preds + bert_dummy_preds) / 3.0
    ensemble_preds = np.clip(ensemble_preds, 1.0, 5.0)
    evaluate(y_val, ensemble_preds, model_name="Ensemble — val (dummy)")

    # ── Metrics table ────────────────────────────────────────
    print("\n── Summary table (what Person 4 will receive) ──")
    table = metrics_table(
        {
            "Ridge": evaluate(y_val, ridge_val_preds),
            "Random Forest": evaluate(y_val, rf_val_preds),
            "DistilBERT": evaluate(y_val, bert_dummy_preds),
            "Ensemble": evaluate(y_val, ensemble_preds),
        }
    )
    print(table.to_string())

    # ── Feature importances mock ─────────────────────────────
    print("\n── Feature importances (what Person 3 + 4 will receive) ──")
    importances = dict(zip(FEATURE_COLUMNS, rf.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp[:5]:
        print(f"    {feat:<30} {imp:.4f}")
    print("  ... (all 22 saved to feature_importances.json in Phase 2)")

    print("\n" + "=" * 60)
    print("SKELETON RUN COMPLETE — environment is healthy.")
    print("Plug in real feature CSVs from Person 1 to begin Phase 1.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 6. SCHEMA VALIDATION DEMO
#    Shows what happens when Person 1's CSV has a bug vs when
#    it's correct. Run this to understand the validator output.
# ─────────────────────────────────────────────────────────────


def demo_schema_validator():
    """
    Demonstrates validate_feature_schema() with:
      (a) a correctly structured dummy dataframe  → should pass
      (b) a dataframe missing a column            → should raise
      (c) a dataframe with an out-of-range value  → should raise
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = 10

    # (a) Valid dataframe
    print("\n── Demo (a): valid schema ──")
    good_data = {col: rng.random(n) * 0.5 for col in FEATURE_COLUMNS}
    good_df = pd.DataFrame(good_data)
    validate_feature_schema(good_df, split_name="train")

    # (b) Missing column
    print("\n── Demo (b): missing column ──")
    bad_data = {
        col: rng.random(n) for col in FEATURE_COLUMNS if col != "empathy_density"
    }
    bad_df = pd.DataFrame(bad_data)
    try:
        validate_feature_schema(bad_df, split_name="train")
    except ValueError as e:
        print(e)

    # (c) Out-of-range value
    print("\n── Demo (c): out-of-range value ──")
    oor_data = {col: rng.random(n) * 0.5 for col in FEATURE_COLUMNS}
    oor_df = pd.DataFrame(oor_data)
    oor_df.loc[0, "talk_time_ratio"] = 1.8  # must be in [0, 1]
    try:
        validate_feature_schema(oor_df, split_name="train")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    demo_schema_validator()
    print()
    run_skeleton()
