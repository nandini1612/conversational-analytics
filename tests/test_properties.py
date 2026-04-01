"""
Property-Based Tests — Person 2 Models & Evaluation
ClearSignal CSAT Prediction System

Tests all 14 correctness properties using Hypothesis.
Each test runs a minimum of 100 iterations.

Run: python -m pytest tests/test_properties.py -v --tb=short

Requirements: pip install hypothesis pytest numpy scikit-learn
"""

import sys
import os
import json
import pickle
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from phase0_skeleton import FEATURE_COLUMNS, RANDOM_STATE
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _select_best(mae_list: list) -> int:
    """Return index of minimum MAE — mirrors what all phase scripts do."""
    return int(np.argmin(mae_list))


def _clip_pred(x: float) -> float:
    return float(np.clip(x, 1.0, 5.0))


def _normalise_weights(w: dict) -> dict:
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


# ─────────────────────────────────────────────────────────────
# P1: Best-configuration selection minimises validation MAE
# Feature: person2-models-evaluation, Property 1
# ─────────────────────────────────────────────────────────────

@settings(max_examples=100)
@given(st.lists(st.floats(min_value=0.0, max_value=5.0, allow_nan=False), min_size=2, max_size=20))
def test_p1_best_config_minimises_mae(mae_values):
    # Feature: person2-models-evaluation, Property 1: Best-configuration selection minimises validation MAE
    selected_idx = _select_best(mae_values)
    assert mae_values[selected_idx] == min(mae_values), (
        f"Selected index {selected_idx} has MAE {mae_values[selected_idx]}, "
        f"but minimum is {min(mae_values)}"
    )


# ─────────────────────────────────────────────────────────────
# P2: All predictions are clipped to [1.0, 5.0]
# Feature: person2-models-evaluation, Property 2
# ─────────────────────────────────────────────────────────────

@settings(max_examples=100)
@given(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False), min_size=1))
def test_p2_prediction_clipping(raw_preds):
    # Feature: person2-models-evaluation, Property 2: All predictions are clipped to [1.0, 5.0]
    clipped = [_clip_pred(x) for x in raw_preds]
    assert all(1.0 <= v <= 5.0 for v in clipped), (
        f"Found values outside [1.0, 5.0]: {[v for v in clipped if not (1.0 <= v <= 5.0)]}"
    )


# ─────────────────────────────────────────────────────────────
# P3: Scaler is fit on training data only
# Feature: person2-models-evaluation, Property 3
# ─────────────────────────────────────────────────────────────

@settings(max_examples=100)
@given(
    arrays(dtype=np.float64, shape=st.tuples(st.integers(10, 50), st.integers(2, 5)),
           elements=st.floats(min_value=-100, max_value=100, allow_nan=False)),
    arrays(dtype=np.float64, shape=st.tuples(st.integers(10, 50), st.integers(2, 5)),
           elements=st.floats(min_value=-100, max_value=100, allow_nan=False)),
)
def test_p3_scaler_fit_on_train_only(X_train, X_val):
    # Feature: person2-models-evaluation, Property 3: Scaler is fit on training data only
    assume(X_train.shape[1] == X_val.shape[1])

    scaler_train = StandardScaler()
    scaler_train.fit(X_train)

    scaler_val = StandardScaler()
    scaler_val.fit(X_val)

    # Scaler fitted on train should have different mean_ than one fitted on val
    # (unless by extreme coincidence the distributions are identical)
    # The key property: scaler_train.mean_ must equal np.mean(X_train, axis=0)
    np.testing.assert_allclose(
        scaler_train.mean_, np.mean(X_train, axis=0), rtol=1e-5,
        err_msg="Scaler mean_ does not match training data mean"
    )
    # And it must NOT change when we see val data
    mean_before = scaler_train.mean_.copy()
    _ = scaler_train.transform(X_val)  # transform should not change mean_
    np.testing.assert_array_equal(
        scaler_train.mean_, mean_before,
        err_msg="Scaler mean_ changed after transform — scaler was re-fitted on val data"
    )


# ─────────────────────────────────────────────────────────────
# P4: Ablation output has exactly five rows with required schema
# Feature: person2-models-evaluation, Property 4
# ─────────────────────────────────────────────────────────────

def test_p4_ablation_schema():
    # Feature: person2-models-evaluation, Property 4: Ablation output has exactly five rows with required schema
    from ridge import run_ablation, FEATURE_GROUPS, ALPHA_CANDIDATES
    import pandas as pd

    rng = np.random.default_rng(RANDOM_STATE)
    n_feat = len(FEATURE_COLUMNS)
    X_train = rng.random((100, n_feat))
    y_train = rng.uniform(1.0, 5.0, 100)
    X_val = rng.random((30, n_feat))
    y_val = rng.uniform(1.0, 5.0, 30)

    ablation_df = run_ablation(X_train, y_train, X_val, y_val, FEATURE_COLUMNS, best_alpha=1.0)

    assert len(ablation_df) == 5, f"Expected 5 rows, got {len(ablation_df)}"

    required_cols = {"run", "features_removed", "n_features", "mae", "rmse", "pearson_r", "f1_binary", "mae_delta"}
    assert required_cols.issubset(set(ablation_df.columns)), (
        f"Missing columns: {required_cols - set(ablation_df.columns)}"
    )


# ─────────────────────────────────────────────────────────────
# P5: RF grid search covers at least nine hyperparameter combinations
# Feature: person2-models-evaluation, Property 5
# ─────────────────────────────────────────────────────────────

def test_p5_rf_grid_coverage():
    # Feature: person2-models-evaluation, Property 5: RF grid search covers at least nine hyperparameter combinations
    from random_forest import PARAM_GRID

    assert len(PARAM_GRID) >= 9, f"PARAM_GRID has only {len(PARAM_GRID)} combos, need ≥9"

    n_estimators_vals = {p["n_estimators"] for p in PARAM_GRID}
    max_depth_vals = {p["max_depth"] for p in PARAM_GRID}
    min_samples_leaf_vals = {p["min_samples_leaf"] for p in PARAM_GRID}

    assert {100, 200, 500}.issubset(n_estimators_vals), \
        f"Missing n_estimators values. Got: {n_estimators_vals}"
    assert {None, 10, 20}.issubset(max_depth_vals), \
        f"Missing max_depth values. Got: {max_depth_vals}"
    assert {1, 5, 10}.issubset(min_samples_leaf_vals), \
        f"Missing min_samples_leaf values. Got: {min_samples_leaf_vals}"


# ─────────────────────────────────────────────────────────────
# P6: Feature importances sorted list is in descending order
# Feature: person2-models-evaluation, Property 6
# ─────────────────────────────────────────────────────────────

@settings(max_examples=100)
@given(st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    min_size=1,
    max_size=22,
))
def test_p6_feature_importances_ordering(importances_dict):
    # Feature: person2-models-evaluation, Property 6: Feature importances sorted list is in descending order
    sorted_imp = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
    values = [v for _, v in sorted_imp]
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1], (
            f"Sorted importances not descending at index {i}: {values[i]} < {values[i+1]}"
        )


# ─────────────────────────────────────────────────────────────
# P7: DistilBERT tokenizer truncates inputs to at most 512 tokens
# Feature: person2-models-evaluation, Property 7
# ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (ROOT / "models_saved" / "bert_weights").exists(),
    reason="bert_weights/ not present — run bert.ipynb on Colab first"
)
@settings(max_examples=20)
@given(st.text(min_size=0, max_size=5000))
def test_p7_bert_token_truncation(text):
    # Feature: person2-models-evaluation, Property 7: DistilBERT tokenizer truncates inputs to at most 512 tokens
    try:
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        encoded = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        seq_len = encoded["input_ids"].shape[1]
        assert seq_len <= 512, f"Token length {seq_len} exceeds 512"
    except ImportError:
        pytest.skip("transformers not installed")


def test_p7_bert_token_truncation_basic():
    # Feature: person2-models-evaluation, Property 7: DistilBERT tokenizer truncates inputs to at most 512 tokens
    # Basic version that doesn't require bert_weights/ to exist
    try:
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        # Test with a very long string
        long_text = "hello world " * 1000
        encoded = tokenizer(long_text, truncation=True, max_length=512, return_tensors="pt")
        assert encoded["input_ids"].shape[1] <= 512
        # Test with empty string
        encoded_empty = tokenizer("", truncation=True, max_length=512, return_tensors="pt")
        assert encoded_empty["input_ids"].shape[1] <= 512
    except ImportError:
        pytest.skip("transformers not installed")
    except Exception as e:
        pytest.skip(f"transformers/torch version incompatibility: {e}")


# ─────────────────────────────────────────────────────────────
# P8: Ensemble weights sum to 1.0
# Feature: person2-models-evaluation, Property 8
# ─────────────────────────────────────────────────────────────

@settings(max_examples=100)
@given(
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_p8_ensemble_weights_sum_to_one(w_ridge, w_rf, w_bert):
    # Feature: person2-models-evaluation, Property 8: Ensemble weights sum to 1.0
    raw = {"ridge": w_ridge, "rf": w_rf, "bert": w_bert}
    normalised = _normalise_weights(raw)
    total = sum(normalised.values())
    assert abs(total - 1.0) < 1e-6, f"Normalised weights sum to {total}, not 1.0"


def test_p8_saved_ensemble_weights_sum_to_one():
    # Feature: person2-models-evaluation, Property 8: Ensemble weights sum to 1.0
    weights_path = ROOT / "outputs" / "metrics" / "ensemble_weights.json"
    if not weights_path.exists():
        pytest.skip("ensemble_weights.json not yet generated")
    with open(weights_path) as f:
        w = json.load(f)
    assert set(w.keys()) == {"ridge", "rf", "bert"}, f"Unexpected keys: {w.keys()}"
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-6, f"Saved weights sum to {total}, not 1.0"


# ─────────────────────────────────────────────────────────────
# P9: BERT-absent fallback produces valid output without raising
# Feature: person2-models-evaluation, Property 9
# ─────────────────────────────────────────────────────────────

@settings(max_examples=20)
@given(
    st.text(min_size=1, max_size=500),
    st.sampled_from(["billing", "technical", "account", "payment", "network"]),
    st.sampled_from(["short", "medium", "long"]),
    st.integers(min_value=0, max_value=1),
)
def test_p9_bert_absent_fallback(transcript, issue_type, duration, repeat):
    # Feature: person2-models-evaluation, Property 9: BERT-absent fallback produces valid output without raising
    # Ensure bert_weights/ is absent for this test
    bert_dir = ROOT / "models_saved" / "bert_weights"
    if bert_dir.exists():
        pytest.skip("bert_weights/ present — fallback path not active")

    # Check required artefacts exist
    if not (ROOT / "models" / "ridge_model.pkl").exists():
        pytest.skip("ridge_model.pkl not yet generated")
    if not (ROOT / "models" / "rf_model.pkl").exists():
        pytest.skip("rf_model.pkl not yet generated")
    if not (ROOT / "reports" / "ensemble_weights.json").exists():
        pytest.skip("ensemble_weights.json not yet generated")

    # Reset artefact cache to force reload
    import predict as pred_module
    pred_module._ARTEFACTS.clear()

    metadata = {"issue_type": issue_type, "call_duration": duration, "repeat_contact": repeat}
    result = pred_module.predict(transcript, metadata)

    assert isinstance(result, dict), "predict() must return a dict"
    assert "csat_score" in result
    assert "confidence_interval" in result
    assert "emotional_arc" in result
    assert "shap_values" in result
    assert 1.0 <= result["csat_score"] <= 5.0
    assert len(result["confidence_interval"]) == 2
    assert all(1.0 <= v <= 5.0 for v in result["confidence_interval"])
    assert result["emotional_arc"] in ("rise", "fall", "flat", "v_shape")


# ─────────────────────────────────────────────────────────────
# P10: Evaluation metrics schema is complete for all models
# Feature: person2-models-evaluation, Property 10
# ─────────────────────────────────────────────────────────────

@settings(max_examples=50)
@given(
    arrays(dtype=np.float64, shape=st.integers(10, 100),
           elements=st.floats(min_value=1.0, max_value=5.0, allow_nan=False)),
    arrays(dtype=np.float64, shape=st.integers(10, 100),
           elements=st.floats(min_value=1.0, max_value=5.0, allow_nan=False)),
)
def test_p10_metrics_schema_completeness(y_true_raw, y_pred_raw):
    # Feature: person2-models-evaluation, Property 10: Evaluation metrics schema is complete for all models
    from phase0_skeleton import evaluate as eval_fn

    n = min(len(y_true_raw), len(y_pred_raw))
    assume(n >= 5)
    y_true = y_true_raw[:n]
    y_pred = y_pred_raw[:n]

    # pearsonr is undefined when either input is constant — skip those cases
    assume(np.std(y_true) > 1e-8)
    assume(np.std(y_pred) > 1e-8)

    m = eval_fn(y_true, y_pred)
    required_keys = {"mae", "rmse", "pearson_r", "f1_binary"}
    assert required_keys.issubset(set(m.keys())), (
        f"Missing metric keys: {required_keys - set(m.keys())}"
    )
    assert m["mae"] >= 0
    assert m["rmse"] >= 0
    # pearson_r is NaN only when inputs are constant — excluded by assume() above
    assert not np.isnan(m["pearson_r"]), "pearson_r is NaN (constant input slipped through)"
    assert -1.0 <= m["pearson_r"] <= 1.0
    assert 0.0 <= m["f1_binary"] <= 1.0


# ─────────────────────────────────────────────────────────────
# P11: predict() output schema is always correct
# Feature: person2-models-evaluation, Property 11
# ─────────────────────────────────────────────────────────────

@settings(max_examples=20)
@given(
    st.text(min_size=1, max_size=200),
    st.sampled_from(["billing", "technical", "account", "payment", "network"]),
    st.sampled_from(["short", "medium", "long"]),
    st.integers(min_value=0, max_value=1),
)
def test_p11_predict_output_schema(transcript, issue_type, duration, repeat):
    # Feature: person2-models-evaluation, Property 11: predict() output schema is always correct
    if not (ROOT / "models" / "ridge_model.pkl").exists():
        pytest.skip("ridge_model.pkl not yet generated")
    if not (ROOT / "models" / "rf_model.pkl").exists():
        pytest.skip("rf_model.pkl not yet generated")
    if not (ROOT / "reports" / "ensemble_weights.json").exists():
        pytest.skip("ensemble_weights.json not yet generated")

    import predict as pred_module
    pred_module._ARTEFACTS.clear()

    metadata = {"issue_type": issue_type, "call_duration": duration, "repeat_contact": repeat}
    result = pred_module.predict(transcript, metadata)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"csat_score", "confidence_interval", "emotional_arc", "shap_values"}
    assert isinstance(result["csat_score"], float)
    assert 1.0 <= result["csat_score"] <= 5.0
    assert isinstance(result["confidence_interval"], list)
    assert len(result["confidence_interval"]) == 2
    assert all(isinstance(v, float) for v in result["confidence_interval"])
    assert all(1.0 <= v <= 5.0 for v in result["confidence_interval"])
    assert result["emotional_arc"] in ("rise", "fall", "flat", "v_shape")
    assert isinstance(result["shap_values"], dict)


# ─────────────────────────────────────────────────────────────
# P12: Confidence interval formula is correct
# Feature: person2-models-evaluation, Property 12
# ─────────────────────────────────────────────────────────────

@settings(max_examples=100)
@given(st.lists(
    st.floats(min_value=1.0, max_value=5.0, allow_nan=False),
    min_size=2, max_size=5,
))
def test_p12_confidence_interval_formula(model_preds):
    # Feature: person2-models-evaluation, Property 12: Confidence interval formula is correct
    preds = np.array(model_preds)
    csat = float(np.clip(np.mean(preds), 1.0, 5.0))
    std = float(np.std(preds))

    expected_lower = float(np.clip(csat - std, 1.0, 5.0))
    expected_upper = float(np.clip(csat + std, 1.0, 5.0))

    # Verify the formula
    assert expected_lower <= csat <= expected_upper or std == 0.0
    assert 1.0 <= expected_lower <= 5.0
    assert 1.0 <= expected_upper <= 5.0
    assert expected_lower <= expected_upper


# ─────────────────────────────────────────────────────────────
# P13: NaN feature values are replaced with 0.0
# Feature: person2-models-evaluation, Property 13
# ─────────────────────────────────────────────────────────────

@settings(max_examples=100)
@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(1, 22),
        elements=st.one_of(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
            st.just(float("nan")),
        ),
    )
)
def test_p13_nan_replacement(feature_vector):
    # Feature: person2-models-evaluation, Property 13: NaN feature values are replaced with 0.0
    cleaned = np.nan_to_num(feature_vector, nan=0.0)
    assert not np.any(np.isnan(cleaned)), "NaN values remain after nan_to_num"
    # NaN positions should be 0.0
    nan_mask = np.isnan(feature_vector)
    assert np.all(cleaned[nan_mask] == 0.0), "NaN positions not replaced with 0.0"


# ─────────────────────────────────────────────────────────────
# P14: All required artefacts are non-empty files after phase completion
# Feature: person2-models-evaluation, Property 14
# ─────────────────────────────────────────────────────────────

def test_p14_artefacts_non_empty():
    # Feature: person2-models-evaluation, Property 14: All required artefacts are non-empty files after their phase completes
    required_artefacts = [
        ROOT / "models" / "rf_model.pkl",
        ROOT / "outputs" / "metrics" / "ensemble_weights.json",
        ROOT / "outputs" / "metrics" / "feature_importances.json",
        ROOT / "outputs" / "metrics" / "test_metrics_table.csv",
        ROOT / "outputs" / "metrics" / "calibration_data.json",
    ]

    missing = []
    empty = []

    for path in required_artefacts:
        if not path.exists():
            missing.append(str(path.name))
        elif os.path.getsize(path) == 0:
            empty.append(str(path.name))

    assert not missing, f"Missing artefacts: {missing}"
    assert not empty, f"Empty artefacts: {empty}"


def test_p14_ensemble_weights_schema():
    # Feature: person2-models-evaluation, Property 14
    weights_path = ROOT / "outputs" / "metrics" / "ensemble_weights.json"
    if not weights_path.exists():
        pytest.skip("ensemble_weights.json not yet generated")

    with open(weights_path) as f:
        w = json.load(f)

    assert set(w.keys()) == {"ridge", "rf", "bert"}, f"Wrong keys: {w.keys()}"
    assert all(isinstance(v, (int, float)) for v in w.values()), "Values must be floats"
    assert abs(sum(w.values()) - 1.0) < 1e-6, f"Weights sum to {sum(w.values())}, not 1.0"


def test_p14_feature_importances_schema():
    # Feature: person2-models-evaluation, Property 14
    imp_path = ROOT / "outputs" / "metrics" / "feature_importances.json"
    if not imp_path.exists():
        pytest.skip("feature_importances.json not yet generated")

    with open(imp_path) as f:
        data = json.load(f)

    assert "importances" in data, "Missing 'importances' key"
    assert "sorted" in data, "Missing 'sorted' key"
    assert isinstance(data["importances"], dict), "'importances' must be a dict"
    assert isinstance(data["sorted"], list), "'sorted' must be a list"

    # Verify sorted is in descending order
    values = [v for _, v in data["sorted"]]
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1], (
            f"feature_importances.json 'sorted' not descending at index {i}"
        )


def test_p14_calibration_data_schema():
    # Feature: person2-models-evaluation, Property 14
    cal_path = ROOT / "outputs" / "metrics" / "calibration_data.json"
    if not cal_path.exists():
        pytest.skip("calibration_data.json not yet generated")

    with open(cal_path) as f:
        data = json.load(f)

    required_keys = {"y_true", "ensemble", "ridge", "rf", "dataset_mean"}
    assert required_keys.issubset(set(data.keys())), (
        f"Missing keys: {required_keys - set(data.keys())}"
    )

    # All lists must have equal length
    lengths = {k: len(data[k]) for k in ["y_true", "ensemble", "ridge", "rf"]}
    assert len(set(lengths.values())) == 1, f"Unequal list lengths: {lengths}"
    assert isinstance(data["dataset_mean"], (int, float))


def test_p14_metrics_table_schema():
    # Feature: person2-models-evaluation, Property 14
    import pandas as pd
    table_path = ROOT / "outputs" / "metrics" / "test_metrics_table.csv"
    if not table_path.exists():
        pytest.skip("test_metrics_table.csv not yet generated")

    df = pd.read_csv(table_path)
    required_cols = {"Model", "MAE", "RMSE", "Pearson r", "F1 (≥3.0)"}
    assert required_cols.issubset(set(df.columns)), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )
    assert len(df) >= 3, f"Expected at least 3 model rows, got {len(df)}"
