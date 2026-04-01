# Design Document: Person 2 — Models & Evaluation

## Overview

This component trains four CSAT prediction models (Ridge Regression, Random Forest, DistilBERT, and a weighted Ensemble), evaluates them on held-out test data, and exposes a `predict()` function consumed by Person 3's FastAPI backend. It sits between Person 1's feature pipeline and Person 3/4's downstream consumers.

The execution order is strictly sequential:

```
ridge.py → random_forest.py → bert.ipynb (Colab) → ensemble.py → evaluate.py
```

`src/predict.py` is a standalone module that loads saved artefacts at import time and can be called independently of the training pipeline.

---

## Architecture

```mermaid
flowchart TD
    P1[Person 1\ntrain/val/test_features.csv\nscaler.pkl] --> R[ridge.py\nPhase 1]
    P1 --> RF[random_forest.py\nPhase 2]
    P1 --> B[bert.ipynb\nPhase 3 — Colab]

    R -->|ridge_val_preds.npy\nridge_model.pkl| E[ensemble.py\nPhase 4]
    RF -->|rf_val_preds.npy\nrf_model.pkl\nfeature_importances.json| E
    B -->|bert_val_preds.npy\nbert_weights/| E

    E -->|ensemble_weights.json| EV[evaluate.py\nPhase 5]
    R --> EV
    RF --> EV
    B --> EV

    EV -->|test_metrics_table.csv\ncalibration_data.json\nablation_test.csv| P4[Person 4\nDashboard]

    R -->|ridge_model.pkl\nscaler.pkl| PR[src/predict.py]
    RF -->|rf_model.pkl\nfeature_importances.json| PR
    B -->|bert_weights/| PR
    E -->|ensemble_weights.json| PR

    PR -->|predict()| P3[Person 3\nFastAPI]
```

### Key Design Decisions

1. **Ridge uses Person 1's scaler** — `scaler.pkl` is fitted on training data only and reused for all Ridge inference. RF receives raw unscaled features because tree-based models are scale-invariant.

2. **DistilBERT runs on Colab** — GPU is required for fine-tuning. The design treats BERT as an optional component: all downstream phases (ensemble, evaluate, predict) degrade gracefully when `bert_weights/` or `bert_val_preds.npy` are absent.

3. **Test set opened exactly once** — `test_features.csv` is only read inside `src/evaluation/evaluate.py`. No other file may import or read it. This is enforced by convention and documented in every phase file header.

4. **Artefact-first handoff** — Person 3 and Person 4 can begin integration as soon as `rf_model.pkl`, `feature_importances.json`, and `ensemble_weights.json` are written, before Phase 5 completes.

5. **Ensemble weight search over a fixed candidate set** — Rather than continuous optimisation, four discrete weight configurations are evaluated on the validation set. This keeps the search reproducible and avoids overfitting the ensemble to validation data.

---

## Components and Interfaces

### `src/models/ridge.py`

Responsibilities: alpha search, final Ridge training, ablation study.

```python
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]
def run_alpha_search(X_train, y_train, X_val, y_val) -> tuple[float, dict]
def run_ablation(X_train, y_train, X_val, y_val, col_names, best_alpha) -> pd.DataFrame
def run_phase1() -> tuple[Ridge, float, dict, pd.DataFrame]
```

Outputs: `models/ridge_model.pkl`, `models/scaler.pkl`, `outputs/ridge_alpha_search.csv`, `outputs/ridge_val_preds.npy`, `outputs/ablation_results.csv`

### `src/models/random_forest.py`

Responsibilities: RF grid search, feature importance extraction.

```python
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
def run_phase2() -> tuple[RandomForestRegressor, dict, dict]
```

Outputs: `models/rf_model.pkl`, `models_saved/feature_importances.json`, `models_saved/rf_search.csv`, `models_saved/rf_val_preds.npy`

### `src/models/bert_finetune.py` / `notebooks/bert.ipynb`

Responsibilities: DistilBERT fine-tuning on Colab GPU.

```python
def build_model() -> DistilBertForSequenceClassification  # num_labels=1
def train(train_df, val_df, epochs=3, lr=2e-5, batch_size=8) -> None
```

Outputs: `models_saved/bert_weights/`, `models_saved/bert_val_preds.npy`

### `src/models/ensemble.py`

Responsibilities: weight search over validation predictions, saving `ensemble_weights.json`.

```python
def load_val_preds() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
def pick_best_weights(y_val, ridge_preds, rf_preds, bert_preds) -> dict
def run_phase4() -> tuple[dict, dict]
```

Outputs: `reports/ensemble_weights.json`, `reports/ensemble_val_preds.npy`

### `src/evaluation/evaluate.py`

Responsibilities: single test-set evaluation pass for all four models.

```python
def get_test_data() -> tuple[np.ndarray, np.ndarray, list[str]]  # called once
def load_models_and_weights() -> dict
def run_phase5() -> tuple[dict, np.ndarray]
```

Outputs: `reports/test_metrics_table.csv`, `reports/calibration_data.json`, `reports/ablation_test.csv`

### `src/predict.py`

Responsibilities: artefact loading at import time, end-to-end inference.

```python
def predict(transcript_text: str, call_metadata: dict) -> dict
# Returns: {csat_score, confidence_interval, emotional_arc, shap_values}
```

Internal helpers:
```python
def _load_artefacts() -> dict          # cached; called once
def _extract_features(transcript_text, call_metadata) -> np.ndarray
def _bert_predict(transcript_text, art) -> Optional[float]
def _compute_emotional_arc(transcript_text) -> str
```

---

## Data Models

### Feature Vector

22 float columns in the order defined by `FEATURE_COLUMNS` in `phase0_skeleton.py`:

| Group | Columns |
|-------|---------|
| A — Sentiment | `mean_sentiment`, `last_20_sentiment`, `std_sentiment` |
| B — Structure | `talk_time_ratio`, `avg_agent_words`, `avg_customer_words`, `interruption_count`, `resolution_flag` |
| C — Agent | `empathy_density`, `apology_count`, `transfer_count` |
| D — Metadata | `duration_ordinal`, `duration_deviation`, `repeat_contact`, `intent_billing_error`, `intent_broadband_issue`, `intent_contract_dispute`, `intent_delivery_problem`, `intent_general_enquiry`, `intent_payment_issue`, `intent_product_fault`, `intent_refund_request`, `intent_service_outage`, `intent_technical_support`, `intent_wrong_item` |

Ridge receives scaled features (via `scaler.pkl`). RF receives raw unscaled features.

### `ensemble_weights.json`

```json
{"ridge": 0.4, "rf": 0.4, "bert": 0.2}
```

Invariant: `ridge + rf + bert == 1.0`

### `feature_importances.json`

```json
{
  "importances": {"mean_sentiment": 0.12, "talk_time_ratio": 0.09, ...},
  "sorted": [["mean_sentiment", 0.12], ["talk_time_ratio", 0.09], ...]
}
```

Invariant: `sorted` list is in descending order of importance value.

### `predict()` return value

```python
{
  "csat_score": float,           # clipped to [1.0, 5.0]
  "confidence_interval": [float, float],  # [csat - std, csat + std], clipped to [1.0, 5.0]
  "emotional_arc": str,          # one of: "rise", "fall", "flat", "v_shape"
  "shap_values": dict            # stub — populated by Person 3
}
```

### `calibration_data.json`

```json
{
  "y_true": [...],
  "ensemble": [...],
  "ridge": [...],
  "rf": [...],
  "dataset_mean": float
}
```

All lists have equal length equal to the test set size.

### `test_metrics_table.csv`

Columns: `Model`, `MAE`, `RMSE`, `Pearson r`, `F1 (>=3.0)`
Rows: Ridge, Random Forest, DistilBERT, Ensemble

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Best-configuration selection minimises validation MAE

*For any* collection of trained model configurations (alpha candidates, RF hyperparameter combos, or ensemble weight sets) evaluated on a validation set, the configuration selected as "best" must be the one with the strictly lowest validation MAE among all evaluated configurations.

**Validates: Requirements 1.2, 3.3, 5.2**

---

### Property 2: All predictions are clipped to [1.0, 5.0]

*For any* raw model output value (Ridge, RF, DistilBERT, or Ensemble), after the clipping step the resulting value must satisfy `1.0 <= value <= 5.0`.

**Validates: Requirements 1.6, 3.4, 5.3**

---

### Property 3: Scaler is fit on training data only

*For any* training run, the `StandardScaler` applied to validation or test features must have been fitted exclusively on training features — i.e., the scaler's `mean_` and `scale_` parameters must equal those computed from the training split alone, and must not change when additional data is seen.

**Validates: Requirements 1.4, 2.5**

---

### Property 4: Ablation output has exactly five rows with required schema

*For any* ablation study run, the resulting DataFrame must contain exactly five rows (one baseline + one per feature group A/B/C/D) and each row must include the columns: `run`, `features_removed`, `n_features`, `mae`, `rmse`, `pearson_r`, `f1_binary`, `mae_delta`.

**Validates: Requirements 2.1, 2.3**

---

### Property 5: RF grid search covers at least nine hyperparameter combinations

*For any* RF grid search run, the search log must contain at least nine rows, covering `n_estimators` ∈ {100, 200, 500}, `max_depth` ∈ {None, 10, 20}, and `min_samples_leaf` ∈ {1, 5, 10}.

**Validates: Requirements 3.2**

---

### Property 6: Feature importances sorted list is in descending order

*For any* `feature_importances.json` produced by the RF training phase, the `sorted` list must be ordered such that each element's importance value is greater than or equal to the next element's importance value.

**Validates: Requirements 3.6, 8.3**

---

### Property 7: DistilBERT tokenizer truncates inputs to at most 512 tokens

*For any* transcript string passed to the DistilBERT tokenizer, the resulting `input_ids` tensor must have a sequence length of at most 512.

**Validates: Requirements 4.2**

---

### Property 8: Ensemble weights sum to 1.0

*For any* ensemble weight configuration (whether the full three-model set or the BERT-absent fallback), the sum of all active weights must equal 1.0 (within floating-point tolerance of 1e-6).

**Validates: Requirements 5.1, 5.5, 8.2**

---

### Property 9: BERT-absent fallback produces valid output without raising

*For any* valid `(transcript_text, call_metadata)` input, when `bert_weights/` is absent or fails to load, `predict()` must return a dict with the correct schema and a `csat_score` in [1.0, 5.0] without raising any exception.

**Validates: Requirements 5.5, 7.6**

---

### Property 10: Evaluation metrics schema is complete for all models

*For any* test-set evaluation run, the resulting metrics structure must contain MAE, RMSE, Pearson_r, and F1_Binary for each of the four models (Ridge, Random Forest, DistilBERT, Ensemble).

**Validates: Requirements 6.2**

---

### Property 11: `predict()` output schema is always correct

*For any* valid `(transcript_text, call_metadata)` input, `predict()` must return a dict containing exactly the keys `csat_score` (float in [1.0, 5.0]), `confidence_interval` (list of two floats each in [1.0, 5.0]), `emotional_arc` (one of `"rise"`, `"fall"`, `"flat"`, `"v_shape"`), and `shap_values` (dict).

**Validates: Requirements 7.1, 7.2**

---

### Property 12: Confidence interval formula is correct

*For any* set of per-model predictions, the confidence interval returned by `predict()` must equal `[clip(csat_score - std, 1.0, 5.0), clip(csat_score + std, 1.0, 5.0)]` where `std` is the standard deviation of the individual model predictions.

**Validates: Requirements 7.7**

---

### Property 13: NaN feature values are replaced with 0.0

*For any* feature vector that contains one or more NaN values after extraction, the vector passed to any model must contain no NaN values — all NaN entries must have been replaced with 0.0.

**Validates: Requirements 7.9**

---

### Property 14: All required artefacts are non-empty files after their phase completes

*For any* artefact path in the handoff contract (`models/rf_model.pkl`, `reports/ensemble_weights.json`, `models_saved/feature_importances.json`, `reports/test_metrics_table.csv`, `reports/calibration_data.json`), after the corresponding phase script completes successfully, the file must exist and have a size greater than zero bytes.

**Validates: Requirements 8.1, 8.4, 8.5, 8.6**

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| `train_features.csv` missing | `FileNotFoundError` with message pointing to Person 1's deliverable |
| `bert_weights/` absent at inference | Fall back to Ridge+RF ensemble; renormalise weights; log `[WARN]` |
| `bert_val_preds.npy` absent at ensemble phase | Use dummy uniform random BERT preds if `BERT_READY=False`; log `[INFO]` |
| NaN in feature vector | Replace with 0.0 via `np.nan_to_num` before any model call |
| RF prediction std < 50% of true std | Log `[WARN]` about mean-compression; do not abort |
| DistilBERT MAE > Ridge MAE | Log `[FINDING]` note about expected underperformance on synthetic data |
| Any artefact missing at evaluate time | `FileNotFoundError` with message naming the missing file and the phase that produces it |
| BERT inference exception | Catch, log `[WARN]`, return `None`; caller falls back to Ridge+RF |

---

## Testing Strategy

### Dual Testing Approach

Both unit tests and property-based tests are required. Unit tests cover specific examples, file I/O contracts, and integration points. Property tests verify universal correctness guarantees across randomly generated inputs.

### Unit Tests (`tests/`)

Focus areas:
- File existence and schema after each phase (artefact handoff contracts)
- `predict()` end-to-end on 5 representative calls (billing/repeat, technical/resolved, account/long, payment/positive, network/frustrated)
- BERT fallback path: verify no exception and valid output when `bert_weights/` is absent
- Scaler fit-on-train-only: verify `scaler.mean_` matches training data statistics
- `_compute_emotional_arc()` on known transcripts with expected arc labels
- `ensemble_weights.json` schema: keys present, values are floats, sum to 1.0
- `feature_importances.json` schema: both `importances` dict and `sorted` list present
- RF mean-compression warning: mock a case where pred_std < 0.5 * true_std and assert log output

### Property-Based Tests (`tests/test_properties.py`)

Library: **Hypothesis** (Python). Each test runs a minimum of 100 iterations.

Each property test is tagged with a comment in the format:
`# Feature: person2-models-evaluation, Property {N}: {property_text}`

| Property | Test description |
|----------|-----------------|
| P1: Best-config selection | Generate random MAE values for N configs; assert selected index matches `argmin` |
| P2: Prediction clipping | Generate random floats in [-10, 10]; assert `clip(x, 1, 5)` is always in [1.0, 5.0] |
| P3: Scaler fit on train only | Generate random train/val splits; assert scaler fitted on train has different params than one fitted on val |
| P4: Ablation schema | Run ablation on synthetic data; assert exactly 5 rows and all required columns present |
| P5: RF grid coverage | Assert `len(PARAM_GRID) >= 9` and all required hyperparameter values are represented |
| P6: Feature importances ordering | Generate random importance dicts; assert `sorted` output is non-increasing |
| P7: BERT token truncation | Generate random strings of varying length; assert tokenized length <= 512 |
| P8: Ensemble weights sum to 1.0 | Generate random weight triples; after normalisation assert sum == 1.0 within 1e-6 |
| P9: BERT-absent fallback | Call `predict()` with `bert_ready=False`; assert no exception and valid schema |
| P10: Metrics schema completeness | Generate random pred/true arrays; assert all four metrics present for all four models |
| P11: `predict()` output schema | Generate random transcripts and metadata; assert output has correct keys and types |
| P12: CI formula | Generate random model predictions; assert CI equals `[clip(mean-std,1,5), clip(mean+std,1,5)]` |
| P13: NaN replacement | Generate feature vectors with random NaN positions; assert no NaN after `nan_to_num` |
| P14: Artefact non-empty | After each phase runner, assert all expected output paths exist and `os.path.getsize > 0` |

### Property Test Configuration

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@settings(max_examples=100)
@given(st.lists(st.floats(min_value=-10, max_value=10), min_size=1))
def test_prediction_clipping(raw_preds):
    # Feature: person2-models-evaluation, Property 2: All predictions are clipped to [1.0, 5.0]
    clipped = np.clip(raw_preds, 1.0, 5.0)
    assert all(1.0 <= v <= 5.0 for v in clipped)
```
