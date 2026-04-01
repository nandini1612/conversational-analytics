# Implementation Plan: Person 2 — Models & Evaluation

## Overview

Implement the full ClearSignal Models & Evaluation pipeline in strict phase order:
`ridge.py` → `random_forest.py` → `bert_finetune.py` / `bert.ipynb` → `ensemble.py` → `evaluate.py` → `predict.py` → `tests/test_properties.py`.
Each phase produces artefacts consumed by the next. Test set is opened exactly once in `evaluate.py`.

## Tasks

- [x] 1. Fix and harden `src/models/ridge.py` (Phase 1)
  - [x] 1.1 Expand alpha candidate list and fix output paths
    - Ensure `ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]` (already correct)
    - Fix `OUTPUTS_DIR` to write `outputs/ridge_alpha_search.csv` and `outputs/ridge_val_preds.npy`
    - Fix `MODELS_DIR` to write `models/ridge_model.pkl` and `models/scaler.pkl`
    - Confirm `USE_DUMMY_DATA = False` path reads from `data/processed/`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7_

  - [ ]* 1.2 Write property test for best-alpha selection (Property 1)
    - **Property 1: Best-configuration selection minimises validation MAE**
    - **Validates: Requirements 1.2**

  - [ ]* 1.3 Write property test for prediction clipping (Property 2)
    - **Property 2: All predictions are clipped to [1.0, 5.0]**
    - **Validates: Requirements 1.6**

  - [ ]* 1.4 Write property test for scaler fit-on-train-only (Property 3)
    - **Property 3: Scaler is fit on training data only**
    - **Validates: Requirements 1.4, 2.5**

  - [x] 1.5 Fix and harden `run_ablation()` output schema
    - Ensure ablation DataFrame has exactly 5 rows: `all_features` + one per group A/B/C/D
    - Ensure all required columns present: `run`, `features_removed`, `n_features`, `mae`, `rmse`, `pearson_r`, `f1_binary`, `mae_delta`
    - Save to `outputs/ablation_results.csv` (not `reports/`)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 1.6 Write property test for ablation output schema (Property 4)
    - **Property 4: Ablation output has exactly five rows with required schema**
    - **Validates: Requirements 2.1, 2.3**

- [x] 2. Checkpoint — run `python src/models/ridge.py` and verify outputs
  - Confirm `models/ridge_model.pkl`, `models/scaler.pkl`, `outputs/ridge_alpha_search.csv`, `outputs/ridge_val_preds.npy`, `outputs/ablation_results.csv` all exist and are non-empty.
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Fix and harden `src/models/random_forest.py` (Phase 2)
  - [x] 3.1 Expand hyperparameter grid to ≥9 combos and fix output paths
    - Update `PARAM_GRID` to cover `n_estimators` ∈ {100, 200, 500}, `max_depth` ∈ {None, 10, 20}, `min_samples_leaf` ∈ {1, 5, 10} — at least 9 combinations
    - Fix `FEATURES_DIR` to read from `data/processed/` (not `data/features/`)
    - Fix `MODELS_DIR` to write `models/rf_model.pkl` (not `src/models/`)
    - Fix `OUTPUTS_DIR` to write `models_saved/rf_search.csv`, `models_saved/rf_val_preds.npy`, `models_saved/feature_importances.json`
    - Set `USE_DUMMY_DATA = False`
    - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6, 3.7_

  - [ ]* 3.2 Write property test for RF grid coverage (Property 5)
    - **Property 5: RF grid search covers at least nine hyperparameter combinations**
    - **Validates: Requirements 3.2**

  - [x] 3.3 Add mean-compression warning and feature importances sort
    - After best model selected, compare `best_preds.std()` vs `y_val.std()` and log `[WARN]` if ratio < 0.5
    - Ensure `feature_importances.json` has both `importances` (dict) and `sorted` (descending list of [name, value] pairs)
    - _Requirements: 3.6, 3.8, 8.3_

  - [ ]* 3.4 Write property test for feature importances ordering (Property 6)
    - **Property 6: Feature importances sorted list is in descending order**
    - **Validates: Requirements 3.6, 8.3**

- [x] 4. Checkpoint — run `python src/models/random_forest.py` and verify outputs
  - Confirm `models/rf_model.pkl`, `models_saved/feature_importances.json`, `models_saved/rf_search.csv`, `models_saved/rf_val_preds.npy` all exist and are non-empty.
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Create `src/models/bert_finetune.py` and `notebooks/bert.ipynb` (Phase 3)
  - [x] 5.1 Write `src/models/bert_finetune.py` as a self-contained Colab-ready script
    - Import `distilbert-base-uncased` with a single-output regression head (`num_labels=1`)
    - Implement `build_model()` returning `DistilBertForSequenceClassification`
    - Implement `train(train_df, val_df, epochs=3, lr=2e-5, batch_size=8)` with early stopping after 1 non-improving epoch
    - Tokenize with `truncation=True, max_length=512`
    - Use `torch.no_grad()` for validation inference
    - Save fine-tuned weights to `models_saved/bert_weights/`
    - Save validation predictions to `models_saved/bert_val_preds.npy`
    - Log `[FINDING]` if BERT val MAE > Ridge val MAE
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 5.2 Write property test for BERT token truncation (Property 7)
    - **Property 7: DistilBERT tokenizer truncates inputs to at most 512 tokens**
    - **Validates: Requirements 4.2**

  - [x] 5.3 Convert `bert_finetune.py` to `notebooks/bert.ipynb`
    - Mirror all logic from `bert_finetune.py` as notebook cells
    - Add Colab GPU setup cell (`!pip install transformers torch`) at top
    - Add Google Drive mount cell for saving `bert_weights/` and `bert_val_preds.npy`
    - Ensure notebook is self-contained and runnable end-to-end on Colab
    - _Requirements: 4.1–4.6_

- [x] 6. Fix and harden `src/models/ensemble.py` (Phase 4)
  - [x] 6.1 Fix artefact paths and expand weight candidates
    - Fix `load_val_preds()` to read from `reports/` for ridge/rf preds (or `outputs/` — align with ridge.py output)
    - Ensure at least 4 weight configurations are evaluated including a BERT-absent fallback
    - Ensure `pick_best_weights()` selects the config with strictly lowest val MAE
    - Save `reports/ensemble_weights.json` with keys `ridge`, `rf`, `bert` summing to 1.0
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 8.2_

  - [ ]* 6.2 Write property test for ensemble weights sum (Property 8)
    - **Property 8: Ensemble weights sum to 1.0**
    - **Validates: Requirements 5.1, 5.5, 8.2**

- [x] 7. Checkpoint — run `python src/models/ensemble.py` and verify outputs
  - Confirm `reports/ensemble_weights.json` exists, is non-empty, keys sum to 1.0.
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Fix and harden `src/evaluation/evaluate.py` (Phase 5)
  - [x] 8.1 Ensure test data is loaded exactly once and all metrics are computed
    - Confirm `get_test_data()` is the only place `test_features.csv` is read
    - Compute MAE, RMSE, Pearson_r, F1_Binary for Ridge, RF, DistilBERT, and Ensemble
    - Log `[FINDING]` if BERT test MAE > Ridge test MAE
    - Log ensemble prediction range and std vs true std
    - _Requirements: 6.1, 6.2, 6.6, 6.7_

  - [x] 8.2 Fix output paths for all three report artefacts
    - Save `reports/test_metrics_table.csv` with columns `Model`, `MAE`, `RMSE`, `Pearson r`, `F1 (>=3.0)`
    - Save `reports/calibration_data.json` with keys `y_true`, `ensemble`, `ridge`, `rf`, `dataset_mean`
    - Save `reports/ablation_test.csv` from test-set ablation re-run
    - _Requirements: 6.3, 6.4, 6.5, 8.4, 8.5_

  - [ ]* 8.3 Write property test for evaluation metrics schema completeness (Property 10)
    - **Property 10: Evaluation metrics schema is complete for all models**
    - **Validates: Requirements 6.2**

  - [ ]* 8.4 Write property test for artefact non-empty after phase completion (Property 14)
    - **Property 14: All required artefacts are non-empty files after their phase completes**
    - **Validates: Requirements 8.1, 8.4, 8.5, 8.6**

- [x] 9. Fix and harden `src/predict.py` (Phase 6)
  - [x] 9.1 Fix artefact loading paths and NaN handling
    - Update `MODELS_DIR` to load `models/ridge_model.pkl`, `models/rf_model.pkl`, `models/scaler.pkl`
    - Update `REPORTS_DIR` to load `reports/ensemble_weights.json`
    - Ensure `np.nan_to_num(X_raw, nan=0.0)` is applied before any model call
    - _Requirements: 7.3, 7.9_

  - [x] 9.2 Implement BERT-absent fallback with renormalised weights
    - When `bert_weights/` is absent or BERT load fails, renormalise `ridge` and `rf` weights to sum to 1.0
    - Ensure no exception is raised; log `[WARN]`
    - _Requirements: 7.6, 5.5_

  - [x] 9.3 Implement correct confidence interval formula
    - CI = `[clip(csat_score - std, 1.0, 5.0), clip(csat_score + std, 1.0, 5.0)]`
    - `std` = standard deviation of individual model predictions (not ensemble)
    - _Requirements: 7.7_

  - [x] 9.4 Verify `_compute_emotional_arc()` uses VADER across turns
    - Confirm arc is computed from per-turn VADER compound scores
    - Returns one of `"rise"`, `"fall"`, `"flat"`, `"v_shape"`
    - _Requirements: 7.8_

  - [ ]* 9.5 Write property test for `predict()` output schema (Property 11)
    - **Property 11: `predict()` output schema is always correct**
    - **Validates: Requirements 7.1, 7.2**

  - [ ]* 9.6 Write property test for BERT-absent fallback (Property 9)
    - **Property 9: BERT-absent fallback produces valid output without raising**
    - **Validates: Requirements 5.5, 7.6**

  - [ ]* 9.7 Write property test for confidence interval formula (Property 12)
    - **Property 12: Confidence interval formula is correct**
    - **Validates: Requirements 7.7**

  - [ ]* 9.8 Write property test for NaN replacement (Property 13)
    - **Property 13: NaN feature values are replaced with 0.0**
    - **Validates: Requirements 7.9**

- [x] 10. Checkpoint — run `python src/predict.py` on 5 dummy calls and verify timing
  - All 5 calls must complete in < 5 seconds each on CPU.
  - Output dict must contain `csat_score`, `confidence_interval`, `emotional_arc`, `shap_values`.
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Write `tests/test_properties.py` — Hypothesis property-based tests
  - [x] 11.1 Scaffold test file with Hypothesis imports and shared fixtures
    - Import `hypothesis`, `hypothesis.strategies as st`, `numpy`, `pickle`, `json`
    - Add `@settings(max_examples=100)` to all property tests
    - Tag each test with `# Feature: person2-models-evaluation, Property N: ...`
    - _Requirements: all_

  - [x] 11.2 Implement P1 — best-config selection minimises MAE
    - Generate random list of MAE floats; assert selected index == `argmin`
    - _Requirements: 1.2, 3.3, 5.2_

  - [x] 11.3 Implement P2 — prediction clipping
    - Generate random floats in [-10, 10]; assert `np.clip(x, 1, 5)` always in [1.0, 5.0]
    - _Requirements: 1.6, 3.4, 5.3_

  - [x] 11.4 Implement P3 — scaler fit on train only
    - Generate random train/val splits; assert scaler fitted on train has different `mean_` than one fitted on val
    - _Requirements: 1.4, 2.5_

  - [x] 11.5 Implement P4 — ablation schema
    - Run `run_ablation()` on synthetic data; assert exactly 5 rows and all required columns
    - _Requirements: 2.1, 2.3_

  - [x] 11.6 Implement P5 — RF grid coverage
    - Assert `len(PARAM_GRID) >= 9` and all required `n_estimators`, `max_depth`, `min_samples_leaf` values present
    - _Requirements: 3.2_

  - [x] 11.7 Implement P6 — feature importances ordering
    - Generate random importance dicts; assert `sorted` output is non-increasing
    - _Requirements: 3.6, 8.3_

  - [x] 11.8 Implement P7 — BERT token truncation
    - Generate random strings of varying length; assert tokenized `input_ids` length <= 512
    - _Requirements: 4.2_

  - [x] 11.9 Implement P8 — ensemble weights sum to 1.0
    - Generate random positive weight triples; after normalisation assert sum == 1.0 within 1e-6
    - _Requirements: 5.1, 5.5, 8.2_

  - [x] 11.10 Implement P9 — BERT-absent fallback
    - Call `predict()` with `bert_weights/` absent; assert no exception and valid schema
    - _Requirements: 5.5, 7.6_

  - [x] 11.11 Implement P10 — metrics schema completeness
    - Generate random pred/true arrays; call `evaluate()`; assert all four metrics present
    - _Requirements: 6.2_

  - [x] 11.12 Implement P11 — `predict()` output schema
    - Generate random transcript strings and metadata dicts; assert output has correct keys and types
    - _Requirements: 7.1, 7.2_

  - [x] 11.13 Implement P12 — CI formula
    - Generate random model prediction lists; assert CI == `[clip(mean-std,1,5), clip(mean+std,1,5)]`
    - _Requirements: 7.7_

  - [x] 11.14 Implement P13 — NaN replacement
    - Generate feature vectors with random NaN positions; assert `np.nan_to_num(v, nan=0.0)` has no NaN
    - _Requirements: 7.9_

  - [x] 11.15 Implement P14 — artefact non-empty
    - After running each phase runner function, assert all expected output paths exist and `os.path.getsize > 0`
    - _Requirements: 8.1, 8.4, 8.5, 8.6_

- [x] 12. Final checkpoint — full pipeline smoke test
  - Run `python src/models/ridge.py`, `python src/models/random_forest.py`, `python src/models/ensemble.py`, `python src/evaluation/evaluate.py`, `python src/predict.py` in order.
  - Run `python -m pytest tests/test_properties.py --tb=short` and confirm all non-optional tests pass.
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at each phase boundary
- Property tests validate universal correctness guarantees; unit tests validate specific examples
- BERT fine-tuning (task 5) must run on Colab GPU — `bert_finetune.py` and `bert.ipynb` are both provided so either can be used
- Test set (`test_features.csv`) must only be opened inside `src/evaluation/evaluate.py` — this constraint is enforced by convention across all phase files
