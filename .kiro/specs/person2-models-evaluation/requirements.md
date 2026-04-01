# Requirements Document

## Introduction

Person 2 is responsible for the Models & Evaluation component of ClearSignal, a CSAT score prediction system for call centre transcripts. This component trains four models (Ridge Regression, Random Forest, DistilBERT, and a weighted Ensemble), evaluates them on held-out test data, and exports a `predict()` function that downstream consumers (Person 3 — Backend & Explainability, Person 4 — Dashboard) depend on.

Inputs come from Person 1's feature pipeline (`train_features.csv`, `val_features.csv`, `test_features.csv`, `scaler.pkl`). Outputs are `rf_model.pkl`, `ensemble_weights.json`, `feature_importances.json`, a metrics table, and `src/predict.py`.

---

## Glossary

- **ClearSignal**: The CSAT score prediction system for call centre transcripts.
- **CSAT_Score**: A continuous satisfaction score in the range [1.0, 5.0] assigned to a call.
- **Feature_Pipeline**: Person 1's preprocessing and feature extraction code that produces the feature CSVs.
- **Ridge_Model**: A Ridge Regression model trained on scaled feature vectors to predict CSAT_Score.
- **RF_Model**: A Random Forest Regressor trained on unscaled feature vectors to predict CSAT_Score.
- **DistilBERT_Model**: A fine-tuned `distilbert-base-uncased` model with a regression head that predicts CSAT_Score from raw transcript text.
- **Ensemble**: A weighted linear combination of Ridge_Model, RF_Model, and DistilBERT_Model predictions.
- **Scaler**: The `StandardScaler` fitted on training data and saved as `scaler.pkl`; required by Ridge_Model and the predict function.
- **Feature_Vector**: A 22-element float array in the column order defined by `FEATURE_COLUMNS` in `phase0_skeleton.py`.
- **Ablation_Study**: A series of Ridge_Model training runs where one feature group (A, B, C, or D) is removed per run to measure each group's contribution to validation MAE.
- **Feature_Group_A**: Sentiment features — `mean_sentiment`, `last_20_sentiment`, `std_sentiment`.
- **Feature_Group_B**: Conversational structure features — `talk_time_ratio`, `avg_agent_words`, `avg_customer_words`, `interruption_count`, `resolution_flag`.
- **Feature_Group_C**: Agent behaviour features — `empathy_density`, `apology_count`, `transfer_count`.
- **Feature_Group_D**: Metadata features — `duration_ordinal`, `duration_deviation`, `repeat_contact`, and all ten `intent_*` one-hot columns.
- **Val_Set**: The validation split (`val_features.csv`) used for hyperparameter selection and ablation. Never used for final reporting.
- **Test_Set**: The held-out test split (`test_features.csv`) opened exactly once during Phase 5 evaluation.
- **Predict_Function**: The `predict(transcript_text, call_metadata)` function in `src/predict.py` consumed by Person 3's FastAPI endpoint.
- **Confidence_Interval**: A [lower, upper] pair derived from the standard deviation across per-model predictions, clipped to [1.0, 5.0].
- **Emotional_Arc**: A string label (`rise`, `fall`, `flat`, `v_shape`) describing the sentiment trajectory of a call transcript.
- **MAE**: Mean Absolute Error — primary regression metric.
- **RMSE**: Root Mean Squared Error.
- **Pearson_r**: Pearson correlation coefficient between predicted and true CSAT scores.
- **F1_Binary**: Binary F1 score using a classification threshold of 3.0 (satisfied = CSAT ≥ 3.0).

---

## Requirements

### Requirement 1: Ridge Regression Baseline Training

**User Story:** As Person 2, I want to train a Ridge Regression baseline on the feature CSVs from Person 1, so that I have a well-tuned linear model to compare against more complex approaches.

#### Acceptance Criteria

1. WHEN `train_features.csv` and `val_features.csv` are present in `data/processed/`, THE Ridge_Model SHALL be trained using `FEATURE_COLUMNS` from `phase0_skeleton.py` as the exact feature set.
2. THE Ridge_Model SHALL be trained with each of the five alpha candidates (0.01, 0.1, 1.0, 10.0, 100.0) and the candidate with the lowest validation MAE SHALL be selected as `best_alpha`.
3. WHEN alpha search completes, THE Ridge_Model SHALL report validation MAE and Pearson_r for each alpha candidate and save the results to `outputs/ridge_alpha_search.csv`.
4. THE Ridge_Model SHALL apply the Scaler (fitted on training data only) to both training and validation Feature_Vectors before fitting and predicting.
5. WHEN the best Ridge_Model is trained, THE Ridge_Model SHALL save the fitted model to `models/ridge_model.pkl` and the fitted Scaler to `models/scaler.pkl`.
6. THE Ridge_Model SHALL clip all predictions to [1.0, 5.0] before computing any metric.
7. WHEN training completes, THE Ridge_Model SHALL save validation predictions to `outputs/ridge_val_preds.npy` for use in Phase 4 ensemble weight selection.

---

### Requirement 2: Ridge Regression Ablation Study

**User Story:** As Person 2, I want to run a structured ablation study on Ridge Regression, so that I can quantify the contribution of each feature group and include the findings in the final report.

#### Acceptance Criteria

1. THE Ablation_Study SHALL consist of exactly five Ridge_Model training runs: one baseline run using all features, and one run for each of Feature_Group_A, Feature_Group_B, Feature_Group_C, and Feature_Group_D where that group's columns are excluded.
2. WHEN a feature group is removed, THE Ablation_Study SHALL use the same `best_alpha` selected in Requirement 1 for all ablation runs.
3. THE Ablation_Study SHALL record, for each run: run name, features removed, number of remaining features, validation MAE, RMSE, Pearson_r, F1_Binary, and MAE delta relative to the baseline run.
4. WHEN all ablation runs complete, THE Ablation_Study SHALL save results to `outputs/ablation_results.csv`.
5. THE Ablation_Study SHALL use the Scaler fitted on the full training feature set for all ablation runs, without refitting the Scaler on the reduced feature sets.

---

### Requirement 3: Random Forest Regressor Training

**User Story:** As Person 2, I want to train a Random Forest Regressor with a structured hyperparameter search, so that I have a non-linear model that can capture feature interactions Ridge cannot.

#### Acceptance Criteria

1. WHEN `train_features.csv` and `val_features.csv` are present, THE RF_Model SHALL be trained on unscaled Feature_Vectors (no Scaler applied).
2. THE RF_Model SHALL be evaluated across a grid of at least nine hyperparameter combinations covering `n_estimators` ∈ {100, 200, 500}, `max_depth` ∈ {None, 10, 20}, and `min_samples_leaf` ∈ {1, 5, 10}.
3. WHEN the grid search completes, THE RF_Model SHALL select the combination with the lowest validation MAE as the best configuration and save the search log to `models_saved/rf_search.csv`.
4. THE RF_Model SHALL clip all predictions to [1.0, 5.0] before computing any metric.
5. WHEN the best RF_Model is trained, THE RF_Model SHALL save the fitted model to `models/rf_model.pkl`.
6. WHEN the best RF_Model is trained, THE RF_Model SHALL extract `feature_importances_` from the fitted model, sort them in descending order, and save them as `models_saved/feature_importances.json` with both an unsorted dict and a sorted list.
7. WHEN the best RF_Model is trained, THE RF_Model SHALL save validation predictions to `models_saved/rf_val_preds.npy` for use in Phase 4 ensemble weight selection.
8. WHEN RF_Model validation predictions are generated, THE RF_Model SHALL compare prediction standard deviation against true CSAT standard deviation and log a warning if the prediction standard deviation is less than 50% of the true standard deviation.

---

### Requirement 4: DistilBERT Fine-Tuning

**User Story:** As Person 2, I want to fine-tune a DistilBERT model on raw transcript text, so that I can assess whether a language model adds predictive value over structured features.

#### Acceptance Criteria

1. THE DistilBERT_Model SHALL use `distilbert-base-uncased` from HuggingFace with a single-output regression head replacing the classification head.
2. WHEN a transcript exceeds 512 tokens, THE DistilBERT_Model SHALL truncate the input to 512 tokens.
3. THE DistilBERT_Model SHALL be fine-tuned with learning rate 2e-5, batch size 8, and between 2 and 3 epochs, with early stopping triggered when validation loss does not improve for one consecutive epoch.
4. WHEN fine-tuning completes, THE DistilBERT_Model SHALL save the fine-tuned weights to `models_saved/bert_weights/`.
5. WHEN fine-tuning completes, THE DistilBERT_Model SHALL save validation predictions to `models_saved/bert_val_preds.npy` for use in Phase 4 ensemble weight selection.
6. WHEN DistilBERT_Model validation MAE exceeds Ridge_Model validation MAE, THE DistilBERT_Model SHALL log a finding note stating that the underperformance is expected due to the synthetic nature of the transcript text and that this finding must be documented in the final report.

---

### Requirement 5: Ensemble Construction

**User Story:** As Person 2, I want to combine Ridge, RF, and DistilBERT predictions into a weighted ensemble, so that the final predictor benefits from the complementary strengths of each model.

#### Acceptance Criteria

1. THE Ensemble SHALL combine Ridge_Model, RF_Model, and DistilBERT_Model predictions using a weighted sum where the weights sum to 1.0.
2. WHEN selecting ensemble weights, THE Ensemble SHALL evaluate at least four weight configurations on the validation set and select the configuration with the lowest validation MAE.
3. THE Ensemble SHALL clip the final weighted sum to [1.0, 5.0].
4. WHEN weight selection completes, THE Ensemble SHALL save the selected weights to `reports/ensemble_weights.json` with keys `ridge`, `rf`, and `bert`.
5. IF DistilBERT_Model validation predictions are unavailable, THEN THE Ensemble SHALL fall back to a two-model combination of Ridge_Model and RF_Model with weights renormalised to sum to 1.0, and log a warning that BERT predictions are missing.

---

### Requirement 6: Final Test Set Evaluation

**User Story:** As Person 2, I want to evaluate all four models on the held-out test set exactly once, so that I can produce unbiased final metrics for the project report and Person 4's dashboard.

#### Acceptance Criteria

1. THE Evaluator SHALL open `test_features.csv` exactly once, in `src/evaluation/evaluate.py`, and SHALL NOT open it in any other phase file.
2. WHEN test set evaluation runs, THE Evaluator SHALL compute MAE, RMSE, Pearson_r, and F1_Binary for Ridge_Model, RF_Model, DistilBERT_Model, and Ensemble on the test set.
3. WHEN test set evaluation completes, THE Evaluator SHALL save the metrics table to `reports/test_metrics_table.csv` for Person 4's dashboard.
4. WHEN test set evaluation completes, THE Evaluator SHALL save predicted vs actual values for all four models to `reports/calibration_data.json` for Person 4's calibration scatter plot.
5. WHEN test set evaluation completes, THE Evaluator SHALL re-run the ablation study on the test set using the saved Ridge_Model and Scaler, and save results to `reports/ablation_test.csv`.
6. WHEN test set evaluation completes, THE Evaluator SHALL log the prediction range and standard deviation of the Ensemble on the test set alongside the true CSAT standard deviation.
7. IF DistilBERT_Model test MAE exceeds Ridge_Model test MAE, THEN THE Evaluator SHALL log a finding note confirming the expected underperformance of DistilBERT on synthetic transcripts.
8. WHEN test set evaluation completes, THE Evaluator SHALL NOT be re-run to adjust any model or weight — test results are final.

---

### Requirement 7: Predict Function

**User Story:** As Person 3, I want a `predict()` function in `src/predict.py` that accepts a raw transcript and call metadata and returns a CSAT prediction with supporting outputs, so that I can wrap it in a FastAPI endpoint without needing to understand the model internals.

#### Acceptance Criteria

1. THE Predict_Function SHALL accept two arguments: `transcript_text` (str) and `call_metadata` (dict with keys `issue_type`, `call_duration`, `repeat_contact`) and return a dict.
2. THE Predict_Function SHALL return a dict containing exactly: `csat_score` (float, clipped to [1.0, 5.0]), `confidence_interval` (list of two floats [lower, upper] clipped to [1.0, 5.0]), `emotional_arc` (str, one of `rise`, `fall`, `flat`, `v_shape`), and `shap_values` (dict, stub for Person 3 to populate).
3. THE Predict_Function SHALL load all model artefacts (ridge_model.pkl, rf_model.pkl, scaler.pkl, ensemble_weights.json) once at module import time and cache them for subsequent calls.
4. WHEN called on CPU hardware without GPU acceleration, THE Predict_Function SHALL complete within 5 seconds per call.
5. WHEN DistilBERT_Model inference runs, THE Predict_Function SHALL use `torch.no_grad()` to disable gradient computation.
6. WHEN DistilBERT_Model weights are unavailable, THE Predict_Function SHALL fall back to a two-model Ridge_Model and RF_Model ensemble with renormalised weights and SHALL NOT raise an exception.
7. THE Predict_Function SHALL compute the Confidence_Interval as [csat_score − std, csat_score + std] where std is the standard deviation of the individual model predictions, clipped to [1.0, 5.0].
8. THE Predict_Function SHALL compute the Emotional_Arc from the transcript text using VADER sentiment scores across turns.
9. WHEN `transcript_text` contains NaN feature values after extraction, THE Predict_Function SHALL replace them with 0.0 before passing the Feature_Vector to any model.

---

### Requirement 8: Artefact Handoff Contracts

**User Story:** As Person 3 and Person 4, I want clearly defined output artefacts with agreed schemas, so that I can build the API and dashboard without waiting for Person 2 to finish all phases.

#### Acceptance Criteria

1. THE RF_Model SHALL produce `models/rf_model.pkl` — a serialised `sklearn.ensemble.RandomForestRegressor` that accepts an unscaled Feature_Vector of shape (n, 22) and returns CSAT predictions.
2. THE Ensemble SHALL produce `reports/ensemble_weights.json` — a JSON object with exactly the keys `ridge`, `rf`, and `bert`, each mapping to a float, where the three values sum to 1.0.
3. THE RF_Model SHALL produce `models_saved/feature_importances.json` — a JSON object with key `importances` (dict mapping feature name to importance float) and key `sorted` (list of [feature_name, importance] pairs in descending order).
4. THE Evaluator SHALL produce `reports/test_metrics_table.csv` — a CSV with columns `Model`, `MAE`, `RMSE`, `Pearson r`, `F1 (≥3.0)` and one row per model (Ridge, Random Forest, DistilBERT, Ensemble).
5. THE Evaluator SHALL produce `reports/calibration_data.json` — a JSON object with keys `y_true`, `ensemble`, `ridge`, `rf`, and `dataset_mean`, each containing a list of floats of equal length.
6. FOR ALL artefacts listed in criteria 1–5, THE Artefact_Producer SHALL ensure the file exists and is non-empty before signalling handoff to Person 3 or Person 4.
