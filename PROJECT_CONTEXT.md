# ClearSignal — Person 2 (Models & Evaluation) — Full Project Context

Use this document to continue work with any AI assistant. It contains everything needed to understand the project, what has been built, current results, known issues, and what remains.

---

## Project Overview

**ClearSignal** is a CSAT (Customer Satisfaction Score) prediction system for call centre transcripts. It is a 4-person group project:

- **Person 1** — Data & Features (upstream, mostly done)
- **Person 2** — Models & Evaluation (this codebase — fully implemented)
- **Person 3** — Backend & Explainability (FastAPI server, SHAP)
- **Person 4** — Dashboard (Streamlit)

The task is to predict a continuous CSAT score (1.0–5.0) from call transcripts and metadata. This is a regression problem.

**Dataset:** 2,500 synthetic call centre transcripts (`data/raw/synthetic_calls_v3_final.csv`)
**Split:** 70/15/15 → train=1750, val=375, test=375 (stratified on csat_range × issue_type, random_state=42)

---

## Repository Structure

```
conv_analytics/
├── data/
│   ├── raw/synthetic_calls_v3_final.csv
│   └── processed/
│       ├── train_features.csv      # 1750 rows, 24 features + metadata
│       ├── val_features.csv        # 375 rows
│       └── test_features.csv       # 375 rows
├── models/
│   ├── ridge_model.pkl             # trained Ridge (alpha=100)
│   ├── rf_model.pkl                # trained RF (n=100, depth=10, leaf=1)
│   └── scaler.pkl                  # StandardScaler fitted on train only
├── models_saved/
│   ├── bert_weights/               # fine-tuned DistilBERT weights
│   ├── bert_val_preds.npy          # BERT val predictions
│   ├── bert_test_preds.npy         # BERT test predictions
│   ├── feature_importances.json    # RF importances → Person 3 + 4
│   ├── rf_search.csv               # RF grid search log
│   └── rf_val_preds.npy
├── outputs/
│   ├── ridge_alpha_search.csv
│   ├── ridge_val_preds.npy
│   ├── ablation_results.csv        # val ablation
│   └── figures/                    # 8 visualisation PNGs
├── reports/
│   ├── ensemble_weights.json       # {"ridge":0.6,"rf":0.4,"bert":0.0}
│   ├── test_metrics_table.csv      # final results → Person 4
│   ├── calibration_data.json       # predicted vs actual → Person 4
│   ├── ablation_test.csv           # test ablation
│   └── ensemble_val_preds.npy
├── src/
│   ├── models/
│   │   ├── ridge.py                # Phase 1 — alpha search + ablation
│   │   ├── random_forest.py        # Phase 2 — grid search + importances
│   │   ├── bert_finetune.py        # Phase 3 — Colab fine-tuning script
│   │   └── ensemble.py             # Phase 4 — weight selection
│   ├── evaluation/
│   │   └── evaluate.py             # Phase 5 — test set eval (open once)
│   ├── features/
│   │   ├── preprocessing.py        # Person 1's feature functions
│   │   └── recompute_features.py   # Our fix for zeroed NLP features
│   └── predict.py                  # Phase 6 — public predict() function
├── notebooks/
│   ├── bert.ipynb                  # Colab-ready BERT fine-tuning notebook
│   └── phase0_skeleton.py          # FEATURE_COLUMNS, evaluate(), metrics_table()
└── tests/
    └── test_properties.py          # Hypothesis property-based tests (18 pass)
```

---

## Feature Schema (22 features)

Defined in `notebooks/phase0_skeleton.py` as `FEATURE_COLUMNS`.

| Group | Features |
|---|---|
| A — Sentiment (3) | mean_sentiment, last_20_sentiment, std_sentiment |
| B — Structure (5) | talk_time_ratio, avg_agent_words, avg_customer_words, interruption_count, resolution_flag |
| C — Agent (3) | empathy_density, apology_count, transfer_count |
| D — Metadata (13) | duration_ordinal, duration_deviation, repeat_contact, intent_billing, intent_technical, intent_account, intent_payment, intent_network, intent_delivery, intent_refund, intent_complaint, intent_subscription, intent_login |

**Important:** Person 1's `apply_features.py` had a bug — it read from a `transcript` column but the CSVs have `transcript_text`. This meant Groups A, B, C were all zeros and duration_ordinal/deviation were NaN. We detected this and fixed it in `src/features/recompute_features.py`, which recomputes all NLP features from the transcript_text column using Person 1's own preprocessing functions.

---

## Pipeline Execution Order

```bash
# Fix features (already done — only needed once)
python src/features/recompute_features.py

# Phase 1 — Ridge
python src/models/ridge.py

# Phase 2 — Random Forest
python src/models/random_forest.py

# Phase 3 — BERT (run notebooks/bert.ipynb on Colab GPU)
# Upload: train_features.csv, val_features.csv, test_features.csv
# Download: bert_weights.zip, bert_val_preds.npy, bert_test_preds.npy
# Place: models_saved/bert_weights/, models_saved/bert_val_preds.npy, models_saved/bert_test_preds.npy
# Then set BERT_READY = True in ensemble.py and evaluate.py

# Phase 4 — Ensemble
python src/models/ensemble.py

# Phase 5 — Final evaluation (test set, once only)
python src/evaluation/evaluate.py

# Phase 6 — Verify predict()
python src/predict.py

# Visualisations
python outputs/visualisations.py
```

---

## Final Test Set Results (n=375, held-out)

| Model | MAE | RMSE | Pearson r | F1 (≥3.0) |
|---|---|---|---|---|
| Naive (always mean) | 1.1340 | 1.3567 | — | 0.7981 |
| Ridge | **1.1276** | **1.3189** | **0.2418** | 0.6790 |
| Random Forest | 1.1484 | 1.3406 | 0.1763 | 0.6456 |
| DistilBERT | 1.1468 | 1.3690 | -0.0268 | 0.0000 |
| Ensemble | 1.1336 | 1.3219 | 0.2293 | 0.6748 |

**Ensemble weights:** ridge=0.6, rf=0.4, bert=0.0 (BERT excluded — see findings)

---

## Key Findings (genuine, document in report)

**1. Models barely beat naive baseline on MAE**
The "always predict the training mean (3.09)" baseline gets MAE=1.134. Ridge gets 1.128. The improvement is real but tiny. Pearson r=0.24 is the more honest signal — the model is learning something but the effect is weak. This is a data quality issue, not a modelling failure.

**2. DistilBERT F1=0.000**
BERT never predicts below 3.0 — all predictions land in the "satisfied" bucket. Pearson r=-0.027 (essentially random). Synthetic transcripts like "sample billing_error conversation turn 3 uh" give it no real language signal. This is the expected finding from the spec. BERT weight in ensemble is 0.0.

**3. Mean compression across all models**
True CSAT std=1.357, ensemble std=0.270. Predictions cluster in [2.44, 3.61] vs true range [1.0, 5.0]. Models cannot distinguish extreme scores. Direct consequence of synthetic data.

**4. Only Group D (metadata) contributes meaningfully**
Ablation on test set: removing metadata costs +0.013 MAE. Removing Groups A/B/C costs ≤0.002. After fixing the zeroed features, sentiment features now have real values but still contribute minimally — synthetic text produces near-identical VADER scores across all calls.

**5. Feature importances (RF) after fix**
- std_sentiment: 13.1%
- mean_sentiment: 12.4%
- last_20_sentiment: 12.1%
- avg_agent_words: 11.0%
- avg_customer_words: 10.5%
- empathy_density: 8.9%
- repeat_contact: 5.2% (was 61% before fix — that was the broken data)

**6. Ridge is genuinely best**
Not a coincidence. With real features, RF overfits on noisy synthetic sentiment scores. Ridge's L2 regularisation handles the noise better. Best alpha=100 (high regularisation).

---

## Ablation Results

### Validation set
| Run | ΔMAE |
|---|---|
| Baseline (all features) | 0.0000 |
| Remove Group A (Sentiment) | -0.0023 |
| Remove Group B (Structure) | -0.0031 |
| Remove Group C (Agent) | -0.0003 |
| Remove Group D (Metadata) | +0.0034 |

### Test set
| Run | ΔMAE |
|---|---|
| Baseline (all features) | 0.0000 |
| Remove Group A (Sentiment) | -0.0020 |
| Remove Group B (Structure) | -0.0022 |
| Remove Group C (Agent) | +0.0024 |
| Remove Group D (Metadata) | +0.0125 |

Note: negative delta means removing the group *improved* MAE — those features are adding noise.

---

## predict() Contract

```python
# src/predict.py
predict(transcript_text: str, call_metadata: dict) -> dict

# call_metadata keys:
#   issue_type: str (e.g. "billing", "technical", "account")
#   call_duration: str ("short" | "medium" | "long")
#   repeat_contact: int (0 or 1)

# Returns:
{
    "csat_score": float,           # clipped [1.0, 5.0]
    "confidence_interval": [float, float],  # [csat-std, csat+std], clipped
    "emotional_arc": str,          # "rise" | "fall" | "flat" | "v_shape"
    "shap_values": {}              # stub — Person 3 fills this
}
```

**Behaviour:** Loads all artefacts at import time. If bert_weights/ absent, falls back to Ridge+RF with renormalised weights. Uses Person 1's preprocessing functions directly for feature extraction. All 5 test calls complete under 1 second on CPU.

---

## Artefact Handoff

### Person 3 (Backend/FastAPI) needs:
- `models/ridge_model.pkl`
- `models/rf_model.pkl`
- `models/scaler.pkl`
- `reports/ensemble_weights.json` — `{"ridge": 0.6, "rf": 0.4, "bert": 0.0}`
- `src/predict.py` — call `predict(transcript_text, call_metadata)`
- `models_saved/bert_weights/` — for BERT inference

### Person 4 (Dashboard) needs:
- `reports/test_metrics_table.csv` — 5 rows (naive + 4 models), columns: Model, MAE, RMSE, Pearson r, F1 (≥3.0)
- `reports/calibration_data.json` — keys: y_true, ensemble, ridge, rf, dataset_mean
- `models_saved/feature_importances.json` — keys: importances (dict), sorted (list of [name, value])

---

## Known Issues / Limitations

1. **BERT val preds std=0.004** — BERT converged to predicting ~2.91 for everything. Weights are valid but the model learned nothing useful. This is expected and documented.

2. **predict() scores cluster around 3.0** — mean compression. The models genuinely cannot distinguish CSAT=1 from CSAT=5 on this synthetic data. This is a data limitation, not a code bug.

3. **duration_ordinal computed from call_duration_seconds** — Person 1 left this as NaN. We compute it as: ≤240s → -1 (short), ≤420s → 0 (medium), >420s → 1 (long).

4. **BERT not in ensemble** — weight=0.0 because BERT's val MAE (1.145) is worse than Ridge (1.132) and adding it hurts the ensemble. If BERT is retrained on real data, this would change.

5. **torch/transformers version** — requires torch≥2.4 and transformers≥4.36. Local env has torch 2.11 + transformers 5.4 which works. The `c10.dll` error on Windows is resolved by importing torch before other heavy imports (done in predict.py).

---

## What Remains / Possible Improvements

- **Cross-validation** — 5-fold CV on training data would give confidence intervals on MAE/r
- **Bootstrap CI on test metrics** — error bars on the final table
- **Error analysis** — which calls does the model get most wrong? Segment by issue_type, repeat_contact
- **Stacking ensemble** — train a meta-learner on val predictions instead of fixed weights
- **Better BERT fine-tuning** — longer training, learning rate warmup, on real (non-synthetic) data
- **Reframe primary metric** — Pearson r and F1 are more meaningful than MAE for this dataset given the naive baseline issue

---

## Tests

```bash
python -m pytest tests/test_properties.py -v --tb=short
# 18 passed, 2 skipped (BERT-only, expected)
```

14 correctness properties tested with Hypothesis (100 iterations each):
P1 best-config selection, P2 prediction clipping, P3 scaler fit-on-train-only, P4 ablation schema, P5 RF grid coverage, P6 feature importances ordering, P7 BERT truncation (skipped — needs weights), P8 ensemble weights sum to 1.0, P9 BERT-absent fallback, P10 metrics schema, P11 predict() output schema, P12 CI formula, P13 NaN replacement, P14 artefact non-empty.

---

## Visualisations

All saved to `outputs/figures/`:
1. `1_ridge_alpha_search.png` — MAE and Pearson r vs alpha
2. `2_ablation_study.png` — feature group contribution (val + test)
3. `3_model_comparison.png` — all 4 metrics across all models
4. `4_calibration_scatter.png` — predicted vs actual (Ridge, RF, Ensemble)
5. `5_prediction_distributions.png` — mean compression visible
6. `6_feature_importances.png` — all 22 features colour-coded by group
7. `7_residuals.png` — residuals vs predicted + distribution
8. `8_metrics_table.png` — clean table for slides/report

---

## Environment

- Python 3.10
- torch 2.11.0+cpu
- transformers 5.4.0
- scikit-learn, pandas, numpy, scipy, matplotlib, hypothesis, vaderSentiment
- Windows (paths use backslash but Path() handles this)
- Project root: `conv_analytics/`
