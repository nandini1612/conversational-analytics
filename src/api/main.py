"""
src/api/main.py  —  Person 3 authored, fixed by Person 4.

Bugs fixed:
  1. extract_features() was called but never defined/imported — replaced with
     process_transcript() from src.features.inference
  2. BaseModel was used inside CallRequest but never imported in this file
  3. The entire analyze_call endpoint was nested inside CallRequest class
     (wrong indentation) — FastAPI never registered it. Removed the
     duplicate endpoint; /predict already handles everything.
  4. /predict was calling compute_shap() but then discarding the result
     and returning hardcoded mock values — now returns real inference.

New additions:
  - All models loaded once at startup (not per-request)
  - sentiment_series added to response (used by dashboard emotion trajectory chart)
  - skip_bert query parameter to skip slow DistilBERT during demos
  - Proper error handling with informative 500 messages
"""

import json
import time
import sys
import os

import numpy as np
import joblib

from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.api.shared_types import PredictRequest
from src.features.inference import process_transcript
from src.explainability.shap_explainer import compute_shap
from src.phrase_mapping.mapper import map_phrases
from src.coaching.generator import generate_summary

app = FastAPI(title="Conversational Analytics API", version="1.0")

# ── Load everything once at startup (not per-request) ─────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

ridge_model = joblib.load(os.path.join(BASE_DIR, "models", "ridge_model.pkl"))
rf_model    = joblib.load(os.path.join(BASE_DIR, "models", "rf_model.pkl"))

with open(os.path.join(BASE_DIR, "outputs", "metrics", "ensemble_weights.json")) as f:
    ENSEMBLE_WEIGHTS = json.load(f)

with open(os.path.join(BASE_DIR, "outputs", "metrics", "aggregate_stats.json")) as f:
    AGGREGATE_STATS = json.load(f)

# Feature column names (for SHAP dict keys) — loaded from training CSV header
try:
    import pandas as pd
    _feat_df = pd.read_csv(
        os.path.join(BASE_DIR, "data", "processed", "train_features.csv"), nrows=0
    )
    FEATURE_NAMES = [c for c in _feat_df.columns if c != "csat_score"]
except Exception:
    FEATURE_NAMES = [
        "mean_sentiment", "last_20_sentiment", "sentiment_std",
        "talk_time_ratio", "avg_agent_words", "avg_customer_words",
        "interruption_count", "resolution_flag",
        "empathy_density", "apology_count", "transfer_count",
        "duration_ordinal", "duration_deviation", "repeat_contact",
    ]

# VADER for sentiment series (used by the dashboard emotion trajectory chart)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_sentiment_series(transcript: str) -> list:
    """Return per-turn VADER compound scores for the emotion trajectory chart."""
    if not VADER_AVAILABLE:
        return []
    try:
        from src.features.preprocessing import parse_transcript
        turns = parse_transcript(transcript)
        return [round(_vader.polarity_scores(t["text"])["compound"], 3) for t in turns]
    except Exception:
        return []


def derive_arc(series: list) -> str:
    """Derive emotional arc label from sentiment series."""
    if not series or len(series) < 2:
        return "flat"
    try:
        from src.features.extractor import compute_arc
        return compute_arc(series)
    except Exception:
        pass
    # Fallback: simple heuristic
    first = sum(series[:max(1, len(series)//5)]) / max(1, len(series)//5)
    last  = sum(series[-max(1, len(series)//5):]) / max(1, len(series)//5)
    mid   = min(series[len(series)//4: 3*len(series)//4]) if len(series) > 4 else first
    if last - first > 0.15:
        return "rise"
    elif first - last > 0.15:
        return "fall"
    elif (first - mid > 0.15) and (last - mid > 0.15):
        return "v_shape"
    else:
        return "flat"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict")
def predict(request: PredictRequest, skip_bert: bool = Query(False)):
    """
    Analyse a single call transcript and return CSAT prediction + coaching.

    Query param:
        skip_bert=true  — skip DistilBERT for faster inference during demos
                          (Ridge + RF ensemble only, still ~0.86 Pearson r)
    """
    start = time.time()

    if not request.transcript.strip():
        raise HTTPException(status_code=422, detail="Transcript cannot be empty")

    metadata = {
        "issue_type":    request.call_metadata.issue_type,
        "call_duration": request.call_metadata.call_duration,
        "repeat_contact": request.call_metadata.repeat_contact,
    }

    # 1. Feature extraction
    try:
        feature_vector = process_transcript(request.transcript, metadata)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {exc}")

    # 2. Model predictions
    ridge_pred = float(ridge_model.predict(feature_vector)[0])
    rf_pred    = float(rf_model.predict(feature_vector)[0])

    w_ridge = ENSEMBLE_WEIGHTS.get("ridge", 0.5)
    w_rf    = ENSEMBLE_WEIGHTS.get("rf",    0.5)

    if not skip_bert:
        try:
            from src.models.bert_finetune import predict_bert
            bert_pred = float(predict_bert(request.transcript))
            w_bert    = ENSEMBLE_WEIGHTS.get("bert", 0.2)
            total_w   = w_ridge + w_rf + w_bert
            csat      = (w_ridge * ridge_pred + w_rf * rf_pred + w_bert * bert_pred) / total_w
        except Exception:
            # BERT unavailable — fall back to Ridge + RF silently
            total_w = w_ridge + w_rf
            csat    = (w_ridge * ridge_pred + w_rf * rf_pred) / total_w
    else:
        total_w = w_ridge + w_rf
        csat    = (w_ridge * ridge_pred + w_rf * rf_pred) / total_w

    csat = float(np.clip(csat, 1.0, 5.0))

    # 3. Confidence interval: ±1 std of component model predictions
    std   = float(np.std([ridge_pred, rf_pred]))
    ci    = [round(max(1.0, csat - std), 2), round(min(5.0, csat + std), 2)]

    # 4. SHAP values (on RF model)
    try:
        raw_shap  = compute_shap(feature_vector)          # shape (1, n_features)
        shap_dict = {
            FEATURE_NAMES[i]: round(float(raw_shap[0][i]), 4)
            for i in range(min(len(FEATURE_NAMES), len(raw_shap[0])))
        }
        shap_list = list(shap_dict.items())               # [(name, value), ...]
    except Exception:
        shap_dict = {}
        shap_list = []

    # 5. Phrase mapping
    try:
        positive, negative = map_phrases(request.transcript, shap_list)
    except Exception:
        positive, negative = [], []

    # 6. Coaching summary
    try:
        coaching = generate_summary(positive, negative)
    except Exception:
        coaching = "Coaching summary unavailable."

    # 7. Sentiment series + emotional arc
    sentiment_series = get_sentiment_series(request.transcript)
    arc = derive_arc(sentiment_series)

    elapsed = time.time() - start
    print(f"[predict] {elapsed:.3f}s  CSAT={csat:.2f}  arc={arc}  skip_bert={skip_bert}")

    return {
        "csat_score":           round(csat, 2),
        "confidence_interval":  ci,
        "emotional_arc":        arc,
        "top_positive_phrases": positive,
        "top_negative_phrases": negative,
        "coaching_summary":     coaching,
        "shap_features":        shap_dict,
        "sentiment_series":     sentiment_series,
        "inference_time":       round(elapsed, 3),
    }


@app.get("/aggregate")
def aggregate():
    """Return pre-computed aggregate statistics across the full training set."""
    return AGGREGATE_STATS