from fastapi import FastAPI, HTTPException
from src.api.shared_types import PredictRequest
import time
from src.explainability.shap_explainer import compute_shap

from src.phrase_mapping.mapper import map_phrases
from src.coaching.generator import generate_summary

import pandas as pd
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = FastAPI(
    title="Conversational Analytics API",
    version="1.0"
)


@app.post("/predict")
def predict(request: PredictRequest):

    start_time = time.time()

    # Error Handling
    if not request.transcript.strip():
        raise HTTPException(
            status_code=422,
            detail="Transcript cannot be empty"
        )

    # Mock Response
    response = {
        "csat_score": 3.8,
        "confidence_interval": [3.4, 4.1],
        "emotional_arc": "rise",

        "top_positive_phrases": [
            "I understand your concern",
            "Happy to help you today"
        ],

        "top_negative_phrases": [
            "Let me transfer you",
            "You mentioned calling before"
        ],

        "coaching_summary":
        "Agent demonstrated empathy but call involved transfers. "
        "Priority: reduce transfers.",

        "shap_features": {
            "repeat_contact": -0.32,
            "transfer_count": -0.21,
            "empathy_density": 0.25
        },

        "aggregate_stats": {}
    }
    features = extract_features(request)

    shap_values = compute_shap(features)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")

    return response


@app.get("/aggregate")
def aggregate():

    return {
        "avg_csat_by_issue": {
            "billing": 3.5,
            "technical": 3.8,
            "account": 4.1
        },
        "emotional_distribution": {
            "rise": 40,
            "fall": 25,
            "flat": 20,
            "v_shape": 15
        }
    }


class CallRequest(BaseModel):
    transcript: str
    @app.post("/analyze-call")
    def analyze_call(request: CallRequest):

        transcript = request.transcript

       
        features = extract_features(transcript)

        feature_vector = np.array([features])

        shap_values = compute_shap(feature_vector)

        # Convert SHAP to feature list
        shap_features = [
            ("repeat_contact", shap_values[0][0]),
            ("transfer_count", shap_values[0][1]),
            ("final_sentiment", shap_values[0][2]),
            ("empathy_density", shap_values[0][3])
        ]

        positive, negative = map_phrases(
            transcript,
            shap_features
        )

        summary = generate_summary(
            positive,
            negative
        )

        return {
            "summary": summary,
            "positive": positive,
            "negative": negative
        }