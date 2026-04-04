from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from src.explainability.shap_explainer import compute_shap
from src.phrase_mapping.mapper import map_phrases
from src.coaching.generator import generate_summary
from src.features.extractor import extract_features


app = FastAPI()


# ✅ Define this BEFORE endpoints
class CallRequest(BaseModel):
    transcript: str


@app.get("/")
def home():
    return {"message": "Conversational Analytics API"}


@app.post("/analyze-call")
def analyze_call(request: CallRequest):

    transcript = request.transcript

    features = extract_features(transcript)
    feature_vector = np.array([features])

    shap_values = compute_shap(feature_vector)

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