import shap
import joblib
import numpy as np
import os

# Load model
MODEL_PATH = "models/rf_model.pkl"

rf_model = None
explainer = None


def load_model():
    global rf_model, explainer

    if rf_model is None:
        rf_model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(rf_model)


def compute_shap(feature_vector):

    load_model()

    shap_values = explainer.shap_values(
    feature_vector,
    check_additivity=False
)

    return shap_values

FEATURE_MAPPING = {
    "repeat_contact": "Customer had called before",
    "transfer_count": "Number of times transferred",
    "empathy_density": "Agent empathy frequency",
    "final_sentiment": "How call ended emotionally",
    "call_duration": "Call length deviation",
    "issue_type": "Issue category"
}

def get_top_features(shap_values, feature_names):

    feature_contributions = dict(
        zip(feature_names, shap_values)
    )

    # sort by magnitude
    sorted_features = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_5 = sorted_features[:5]

    return top_5