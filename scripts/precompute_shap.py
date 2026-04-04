import numpy as np
import pandas as pd
import joblib
import shap

# Load model
model = joblib.load("models/rf_model.pkl")

# Load data
df = pd.read_csv("data/processed/train_features.csv")

# Drop non-numeric columns
drop_cols = [
    "issue_type",
    "transcript_text",
    "csat_range",
    "resolution_status",
    "emotional_arc"
]

df = df.drop(columns=drop_cols)

# Optional: also drop target column
df = df.drop(columns=["csat_score"], errors="ignore")

# Convert to numpy
X_train = df.values

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Save
np.save("outputs/shap_training.npy", shap_values)
np.save("outputs/feature_names.npy", df.columns)

print("SHAP values saved successfully")