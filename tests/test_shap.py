import numpy as np
import pandas as pd
from src.explainability.shap_explainer import compute_shap

df = pd.read_csv("data/processed/train_features.csv")

df = df.select_dtypes(include=["number", "bool"])

test = df.iloc[[0]].values

print(compute_shap(test))