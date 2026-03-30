# scale_features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Files
train_file = "../data/processed/train_features.csv"
val_file = "../data/processed/val_features.csv"
test_file = "../data/processed/test_features.csv"

# 1️⃣ Read the CSVs
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

# 2️⃣ Select feature columns only (exclude IDs, transcript, issue_type, call_duration)
feature_cols = [col for col in train_df.columns if col not in ["call_id", "transcript", "issue_type", "call_duration"]]

X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values
X_test = test_df[feature_cols].values

# 3️⃣ Fit StandardScaler on training features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4️⃣ Transform val and test using the same scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Save scaled CSVs (overwrite originals or create new)
pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv(train_file, index=False)
pd.DataFrame(X_val_scaled, columns=feature_cols).to_csv(val_file, index=False)
pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv(test_file, index=False)

print("Saved scaled train, val, test CSVs.")

# 6️⃣ Save scaler for inference
joblib.dump(scaler, "../data/processed/scaler.pkl")
print("Saved scaler.pkl")

# 7️⃣ Save intent encoder (list of intent columns)
intent_cols = [col for col in feature_cols if col.startswith("intent_")]
joblib.dump(intent_cols, "../data/processed/intent_encoder.pkl")
print("Saved intent_encoder.pkl")