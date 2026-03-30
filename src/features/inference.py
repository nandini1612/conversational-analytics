import numpy as np
import joblib

# Load scaler and intent columns
scaler = joblib.load("../data/processed/scaler.pkl")
intent_cols = joblib.load("../data/processed/intent_encoder.pkl")

def process_transcript(raw_text: str, call_metadata: dict) -> np.ndarray:
    """
    raw_text: transcript string
    call_metadata: dict with keys:
        - issue_type
        - call_duration
        - repeat_contact
    Returns: scaled feature vector for the model
    """
    # 1️⃣ Parse transcript and add timestamps
    turns = parse_transcript(raw_text)
    turns = add_synthetic_timestamps(turns, call_metadata.get("call_duration", "medium"))

    # 2️⃣ Extract features
    conv_features = extract_conversation_features(turns)
    sent_features = compute_sentiment(turns)
    agent_features = extract_agent_behavior_features(turns)

    # 3️⃣ Combine features
    features = {**conv_features, **sent_features, **agent_features}

    # 4️⃣ Metadata features
    duration_map = {"short": -1, "medium": 0, "long": 1}
    duration_ordinal = duration_map.get(call_metadata.get("call_duration"), 0)
    features["duration_ordinal"] = duration_ordinal

    # Duration deviation
    # Make sure intent_duration_means is already computed from training
    duration_deviation = duration_ordinal - intent_duration_means.get(call_metadata.get("issue_type"), 0)
    features["duration_deviation"] = duration_deviation

    # One-hot intent
    for col in intent_cols:
        features[col] = 1 if col == f"intent_{call_metadata.get('issue_type')}" else 0

    # Repeat contact
    features["repeat_contact"] = call_metadata.get("repeat_contact", 0)

    # 5️⃣ Build feature vector in same order as training
    feature_vector = np.array([features[col] for col in feature_cols], dtype=float).reshape(1, -1)

    # 6️⃣ Scale
    feature_vector_scaled = scaler.transform(feature_vector)

    return feature_vector_scaled