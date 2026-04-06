"""
src/features/inference.py  —  Fixed by Person 4.

The scaler was fitted on exactly these 24 columns (derived from train_features.csv).
Intent columns use shortened names (intent_billing not intent_billing_error).
"""

import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ── Load saved artifacts ───────────────────────────────────────────────────────
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# ── Exact 24 feature columns in the order the scaler was fitted on ─────────────
FEATURE_COLS = [
    "repeat_contact",
    "duration_ordinal",
    "intent_account",
    "intent_billing",
    "intent_complaint",
    "intent_delivery",
    "intent_login",
    "intent_network",
    "intent_payment",
    "intent_refund",
    "intent_subscription",
    "intent_technical",
    "talk_time_ratio",
    "interruption_count",
    "resolution_flag",
    "avg_agent_words",
    "avg_customer_words",
    "mean_sentiment",
    "last_20_sentiment",
    "std_sentiment",
    "empathy_density",
    "apology_count",
    "transfer_count",
    "duration_deviation",
]

# ── Map full issue_type values to the short intent column names ────────────────
ISSUE_TO_INTENT = {
    "account_access":    "intent_account",
    "billing_error":     "intent_billing",
    "complaint":         "intent_complaint",
    "delivery":          "intent_delivery",
    "login":             "intent_login",
    "network":           "intent_network",
    "payment":           "intent_payment",
    "refund":            "intent_refund",
    "subscription":      "intent_subscription",
    "technical_support": "intent_technical",
    # short forms too
    "account":    "intent_account",
    "billing":    "intent_billing",
    "technical":  "intent_technical",
}

# ── Intent duration means from training data ───────────────────────────────────
try:
    _train = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "train.csv"))
    _dur_map = {"short": -1, "medium": 0, "long": 1}
    if "call_duration" not in _train.columns and "call_duration_seconds" in _train.columns:
        _train["call_duration"] = pd.cut(
            _train["call_duration_seconds"],
            bins=[0, 240, 390, float("inf")],
            labels=["short", "medium", "long"],
        ).astype(str)
    _train["_dur_ord"] = _train.get("call_duration", pd.Series(["medium"]*len(_train))).map(_dur_map).fillna(0)
    INTENT_DURATION_MEANS = _train.groupby("issue_type")["_dur_ord"].mean().to_dict()
except Exception:
    INTENT_DURATION_MEANS = {}

# ── Imports from Person 1's preprocessing module ──────────────────────────────
from src.features.preprocessing import (
    parse_transcript,
    add_synthetic_timestamps,
    extract_conversation_features,
    compute_sentiment,
    extract_agent_behavior_features,
)


def process_transcript(raw_text: str, call_metadata: dict) -> np.ndarray:
    """
    Convert a raw transcript + metadata dict into a scaled 24-feature vector.
    call_metadata keys: issue_type, call_duration, repeat_contact
    Returns: np.ndarray shape (1, 24)
    """
    # 1. Parse + timestamp
    turns = parse_transcript(raw_text)
    turns = add_synthetic_timestamps(turns, call_metadata.get("call_duration", "medium"))

    # 2. Extract feature groups
    conv_feats  = extract_conversation_features(turns)
    sent_feats  = compute_sentiment(turns)
    agent_feats = extract_agent_behavior_features(turns)
    all_feats   = {**conv_feats, **sent_feats, **agent_feats}

    # 3. Duration ordinal
    duration_map = {"short": -1, "medium": 0, "long": 1}
    dur_ord = duration_map.get(call_metadata.get("call_duration", "medium"), 0)

    # 4. Duration deviation
    issue   = call_metadata.get("issue_type", "")
    dur_dev = dur_ord - INTENT_DURATION_MEANS.get(issue, 0.0)

    # 5. One-hot intent
    intent_col  = ISSUE_TO_INTENT.get(issue, None)
    intent_feats = {col: 0 for col in FEATURE_COLS if col.startswith("intent_")}
    if intent_col and intent_col in intent_feats:
        intent_feats[intent_col] = 1

    # 6. Repeat contact
    rc = call_metadata.get("repeat_contact", 0)
    if isinstance(rc, str):
        rc = 1 if rc.strip().lower() in ("yes", "true", "1", "1.0") else 0
    repeat_contact = int(rc)

    # 7. Build dict with all 24 columns
    feature_dict = {
        "repeat_contact":     repeat_contact,
        "duration_ordinal":   dur_ord,
        "duration_deviation": dur_dev,
        **intent_feats,
        "talk_time_ratio":    all_feats.get("talk_time_ratio",    0.0),
        "interruption_count": all_feats.get("interruption_count", 0),
        "resolution_flag":    all_feats.get("resolution_flag",    0),
        "avg_agent_words":    all_feats.get("avg_agent_words",    0.0),
        "avg_customer_words": all_feats.get("avg_customer_words", 0.0),
        "mean_sentiment":     all_feats.get("mean_sentiment",     0.0),
        "last_20_sentiment":  all_feats.get("last_20_sentiment",  0.0),
        "std_sentiment":      all_feats.get("std_sentiment",      0.0),
        "empathy_density":    all_feats.get("empathy_density",    0.0),
        "apology_count":      all_feats.get("apology_count",      0),
        "transfer_count":     all_feats.get("transfer_count",     0),
    }

    # 8. Build vector in exact column order the scaler expects
    vector = np.array(
        [feature_dict[col] for col in FEATURE_COLS], dtype=float
    ).reshape(1, -1)

    # 9. Scale and return
    return scaler.transform(vector)