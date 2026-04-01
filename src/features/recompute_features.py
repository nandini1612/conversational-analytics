"""
Recompute NLP features from transcript_text column.

Person 1's apply_features.py used 'transcript' column but the CSVs have
'transcript_text', so Groups A/B/C were never computed — all zeros.
Also duration_ordinal and duration_deviation are NaN.

This script fixes all three splits in-place using Person 1's own
preprocessing functions from src/features/preprocessing.py.

Run: python src/features/recompute_features.py
"""

import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "features"))

from preprocessing import (
    parse_transcript,
    compute_sentiment,
    extract_conversation_features,
    extract_agent_behavior_features,
)

PROCESSED_DIR = ROOT / "data" / "processed"

# ── Interruption count (missing from preprocessing.py's extract_conversation_features) ──
def interruption_count(turns):
    count = 0
    for i in range(1, len(turns)):
        prev_speaker = turns[i-1]["speaker"]
        curr_speaker = turns[i]["speaker"]
        curr_words   = len(turns[i]["text"].split())
        if curr_speaker != prev_speaker and curr_words < 5:
            count += 1
    return count


def extract_all_features(row, intent_duration_means):
    """Extract all NLP + metadata features for a single row."""
    transcript = str(row.get("transcript_text", ""))
    issue_type = str(row.get("issue_type", "")).lower().strip()

    # Duration ordinal
    dur_secs = row.get("call_duration_seconds", None)
    if pd.notna(dur_secs):
        dur_secs = float(dur_secs)
        if dur_secs <= 240:
            dur_ord = -1
        elif dur_secs <= 420:
            dur_ord = 0
        else:
            dur_ord = 1
    else:
        dur_ord = 0

    # Parse transcript
    turns = parse_transcript(transcript)

    # Group A — sentiment
    sent = compute_sentiment(turns)

    # Group B — structure
    conv = extract_conversation_features(turns)
    conv["interruption_count"] = interruption_count(turns)

    # Group C — agent behaviour
    agent = extract_agent_behavior_features(turns)

    # Group D — metadata
    dur_dev = dur_ord - intent_duration_means.get(issue_type, 0.0)

    return {
        # Group A
        "mean_sentiment":     sent["mean_sentiment"],
        "last_20_sentiment":  sent["last_20_sentiment"],
        "std_sentiment":      sent["std_sentiment"],
        # Group B
        "talk_time_ratio":    conv["talk_time_ratio"],
        "avg_agent_words":    conv["avg_agent_words"],
        "avg_customer_words": conv["avg_customer_words"],
        "interruption_count": conv["interruption_count"],
        "resolution_flag":    conv["resolution_flag"],
        # Group C
        "empathy_density":    agent["empathy_density"],
        "apology_count":      agent["apology_count"],
        "transfer_count":     agent["transfer_count"],
        # Group D
        "duration_ordinal":   dur_ord,
        "duration_deviation": round(dur_dev, 4),
    }


def recompute_split(path: Path, intent_duration_means: dict) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop duplicate .1 columns
    mask = ~df.columns.duplicated(keep="first")
    df = df.loc[:, mask].copy()
    df.drop(columns=[c for c in df.columns if c.endswith(".1")], inplace=True, errors="ignore")

    print(f"  Processing {path.name} ({len(df)} rows)...")

    features = []
    for _, row in df.iterrows():
        features.append(extract_all_features(row, intent_duration_means))

    feat_df = pd.DataFrame(features)

    # Overwrite the zeroed/NaN columns
    for col in feat_df.columns:
        df[col] = feat_df[col].values

    # Fix repeat_contact
    df["repeat_contact"] = (
        df["repeat_contact"].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
        .fillna(0).astype(float)
    )

    return df


def main():
    print("=" * 60)
    print("RECOMPUTING NLP FEATURES FROM TRANSCRIPTS")
    print("=" * 60)

    # Load train first to compute intent_duration_means on training data only
    train_raw = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    mask = ~train_raw.columns.duplicated(keep="first")
    train_raw = train_raw.loc[:, mask]
    train_raw.drop(columns=[c for c in train_raw.columns if c.endswith(".1")], inplace=True, errors="ignore")

    # Compute duration ordinal for train to get per-intent means
    def dur_to_ord(secs):
        try:
            s = float(secs)
            if s <= 240: return -1
            elif s <= 420: return 0
            else: return 1
        except:
            return 0

    train_raw["_dur_ord"] = train_raw["call_duration_seconds"].apply(dur_to_ord)
    intent_duration_means = (
        train_raw.groupby("issue_type")["_dur_ord"].mean()
        .to_dict()
    )
    # Lowercase keys
    intent_duration_means = {k.lower().strip(): v for k, v in intent_duration_means.items()}
    print(f"\n  Intent duration means (from train): {intent_duration_means}")

    for split in ["train_features.csv", "val_features.csv", "test_features.csv"]:
        path = PROCESSED_DIR / split
        df = recompute_split(path, intent_duration_means)
        df.to_csv(path, index=False)
        print(f"  Saved {split}")

        # Quick sanity check
        nlp_cols = ["mean_sentiment", "talk_time_ratio", "empathy_density",
                    "duration_ordinal", "duration_deviation"]
        for col in nlp_cols:
            if col in df.columns:
                print(f"    {col:<25} mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
                      f"nonzero={(df[col] != 0).sum()}")
        print()

    print("=" * 60)
    print("DONE — re-run ridge.py, random_forest.py, ensemble.py, evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
