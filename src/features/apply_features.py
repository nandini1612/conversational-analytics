# apply_features.py

import pandas as pd
from preprocessing import (
    parse_transcript,
    add_synthetic_timestamps,
    extract_conversation_features,
    compute_sentiment,
    extract_agent_behavior_features   # NEW: agent empathy/apology/transfer
)

# Input and output files
input_files = [
    "../../data/processed/train.csv",
    "../../data/processed/val.csv",
    "../../data/processed/test.csv"
]
output_files = [
    "../../data/processed/train_features.csv",
    "../../data/processed/val_features.csv",
    "../../data/processed/test_features.csv"

]

def process_csv(infile, outfile):
    """
    Reads a CSV with 'transcript' column, extracts conversation, sentiment, and agent features,
    merges them with original data, and saves to outfile.
    """
    df = pd.read_csv(infile)
    features_list = []

    # Map duration to ordinal
    duration_map = {"short": -1, "medium": 0, "long": 1}
    df["duration_ordinal"] = df["call_duration_seconds"].map(duration_map)

    # Compute per-intent mean duration (training data only)
    # Only compute once if processing train.csv
    if "train" in infile.lower():
        global intent_duration_means
        intent_duration_means = df.groupby("issue_type")["duration_ordinal"].mean().to_dict()

    # One-hot encode issue_type
    issue_dummies = pd.get_dummies(df["issue_type"], prefix="intent")
    df = pd.concat([df, issue_dummies], axis=1)

    # Ensure repeat_contact column exists
    if "repeat_contact" not in df.columns:
        df["repeat_contact"] = 0

    for idx, row in df.iterrows():
        # Raw transcript and call duration
        raw_text = row.get("transcript", "")
        duration = row.get("call_duration_seconds", "medium")

        # Parse transcript into structured turns
        turns = parse_transcript(raw_text)

        # Add synthetic timestamps
        turns = add_synthetic_timestamps(turns, duration)

        # Extract features
        conv_features = extract_conversation_features(turns)
        sent_features = compute_sentiment(turns)           # MUST return dict
        agent_features = extract_agent_behavior_features(turns)  # NEW agent features

        # Combine all feature dicts
        combined = {**conv_features, **sent_features, **agent_features}

        # Add per-intent duration deviation for this row
        duration_deviation = row["duration_ordinal"] - intent_duration_means.get(row["issue_type"], 0)
        combined["duration_deviation"] = duration_deviation

        # Add one-hot intent columns
        for col in issue_dummies.columns:
            combined[col] = row[col]

        # Add repeat_contact
        combined["repeat_contact"] = row["repeat_contact"]
        
        # Optional: add call ID if exists
        combined["call_id"] = row.get("call_id", idx)

        # Append to feature list
        features_list.append(combined)

    # Convert feature list to DataFrame
    features_df = pd.DataFrame(features_list)

    # Merge with original data
    final_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # Save to CSV
    final_df.to_csv(outfile, index=False)
    print(f"Saved features to {outfile}")


# Process all input files
for infile, outfile in zip(input_files, output_files):
    process_csv(infile, outfile)