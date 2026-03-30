from preprocessing import parse_transcript, add_synthetic_timestamps
from preprocessing import compute_turn_sentiments, extract_conversation_features, compute_arc

# Sample transcript
raw_text = """Turn 1: AGENT: Hello how can I help?
Turn 2: CUSTOMER: My internet is very slow
Turn 3: AGENT: I will fix it
Turn 4: CUSTOMER: Thank you, it's working now"""

# Step 1: Parse
turns = parse_transcript(raw_text)

# Step 2: Add timestamps
turns = add_synthetic_timestamps(turns, "medium")

# Step 3: Get sentiment scores
scores = compute_turn_sentiments(turns)

# Step 4: Extract features
features = extract_conversation_features(turns)

# Step 5: Compute arc
arc = compute_arc(scores)

print("Turns:", turns)
print("Scores:", scores)
print("Features:", features)
print("Arc:", arc)