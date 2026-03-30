from preprocessing import parse_transcript
from preprocessing import extract_conversation_features

raw_text = """Turn 1: AGENT: Hello how can I help
Turn 2: CUSTOMER: My internet is not working
Turn 3: AGENT: Okay
Turn 4: CUSTOMER: Still slow
Turn 5: AGENT: Fixed that for you"""

turns = parse_transcript(raw_text)

features = extract_conversation_features(turns)

print("Parsed Turns:")
print(turns)

print("\nFeatures:")
print(features)