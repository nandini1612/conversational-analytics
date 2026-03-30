from preprocessing import parse_transcript, add_synthetic_timestamps

# Create 20 test transcripts
test_data = [
    "Turn 1: AGENT: Hello\nTurn 2: CUSTOMER: Hi",
    "Turn 1: agent: hello\nTurn 2: customer: issue here",
    "Turn 1: AGENT:   Hello   \nTurn 2: CUSTOMER:   Hi   ",
    "",
    "Random text without format",
] * 4   # total = 20

# Run tests
for i, transcript in enumerate(test_data):
    parsed = parse_transcript(transcript)
    parsed = add_synthetic_timestamps(parsed, "medium")

    print(f"\nTest Case {i+1}")
    print(parsed)