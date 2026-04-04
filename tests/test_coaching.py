from src.coaching.generator import generate_summary

positive = [
    ("Sorry for inconvenience", "positive")
]

negative = [
    ("I called before but still not fixed", "negative"),
    ("Let me transfer you", "negative")
]

print(generate_summary(positive, negative))