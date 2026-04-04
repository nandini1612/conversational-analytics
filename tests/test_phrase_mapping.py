from src.phrase_mapping.mapper import map_phrases

transcript = """
Customer: I called before but issue still not fixed
Agent: Sorry for inconvenience
Agent: Let me transfer you
Customer: This is frustrating
"""

shap_features = [
    ("repeat_contact", -0.4),
    ("empathy_density", 0.3),
    ("transfer_count", -0.2)
]

print(map_phrases(transcript, shap_features))