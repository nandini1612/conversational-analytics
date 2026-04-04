import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def find_repeat_contact_phrase(transcript):

    patterns = [
        "called before",
        "last time",
        "still not fixed",
        "called last week",
        "again calling"
    ]

    lines = transcript.split("\n")

    for line in lines:
        for pattern in patterns:
            if pattern in line.lower():
                return line.strip()

    return None

def find_transfer_phrase(transcript):

    patterns = [
        "transfer you",
        "connect you",
        "hold while I transfer",
        "forward your call"
    ]

    lines = transcript.split("\n")

    for line in lines:
        for pattern in patterns:
            if pattern in line.lower():
                return line.strip()

    return None

def find_final_sentiment_phrase(transcript):

    lines = transcript.split("\n")

    last_20 = lines[int(len(lines)*0.8):]

    scores = []

    for line in last_20:
        score = analyzer.polarity_scores(line)["compound"]
        scores.append((line, score))

    if not scores:
        return None

    worst = min(scores, key=lambda x: x[1])

    return worst[0].strip()

def find_empathy_phrase(transcript):

    empathy_words = [
        "sorry",
        "understand",
        "apologize",
        "inconvenience",
        "frustrating"
    ]

    lines = transcript.split("\n")

    best_line = None
    max_count = 0

    for line in lines:

        count = sum(
            word in line.lower()
            for word in empathy_words
        )

        if count > max_count:
            max_count = count
            best_line = line

    return best_line

def map_phrases(transcript, shap_features):

    positive = []
    negative = []

    for feature, value in shap_features:

        if feature == "repeat_contact":

            phrase = find_repeat_contact_phrase(transcript)

        elif feature == "transfer_count":

            phrase = find_transfer_phrase(transcript)

        elif feature == "final_sentiment":

            phrase = find_final_sentiment_phrase(transcript)

        elif feature == "empathy_density":

            phrase = find_empathy_phrase(transcript)

        else:
            continue

        if phrase:

            if value > 0:
                positive.append((phrase, "positive"))
            else:
                negative.append((phrase, "negative"))

    return positive[:3], negative[:3]