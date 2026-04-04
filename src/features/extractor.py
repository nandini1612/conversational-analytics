from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def extract_features(transcript):

    transcript_lower = transcript.lower()

    repeat_contact = int(
        "called before" in transcript_lower
    )

    transfer_count = transcript_lower.count("transfer")

    sentiment = analyzer.polarity_scores(
        transcript
    )["compound"]

    empathy_density = sum(
        word in transcript_lower
        for word in [
            "sorry",
            "understand",
            "apologize",
            "inconvenience"
        ]
    )

    return [
        repeat_contact,
        transfer_count,
        sentiment,
        empathy_density
    ]