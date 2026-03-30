import re
import statistics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def parse_transcript(raw_text: str) -> list[dict]:
    # Handle empty or invalid input
    if not raw_text or not isinstance(raw_text, str):
        return []

    turns = []

    # Regex pattern to extract turns
    pattern = r"Turn\s*(\d+)\s*:\s*(AGENT|CUSTOMER)\s*:\s*(.*?)(?=Turn\s*\d+\s*:|$)"

    matches = re.findall(pattern, raw_text, re.IGNORECASE | re.DOTALL)

    for match in matches:
        turn_number = int(match[0])
        speaker = match[1].strip().upper()
        text = match[2].strip()

        # Clean extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Validate
        if speaker not in ["AGENT", "CUSTOMER"]:
            continue

        if text:
            turns.append({
                "turn_number": turn_number,
                "speaker": speaker,
                "text": text
            })

    return turns

def add_synthetic_timestamps(turns: list[dict], duration_label: str) -> list[dict]:
    duration_map = {
        "short": 180,
        "medium": 300,
        "long": 480
    }

    total_duration = duration_map.get(duration_label, 300)

    if not turns:
        return turns

    time_per_turn = total_duration / len(turns)

    for i, turn in enumerate(turns):
        turn["timestamp"] = round(i * time_per_turn, 2)

    return turns

def compute_turn_sentiments(turns):
    scores = []

    for turn in turns:
        text = turn["text"]

        # Get sentiment score
        sentiment = analyzer.polarity_scores(text)["compound"]

        scores.append(sentiment)

    return scores


def compute_sentiment(turns):
    analyzer = SentimentIntensityAnalyzer()
    compounds = []

    for turn in turns:
        text = turn.get("text", "")
        if not text:
            compounds.append(0)
        else:
            score = analyzer.polarity_scores(text)["compound"]
            compounds.append(score)

    n = len(compounds)
    if n == 0:
        return {"mean_sentiment": 0, "last_20_sentiment": 0, "std_sentiment": 0}

    mean_sent = sum(compounds)/n
    last_20_sent = sum(compounds[int(n*0.8):]) / max(1, n - int(n*0.8))
    std_sent = (sum((x-mean_sent)**2 for x in compounds)/n)**0.5

    return {
        "mean_sentiment": round(mean_sent, 4),
        "last_20_sentiment": round(last_20_sent, 4),
        "std_sentiment": round(std_sent, 4)
    }


def compute_arc(series):
    if not series:
        return "flat"

    n = len(series)

    first_part = sum(series[:max(1, n//5)]) / max(1, n//5)
    last_part = sum(series[-max(1, n//5):]) / max(1, n//5)

    diff = last_part - first_part

    threshold = 0.15

    if diff > threshold:
        return "rise"
    elif diff < -threshold:
        return "fall"
    else:
        # check V-shape
        mid = min(series)

        if mid < first_part - 0.1 and last_part > mid + 0.1:
            return "v_shape"
        else:
            return "flat"
        

def talk_time_ratio(turns):
    if not turns:
        return 0

    agent_turns = 0

    for t in turns:
        if t["speaker"] == "AGENT":
            agent_turns += 1

    total_turns = len(turns)

    return round(agent_turns / total_turns, 3)   


def avg_word_count(turns):
    agent_words = []
    customer_words = []

    for t in turns:
        word_count = len(t["text"].split())

        if t["speaker"] == "AGENT":
            agent_words.append(word_count)
        else:
            customer_words.append(word_count)

    avg_agent = sum(agent_words)/len(agent_words) if agent_words else 0
    avg_customer = sum(customer_words)/len(customer_words) if customer_words else 0

    return {
        "avg_agent_words": round(avg_agent, 2),
        "avg_customer_words": round(avg_customer, 2)
    }

def resolution_flag(turns):
    if not turns:
        return 0

    keywords = [
        "issue resolved",
        "that's been sorted",
        "all set",
        "fixed that for you",
        "you're good to go"
    ]

    n = len(turns)
    last_n = max(1, int(0.2 * n))

    last_turns = turns[-last_n:]

    for t in last_turns:
        text = t["text"].lower()

        for kw in keywords:
            if kw in text:
                return 1

    return 0


def extract_conversation_features(turns):
    # Handle empty or invalid input
    if not turns or not isinstance(turns, list):
        return {
            "talk_time_ratio": 0,
            "interruption_count": 0,
            "resolution_flag": 0,
            "avg_agent_words": 0,
            "avg_customer_words": 0
        }

    features = {}

    # Safe execution of each feature
    try:
        features["talk_time_ratio"] = talk_time_ratio(turns)
    except:
        features["talk_time_ratio"] = 0

    try:
        features["interruption_count"] = interruption_count(turns)
    except:
        features["interruption_count"] = 0

    try:
        features["resolution_flag"] = resolution_flag(turns)
    except:
        features["resolution_flag"] = 0

    try:
        word_stats = avg_word_count(turns)
        if isinstance(word_stats, dict):
            features["avg_agent_words"] = word_stats.get("avg_agent_words", 0)
            features["avg_customer_words"] = word_stats.get("avg_customer_words", 0)
        else:
            features["avg_agent_words"] = 0
            features["avg_customer_words"] = 0
    except:
        features["avg_agent_words"] = 0
        features["avg_customer_words"] = 0

    return features


def extract_agent_behavior_features(turns: list[dict]) -> dict:
    """
    Extracts agent-specific behavioral features:
    - Empathy density (0-1)
    - Apology count
    - Transfer count

    Args:
        turns: list of turn dicts from parse_transcript

    Returns:
        dict with features
    """
    empathy_phrases = [
        "i understand", "i can see why", "i appreciate your patience",
        "i'm sorry to hear", "that must be frustrating", "i completely understand"
    ]
    apology_phrases = ["sorry", "apologise"]
    transfer_phrases = ["transfer you", "pass you to", "connect you with", "another department", "specialist"]

    agent_turns = [t for t in turns if t.get("speaker", "").upper() == "AGENT"]
    num_agent_turns = len(agent_turns)

    empathy_count = 0
    apology_count = 0
    transfer_count = 0

    for t in agent_turns:
        text = t.get("text", "").lower()
        if any(phrase in text for phrase in empathy_phrases):
            empathy_count += 1
        if any(phrase in text for phrase in apology_phrases):
            apology_count += 1
        if any(phrase in text for phrase in transfer_phrases):
            transfer_count += 1

    empathy_density = empathy_count / num_agent_turns if num_agent_turns > 0 else 0

    return {
        "empathy_density": round(empathy_density, 3),
        "apology_count": apology_count,
        "transfer_count": transfer_count
    }