"""
Phase 6 — predict() Function
Person 2 (Models & Evaluation)

CONTRACT (agreed with Person 3):
  Input:
    transcript_text (str)
    call_metadata (dict):
      issue_type            (str)   e.g. "billing", "technical"
      call_duration_seconds (int)   raw seconds from CRM
      repeat_contact        (int)   0 or 1
      resolution_status     (str)   "resolved" or "unresolved"  ← REQUIRED
                                    Agent records this in CRM at call end.
                                    Most important predictor (r=0.82 with CSAT).

  Output dict:
    csat_score          (float, clipped [1.0, 5.0])
    confidence_interval (list [lower, upper], clipped [1.0, 5.0])
    emotional_arc       (str: 'rise'|'fall'|'flat'|'v_shape')
    shap_values         (dict: stub — Person 3 fills this)

All artefacts are loaded once at module import time and cached.
BERT fallback: if bert_weights/ absent, uses Ridge+RF with renormalised weights.

IMPORTANT: resolution_status MUST be in call_metadata. Inferring resolution
from transcript keywords does not work on synthetic data (only 47 unique
phrases, none contain resolution keywords reliably).
"""

import sys

# Import torch and transformers FIRST before any other heavy imports.
# On Windows, torch DLLs must be initialised before numpy/scipy load their own
# C extensions, otherwise c10.dll fails to initialise (DLL ordering issue).
try:
    import torch
    import transformers as _transformers_preload  # noqa: F401
except Exception:
    pass  # will be caught properly in _load_artefacts

import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "src" / "models"))

MODELS_DIR = ROOT / "models"
MODELS_SAVED_DIR = ROOT / "models"  # consolidated — bert_weights here too
REPORTS_DIR = ROOT / "outputs" / "metrics"

# Artefact cache — loaded once at import
_ARTEFACTS: dict = {}


def _load_artefacts() -> dict:
    global _ARTEFACTS
    if _ARTEFACTS:
        return _ARTEFACTS

    required = {
        "ridge": MODELS_DIR / "ridge_model.pkl",
        "rf": MODELS_DIR / "rf_model.pkl",
        "weights": REPORTS_DIR / "ensemble_weights.json",
    }

    for key, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artefact: {path}\n"
                f"Run the corresponding phase file to generate it."
            )

    with open(required["ridge"], "rb") as f:
        _ARTEFACTS["ridge"] = pickle.load(f)
    with open(required["rf"], "rb") as f:
        _ARTEFACTS["rf"] = pickle.load(f)
    with open(required["weights"]) as f:
        _ARTEFACTS["weights"] = json.load(f)

    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            _ARTEFACTS["scaler"] = pickle.load(f)
    else:
        _ARTEFACTS["scaler"] = None

    # DistilBERT — optional, load only if weights directory exists
    bert_dir = MODELS_SAVED_DIR / "bert_weights"
    if bert_dir.exists() and bert_dir.is_dir():
        try:
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizer,
            )
            import torch

            _ARTEFACTS["bert_model"] = (
                DistilBertForSequenceClassification.from_pretrained(str(bert_dir))
            )
            _ARTEFACTS["bert_tokenizer"] = DistilBertTokenizer.from_pretrained(
                str(bert_dir)
            )
            _ARTEFACTS["bert_model"].eval()
            _ARTEFACTS["bert_ready"] = True
            print(f"  [INFO] DistilBERT loaded from {bert_dir}")
        except Exception as e:
            print(f"  [WARN] DistilBERT load failed ({e}). Falling back to Ridge+RF.")
            _ARTEFACTS["bert_ready"] = False
    else:
        _ARTEFACTS["bert_ready"] = False

    return _ARTEFACTS


def _extract_features(transcript_text: str, call_metadata: dict) -> np.ndarray:
    """
    Extract 22-element feature vector from transcript + metadata.
    Uses Person 1's preprocessing functions for NLP features; reads
    resolution_flag directly from call_metadata['resolution_status'].

    resolution_flag MUST come from call_metadata — NOT from transcript keyword
    matching. Transcript keyword matching gives resolution_flag=0 for all
    synthetic calls (only 47 unique phrases, none match resolution keywords).
    The agent records resolution_status in the CRM at call end; it is
    available at inference time and is NOT data leakage.
    """
    from phase0_skeleton import FEATURE_COLUMNS

    sys.path.insert(0, str(ROOT / "src" / "features"))

    col_idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    vec = np.zeros(len(FEATURE_COLUMNS))

    # ── Group B: resolution_flag — from agent CRM, NOT transcript text ──────
    # This is the single most important feature (r=0.82 with CSAT).
    # Using transcript keyword matching causes resolution_flag=0 for all rows
    # in this dataset, destroying model performance (MAE goes 0.6 → 1.1+).
    resolution_status = str(call_metadata.get("resolution_status", "")).strip().lower()
    resolution_flag = 1 if resolution_status == "resolved" else 0
    if "resolution_flag" in col_idx:
        vec[col_idx["resolution_flag"]] = resolution_flag

    # ── NLP features from transcript (Groups A, B partial, C) ────────────────
    try:
        from preprocessing import (
            parse_transcript,
            compute_sentiment,
            extract_conversation_features,
            extract_agent_behavior_features,
        )

        turns = parse_transcript(transcript_text)

        # Group A — sentiment
        sent = compute_sentiment(turns)
        vec[col_idx["mean_sentiment"]] = sent["mean_sentiment"]
        vec[col_idx["last_20_sentiment"]] = sent["last_20_sentiment"]
        vec[col_idx["std_sentiment"]] = sent["std_sentiment"]

        # Group B — structure (resolution_flag already set above — do NOT overwrite)
        conv = extract_conversation_features(turns)
        vec[col_idx["talk_time_ratio"]] = conv["talk_time_ratio"]
        vec[col_idx["avg_agent_words"]] = conv["avg_agent_words"]
        vec[col_idx["avg_customer_words"]] = conv["avg_customer_words"]
        # resolution_flag intentionally NOT read from conv — see note above

        # Interruption count
        ic = 0
        for i in range(1, len(turns)):
            if (
                turns[i]["speaker"] != turns[i - 1]["speaker"]
                and len(turns[i]["text"].split()) < 5
            ):
                ic += 1
        vec[col_idx["interruption_count"]] = ic

        # Group C — agent behaviour
        agent = extract_agent_behavior_features(turns)
        vec[col_idx["empathy_density"]] = agent["empathy_density"]
        vec[col_idx["apology_count"]] = agent["apology_count"]
        vec[col_idx["transfer_count"]] = agent["transfer_count"]

    except Exception:
        # Fallback: VADER sentiment only (preprocessing module unavailable)
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            sia = SentimentIntensityAnalyzer()
            raw_turns = [t.strip() for t in transcript_text.split("|") if t.strip()]
            scores = [sia.polarity_scores(t)["compound"] for t in raw_turns]
            if scores:
                vec[col_idx["mean_sentiment"]] = float(np.mean(scores))
                vec[col_idx["std_sentiment"]] = float(np.std(scores))
                n_last = max(1, len(scores) // 5)
                vec[col_idx["last_20_sentiment"]] = float(np.mean(scores[-n_last:]))
        except ImportError:
            pass  # leave sentiment features at 0.0

    # ── Group D — metadata ───────────────────────────────────────────────────
    dur_secs = call_metadata.get("call_duration_seconds", 0)
    try:
        dur_secs = int(dur_secs)
    except (TypeError, ValueError):
        dur_secs = 0
    # Bucket into ordinal: short <180s, long >600s, else medium
    if dur_secs < 180:
        dur_ord = -1
    elif dur_secs > 600:
        dur_ord = 1
    else:
        dur_ord = 0
    vec[col_idx["duration_ordinal"]] = dur_ord
    vec[col_idx["repeat_contact"]] = int(call_metadata.get("repeat_contact", 0))
    # duration_deviation: 0 at inference (intent-mean lookup requires training data)
    vec[col_idx["duration_deviation"]] = 0.0

    # Intent one-hot
    issue = str(call_metadata.get("issue_type", "")).lower().strip()
    intent_col = f"intent_{issue}"
    if intent_col in col_idx:
        vec[col_idx[intent_col]] = 1.0

    return vec.reshape(1, -1)


def _bert_predict(transcript_text: str, art: dict) -> Optional[float]:
    """Run DistilBERT inference. Returns None if BERT unavailable or fails."""
    if not art.get("bert_ready"):
        return None
    try:
        import torch

        tokenizer = art["bert_tokenizer"]
        model = art["bert_model"]
        inputs = tokenizer(
            transcript_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            output = model(**inputs)
        pred = output.logits.squeeze().item()
        return float(np.clip(pred, 1.0, 5.0))
    except Exception as e:
        print(f"  [WARN] BERT inference failed: {e}")
        return None


def _compute_emotional_arc(transcript_text: str) -> str:
    """
    Compute emotional arc from per-turn VADER compound scores.
    Returns one of: 'rise', 'fall', 'flat', 'v_shape'
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        turns = [t.strip() for t in transcript_text.split("|") if t.strip()]
        scores = [sia.polarity_scores(t)["compound"] for t in turns]
        if len(scores) < 4:
            return "flat"
        n = max(1, len(scores) // 5)
        first_avg = float(np.mean(scores[:n]))
        last_avg = float(np.mean(scores[-n:]))
        mid_scores = scores[n:-n] if len(scores) > 2 * n else scores
        mid_min = float(min(mid_scores))
        delta = last_avg - first_avg
        if delta > 0.15:
            return "rise"
        if delta < -0.15:
            return "fall"
        if mid_min < first_avg - 0.15 and last_avg > first_avg - 0.05:
            return "v_shape"
        return "flat"
    except ImportError:
        # Fallback: keyword heuristic
        text_lower = transcript_text.lower()
        positive_end = any(
            p in text_lower[-300:]
            for p in ["thank you", "resolved", "all set", "appreciate", "sorted"]
        )
        negative_end = any(
            p in text_lower[-300:]
            for p in ["still not", "not fixed", "frustrated", "again"]
        )
        if positive_end and not negative_end:
            return "rise"
        if negative_end:
            return "fall"
        return "flat"


def predict(transcript_text: str, call_metadata: dict) -> dict:
    """
    End-to-end CSAT prediction for a single call.

    Args:
        transcript_text: raw transcript string (pipe-delimited turns)
        call_metadata:   dict with REQUIRED keys:
                           issue_type        (str)  e.g. "billing", "technical"
                           call_duration_seconds (int) raw call duration
                           repeat_contact    (int: 0 or 1)
                           resolution_status (str: 'resolved' or 'unresolved')
                             ↑ agent records this in CRM at call end — NOT leakage

    Returns:
        {
          "csat_score":          float (1.0–5.0),
          "confidence_interval": [lower, upper],  # clipped to [1.0, 5.0]
          "emotional_arc":       str,
          "shap_values":         {}   # stub — Person 3 fills this
        }

    Raises:
        ValueError if resolution_status is missing from call_metadata.
    """
    if "resolution_status" not in call_metadata:
        raise ValueError(
            "call_metadata must include 'resolution_status' "
            "('resolved' or 'unresolved'). "
            "This is the single most important CSAT predictor (r=0.82) and "
            "must be recorded by the agent in the CRM at call end."
        )
    art = _load_artefacts()
    w = art["weights"]

    # Extract features and replace NaN with 0.0
    X_raw = _extract_features(transcript_text, call_metadata)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    # Scale for Ridge
    if art["scaler"] is not None:
        X_scaled = art["scaler"].transform(X_raw)
    else:
        X_scaled = X_raw

    ridge_pred = float(np.clip(art["ridge"].predict(X_scaled)[0], 1.0, 5.0))
    rf_pred = float(np.clip(art["rf"].predict(X_raw)[0], 1.0, 5.0))
    bert_pred = _bert_predict(transcript_text, art)

    if bert_pred is not None:
        # Full three-model ensemble
        csat = w["ridge"] * ridge_pred + w["rf"] * rf_pred + w["bert"] * bert_pred
        all_preds = [ridge_pred, rf_pred, bert_pred]
    else:
        # BERT absent — renormalise Ridge+RF weights to sum to 1.0
        if not art.get("bert_ready"):
            print(
                "  [WARN] BERT weights absent — using Ridge+RF ensemble with renormalised weights."
            )
        ridge_w = w["ridge"] / (w["ridge"] + w["rf"])
        rf_w = w["rf"] / (w["ridge"] + w["rf"])
        csat = ridge_w * ridge_pred + rf_w * rf_pred
        all_preds = [ridge_pred, rf_pred]

    csat = float(np.clip(csat, 1.0, 5.0))

    # Confidence interval: [csat - std, csat + std], clipped to [1.0, 5.0]
    pred_std = float(np.std(all_preds))
    ci = [
        round(float(np.clip(csat - pred_std, 1.0, 5.0)), 3),
        round(float(np.clip(csat + pred_std, 1.0, 5.0)), 3),
    ]

    arc = _compute_emotional_arc(transcript_text)

    return {
        "csat_score": round(csat, 3),
        "confidence_interval": ci,
        "emotional_arc": arc,
        "shap_values": {},
    }


# ── End-to-end test (5 representative calls) ─────────────────


def _run_end_to_end_test():
    test_calls = [
        {
            "transcript": "Turn 1: CUSTOMER: my bill is wrong | Turn 2: AGENT: I understand | Turn 3: CUSTOMER: still not fixed | Turn 4: AGENT: I apologize",
            "metadata": {
                "issue_type": "billing",
                "call_duration_seconds": 120,
                "repeat_contact": 1,
                "resolution_status": "unresolved",
            },
            "label": "billing, repeat, short, unresolved → expect low CSAT",
        },
        {
            "transcript": "Turn 1: CUSTOMER: nothing works | Turn 2: AGENT: happy to help | Turn 3: CUSTOMER: okay thank you | Turn 4: AGENT: all sorted",
            "metadata": {
                "issue_type": "technical",
                "call_duration_seconds": 300,
                "repeat_contact": 0,
                "resolution_status": "resolved",
            },
            "label": "technical, no-repeat, resolved → expect high CSAT",
        },
        {
            "transcript": "Turn 1: CUSTOMER: called three times | Turn 2: AGENT: I completely understand | Turn 3: CUSTOMER: still frustrated | Turn 4: AGENT: I will escalate",
            "metadata": {
                "issue_type": "account",
                "call_duration_seconds": 720,
                "repeat_contact": 1,
                "resolution_status": "unresolved",
            },
            "label": "account, repeat, long, unresolved → expect low CSAT",
        },
        {
            "transcript": "Turn 1: CUSTOMER: thanks for the support | Turn 2: AGENT: glad to help | Turn 3: CUSTOMER: all set",
            "metadata": {
                "issue_type": "payment",
                "call_duration_seconds": 90,
                "repeat_contact": 0,
                "resolution_status": "resolved",
            },
            "label": "payment, positive, resolved → expect high CSAT",
        },
        {
            "transcript": "Turn 1: CUSTOMER: why is this happening again | Turn 2: AGENT: we are working on it | Turn 3: CUSTOMER: third time calling | Turn 4: AGENT: sorry for the trouble",
            "metadata": {
                "issue_type": "network",
                "call_duration_seconds": 240,
                "repeat_contact": 1,
                "resolution_status": "unresolved",
            },
            "label": "network, repeat, frustrated, unresolved → expect low CSAT",
        },
    ]

    print("=" * 60)
    print("PHASE 6 — END-TO-END TEST (5 calls)")
    print("=" * 60)

    all_passed = True
    for i, call in enumerate(test_calls, 1):
        t0 = time.time()
        result = predict(call["transcript"], call["metadata"])
        elapsed = time.time() - t0

        # Schema assertions
        assert "csat_score" in result, "Missing csat_score"
        assert "confidence_interval" in result, "Missing confidence_interval"
        assert "emotional_arc" in result, "Missing emotional_arc"
        assert "shap_values" in result, "Missing shap_values"
        assert 1.0 <= result["csat_score"] <= 5.0, (
            f"csat_score out of range: {result['csat_score']}"
        )
        assert len(result["confidence_interval"]) == 2, "CI must have 2 elements"
        assert all(1.0 <= v <= 5.0 for v in result["confidence_interval"]), (
            "CI out of range"
        )
        assert result["emotional_arc"] in ("rise", "fall", "flat", "v_shape"), (
            f"Invalid arc: {result['emotional_arc']}"
        )
        assert isinstance(result["shap_values"], dict), "shap_values must be dict"

        timing_ok = elapsed < 5.0
        if not timing_ok:
            all_passed = False

        print(f"\n  Call {i}: {call['label']}")
        print(f"    csat_score:          {result['csat_score']}")
        print(f"    confidence_interval: {result['confidence_interval']}")
        print(f"    emotional_arc:       {result['emotional_arc']}")
        print(
            f"    wall-clock time:     {elapsed:.3f}s  {'OK' if timing_ok else 'SLOW'}"
        )

    print("\n" + "=" * 60)
    if all_passed:
        print("PHASE 6 COMPLETE — all 5 calls passed.")
        print("  predict() is ready for Person 3 to wrap in FastAPI.")
    else:
        print("PHASE 6: some checks failed — review output above.")
    print("=" * 60)


if __name__ == "__main__":
    _load_artefacts()
    _run_end_to_end_test()
