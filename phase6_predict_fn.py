"""
Phase 6 — predict() Function
Person 2 (Models & Evaluation)

Writes the end-to-end predict() function that Person 3 wraps in FastAPI.
Loads all saved artefacts once at import time, then processes each call.

CONTRACT (agreed with Person 3):
  Input:  transcript_text (str), call_metadata (dict with keys:
            issue_type (str), call_duration (str: 'short'|'medium'|'long'),
            repeat_contact (int: 0 or 1))
  Output: dict with:
            csat_score          (float, clipped 1–5)
            confidence_interval (tuple: (lower, upper) = pred ± 1 std)
            emotional_arc       (str: 'rise'|'fall'|'flat'|'v_shape')
            shap_values         (dict: stub — Person 3 fills this)

Run: python phase6_predict_fn.py
     → runs end-to-end on 5 held-out dummy calls and prints results.
     Verify output structure matches shared_types.py before handoff.
     Verify total wall-clock time < 5 seconds per call on CPU.
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ── Artefact cache (loaded once at import) ───────────────────
_ARTEFACTS: dict = {}


def _load_artefacts():
    """Load all model artefacts into module-level cache. Called once."""
    global _ARTEFACTS
    if _ARTEFACTS:
        return _ARTEFACTS

    required = {
        "ridge": MODELS_DIR / "ridge_model.pkl",
        "rf": MODELS_DIR / "rf_model.pkl",
        "weights": OUTPUTS_DIR / "ensemble_weights.json",
    }
    optional = {
        "scaler": MODELS_DIR / "scaler.pkl",
        "bert": MODELS_DIR / "bert_weights",  # directory, checked differently
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

    if optional["scaler"].exists():
        with open(optional["scaler"], "rb") as f:
            _ARTEFACTS["scaler"] = pickle.load(f)
    else:
        _ARTEFACTS["scaler"] = None

    # DistilBERT — load only if weights directory exists
    bert_dir = optional["bert"]
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
            print("  [INFO] DistilBERT loaded from", bert_dir)
        except Exception as e:
            print(
                f"  [WARN] DistilBERT load failed ({e}). Falling back to Ridge+RF ensemble."
            )
            _ARTEFACTS["bert_ready"] = False
    else:
        _ARTEFACTS["bert_ready"] = False

    return _ARTEFACTS


# ── Feature extraction (calls Person 1's preprocessing.py) ──


def _extract_features(transcript_text: str, call_metadata: dict) -> np.ndarray:
    """
    Extract the feature vector for a single call.

    Primary path: call Person 1's process_transcript() from preprocessing.py.
    Fallback path (if preprocessing.py not available): build a zero vector
    and populate the metadata features we can compute without it.

    Args:
        transcript_text: raw transcript string
        call_metadata:   dict with issue_type, call_duration, repeat_contact

    Returns:
        feature vector as np.ndarray of shape (n_features,)
    """
    from phase0_skeleton import FEATURE_COLUMNS

    try:
        # Primary: use Person 1's shared function
        import preprocessing

        features, _ = preprocessing.process_transcript(transcript_text, call_metadata)
        return np.array(features).reshape(1, -1)

    except ImportError:
        # Fallback: build a feature vector manually from metadata only.
        # Sentiment/structure/agent features are zeroed — acceptable for
        # integration testing before preprocessing.py is delivered.
        print("  [WARN] preprocessing.py not found — using metadata-only features.")

        n = len(FEATURE_COLUMNS)
        vec = np.zeros(n)
        col_idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}

        # Duration ordinal
        dur_map = {"short": -1, "medium": 0, "long": 1}
        dur_str = str(call_metadata.get("call_duration", "medium")).lower()
        if col_idx.get("duration_ordinal") is not None:
            vec[col_idx["duration_ordinal"]] = dur_map.get(dur_str, 0)

        # Repeat contact
        if col_idx.get("repeat_contact") is not None:
            vec[col_idx["repeat_contact"]] = int(call_metadata.get("repeat_contact", 0))

        # Issue type one-hot
        issue = str(call_metadata.get("issue_type", "")).lower()
        issue_col = f"issue_type_{issue}"
        if issue_col in col_idx:
            vec[col_idx[issue_col]] = 1.0

        return vec.reshape(1, -1)


def _bert_predict(transcript_text: str, art: dict) -> Optional[float]:
    """Run DistilBERT inference on a single transcript. Returns None if BERT not ready."""
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
    Derive emotional arc from transcript.
    Uses VADER if available; falls back to keyword heuristic.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        turns = [t.strip() for t in transcript_text.split("|") if t.strip()]
        scores = [sia.polarity_scores(t)["compound"] for t in turns]
        if len(scores) < 4:
            return "flat"
        n = max(1, len(scores) // 5)
        first_avg = np.mean(scores[:n])
        last_avg = np.mean(scores[-n:])
        mid_min = min(scores[n:-n]) if len(scores) > 2 * n else first_avg
        delta = last_avg - first_avg
        if delta > 0.15:
            return "rise"
        if delta < -0.15:
            return "fall"
        if mid_min < first_avg - 0.15 and last_avg > first_avg - 0.05:
            return "v_shape"
        return "flat"
    except ImportError:
        # Keyword fallback
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


# ── Main predict function (Person 3 calls this) ──────────────


def predict(transcript_text: str, call_metadata: dict) -> dict:
    """
    End-to-end CSAT prediction for a single call.

    Args:
        transcript_text: raw transcript string (pipe-delimited turns)
        call_metadata:   dict — must have:
                           issue_type    (str)
                           call_duration (str: 'short'|'medium'|'long')
                           repeat_contact (int: 0 or 1)

    Returns:
        {
          "csat_score":          float (1.0–5.0),
          "confidence_interval": [lower, upper],
          "emotional_arc":       str,
          "shap_values":         {}   ← Person 3 fills this
        }
    """
    art = _load_artefacts()
    w = art["weights"]

    # 1. Extract features
    X_raw = _extract_features(transcript_text, call_metadata)
    X_raw = np.nan_to_num(X_raw, nan=0.0)  # guard against NaN from Person 1

    # 2. Scale for Ridge
    if art["scaler"] is not None:
        X_scaled = art["scaler"].transform(X_raw)
    else:
        from sklearn.preprocessing import StandardScaler

        X_scaled = X_raw  # in dummy/no-scaler mode

    # 3. Individual predictions
    ridge_pred = float(np.clip(art["ridge"].predict(X_scaled)[0], 1.0, 5.0))
    rf_pred = float(np.clip(art["rf"].predict(X_raw)[0], 1.0, 5.0))
    bert_pred = _bert_predict(transcript_text, art)

    # 4. Ensemble
    if bert_pred is not None:
        csat = w["ridge"] * ridge_pred + w["rf"] * rf_pred + w["bert"] * bert_pred
        all_preds = [ridge_pred, rf_pred, bert_pred]
    else:
        # BERT not available — redistribute its weight equally to Ridge + RF
        ridge_w = w["ridge"] / (w["ridge"] + w["rf"])
        rf_w = w["rf"] / (w["ridge"] + w["rf"])
        csat = ridge_w * ridge_pred + rf_w * rf_pred
        all_preds = [ridge_pred, rf_pred]

    csat = float(np.clip(csat, 1.0, 5.0))

    # 5. Confidence interval — ±1 std of the individual model predictions
    pred_std = float(np.std(all_preds))
    ci = [round(max(1.0, csat - pred_std), 3), round(min(5.0, csat + pred_std), 3)]

    # 6. Emotional arc
    arc = _compute_emotional_arc(transcript_text)

    return {
        "csat_score": round(csat, 3),
        "confidence_interval": ci,
        "emotional_arc": arc,
        "shap_values": {},  # Person 3 fills via shap.TreeExplainer
    }


# ── End-to-end test (5 dummy calls) ─────────────────────────


def _run_end_to_end_test():
    """
    Test predict() on 5 calls. Checks:
      - Output dict has all required keys
      - csat_score is in [1, 5]
      - confidence_interval is a 2-element list
      - Runs in < 5 seconds per call on CPU
    """
    test_calls = [
        {
            "transcript": "Turn 1: CUSTOMER: my bill is wrong | Turn 2: AGENT: I understand | Turn 3: CUSTOMER: still not fixed | Turn 4: AGENT: I apologize",
            "metadata": {
                "issue_type": "billing",
                "call_duration": "short",
                "repeat_contact": 1,
            },
            "label": "billing, repeat, short",
        },
        {
            "transcript": "Turn 1: CUSTOMER: nothing works | Turn 2: AGENT: happy to help | Turn 3: CUSTOMER: okay thank you | Turn 4: AGENT: all sorted",
            "metadata": {
                "issue_type": "technical",
                "call_duration": "medium",
                "repeat_contact": 0,
            },
            "label": "technical, no-repeat, resolved",
        },
        {
            "transcript": "Turn 1: CUSTOMER: called three times | Turn 2: AGENT: I completely understand | Turn 3: CUSTOMER: still frustrated | Turn 4: AGENT: I will escalate",
            "metadata": {
                "issue_type": "account",
                "call_duration": "long",
                "repeat_contact": 1,
            },
            "label": "account, repeat, long",
        },
        {
            "transcript": "Turn 1: CUSTOMER: thanks for the support | Turn 2: AGENT: glad to help | Turn 3: CUSTOMER: all set",
            "metadata": {
                "issue_type": "payment",
                "call_duration": "short",
                "repeat_contact": 0,
            },
            "label": "payment, positive",
        },
        {
            "transcript": "Turn 1: CUSTOMER: why is this happening again | Turn 2: AGENT: we are working on it | Turn 3: CUSTOMER: third time calling | Turn 4: AGENT: sorry for the trouble",
            "metadata": {
                "issue_type": "network",
                "call_duration": "medium",
                "repeat_contact": 1,
            },
            "label": "network, repeat, frustrated",
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

        # Assertions
        assert "csat_score" in result, "Missing csat_score"
        assert "confidence_interval" in result, "Missing confidence_interval"
        assert "emotional_arc" in result, "Missing emotional_arc"
        assert "shap_values" in result, "Missing shap_values"
        assert 1.0 <= result["csat_score"] <= 5.0, (
            f"csat_score out of range: {result['csat_score']}"
        )
        assert len(result["confidence_interval"]) == 2, "CI must be 2-element list"
        assert result["emotional_arc"] in ("rise", "fall", "flat", "v_shape"), (
            f"Invalid arc: {result['emotional_arc']}"
        )

        timing_ok = elapsed < 5.0
        if not timing_ok:
            all_passed = False
            print(f"  [FAIL] Call {i} took {elapsed:.2f}s (>5s limit)")

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
        print(
            "  Hand off: phase6_predict_fn.py + models/ + outputs/ensemble_weights.json"
        )
    else:
        print("PHASE 6: some checks failed — review output above.")
    print("=" * 60)


if __name__ == "__main__":
    _load_artefacts()
    _run_end_to_end_test()
