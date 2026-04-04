from pydantic import BaseModel
from typing import List, Dict, Tuple


class CallMetadata(BaseModel):
    issue_type: str
    call_duration: str
    repeat_contact: int
    resolution_status: str   # Added by Person 2


class PredictRequest(BaseModel):
    transcript: str
    call_metadata: CallMetadata


class PredictResponse(BaseModel):
    csat_score: float
    confidence_interval: Tuple[float, float]
    emotional_arc: str

    top_positive_phrases: List[str]
    top_negative_phrases: List[str]

    coaching_summary: str

    shap_features: Dict[str, float]

    aggregate_stats: Dict