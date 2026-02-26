from pydantic import BaseModel
from datetime import datetime


# --- Requests ---
class VerifyRequest(BaseModel):
    """מטאדטה שנשלחת עם הקובץ"""
    client_id: str
    context: str | None = None  # הקשר נוסף מהלקוח


class HITLApprovalRequest(BaseModel):
    preferred_domain: str | None = None  # insurance, history, forensics


# --- Responses ---
class AnomalyDetail(BaseModel):
    type: str
    description: str
    severity: str  # low, medium, high, critical
    location: dict | None = None  # מיקום באנומליה (קואורדינטות, timestamp וכו')


class AgentResultResponse(BaseModel):
    agent_type: str
    confidence_score: float
    anomalies: list[AnomalyDetail]
    heatmap_url: str | None = None


class CaseResponse(BaseModel):
    id: str
    status: str
    media_type: str
    file_hash: str
    confidence_score: float | None
    verdict: str | None
    hitl_required: bool
    hitl_recommendation: str | None = None
    agent_results: list[AgentResultResponse] = []
    created_at: datetime

    class Config:
        from_attributes = True


class VerifyResponse(BaseModel):
    case_id: str
    status: str
    message: str
