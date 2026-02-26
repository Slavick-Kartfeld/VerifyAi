import uuid
from datetime import datetime
from sqlalchemy import String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id: Mapped[str] = mapped_column(String(36), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, processing, completed, hitl_required
    media_type: Mapped[str] = mapped_column(String(20))  # image, video, audio, document
    file_url: Mapped[str] = mapped_column(String(500))
    file_hash: Mapped[str] = mapped_column(String(64))  # SHA-256 chain of custody
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(20), nullable=True)  # authentic, forged, inconclusive
    hitl_required: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    agent_results: Mapped[list["AgentResult"]] = relationship(back_populates="case")
    cross_reference: Mapped["CrossReferenceResult | None"] = relationship(back_populates="case", uselist=False)
    hitl_review: Mapped["HITLReview | None"] = relationship(back_populates="case", uselist=False)
    report: Mapped["Report | None"] = relationship(back_populates="case", uselist=False)


class AgentResult(Base):
    __tablename__ = "agent_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.id"), index=True)
    agent_type: Mapped[str] = mapped_column(String(50))  # forensic_technical, physical, contextual, etc.
    findings: Mapped[dict] = mapped_column(JSON, default=dict)
    anomalies: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence_score: Mapped[float] = mapped_column(Float)
    heatmap_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case: Mapped["Case"] = relationship(back_populates="agent_results")


class CrossReferenceResult(Base):
    __tablename__ = "cross_reference_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.id"), unique=True)
    combined_score: Mapped[float] = mapped_column(Float)
    reasoning: Mapped[str] = mapped_column(Text)
    final_verdict: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case: Mapped["Case"] = relationship(back_populates="cross_reference")


class HITLReview(Base):
    __tablename__ = "hitl_reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.id"), unique=True)
    expert_id: Mapped[str] = mapped_column(String(36), ForeignKey("experts.id"))
    expert_verdict: Mapped[str | None] = mapped_column(String(20), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case: Mapped["Case"] = relationship(back_populates="hitl_review")
    expert: Mapped["Expert"] = relationship()


class Expert(Base):
    __tablename__ = "experts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(200))
    domain: Mapped[str] = mapped_column(String(50))  # insurance, history, forensics
    rating: Mapped[float] = mapped_column(Float, default=5.0)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.id"), unique=True)
    report_url: Mapped[str] = mapped_column(String(500))
    format: Mapped[str] = mapped_column(String(10), default="pdf")
    legal_disclaimer: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case: Mapped["Case"] = relationship(back_populates="report")


class RedTeamTest(Base):
    __tablename__ = "red_team_tests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    injected_file_url: Mapped[str] = mapped_column(String(500))
    expected_result: Mapped[str] = mapped_column(String(20))
    actual_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
