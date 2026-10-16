"""Pydantic schemas and structured-output models.

Owner: Youssef. Shared with Samir through the `status` vocabulary:
every tool result and every evidence model uses ToolStatus, so that
"unavailable" is never confused with "not_found" or a negative finding.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ClaimType(str, Enum):
    AUTO = "auto"
    HEALTH = "health"
    PROPERTY = "property"
    TRAVEL = "travel"
    OTHER = "other"


class ToolStatus(str, Enum):
    OK = "ok"
    NOT_FOUND = "not_found"      # looked, nothing exists
    UNAVAILABLE = "unavailable"  # could not check (NOT a negative finding)
    ERROR = "error"


class RiskRoute(str, Enum):
    ROUTINE = "routine"
    HUMAN_REVIEW = "human_review"


# Statuses a saved case is allowed to have. No approved/rejected value exists.
ALLOWED_CASE_STATUSES = ("pending_human_review", "needs_more_info")


# ---------------------------------------------------------------- Step 3: intake
class ClaimFacts(BaseModel):
    """Output of the Intake Agent (LLM)."""

    claim_type: ClaimType
    claim_amount: float | None = Field(
        ge=0, description="Claimed amount, or null if the customer did not state it"
    )
    incident_date: str | None = Field(
        description="Incident date as YYYY-MM-DD, or null if unknown"
    )
    description: str = Field(min_length=1, description="Short neutral summary of the incident")
    missing_fields: list[str] = Field(
        description="Facts that are missing or unclear, e.g. 'incident_date'"
    )

    @field_validator("incident_date")
    @classmethod
    def _valid_iso_date(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            _dt.date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError("incident_date must be YYYY-MM-DD") from exc
        return v


# ---------------------------------------------------------------- Step 4: routing (code)
class RiskDecision(BaseModel):
    """Built by deterministic Python (rules.py), never by an LLM."""

    route: RiskRoute
    risk_flags: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- Step 5: worker evidence
class PolicyEvidence(BaseModel):
    status: ToolStatus
    policy_id: str | None
    policy_status: str | None = Field(description="e.g. active / expired")
    is_covered: bool | None = Field(description="null when coverage could not be determined")
    limit_amount: float | None
    deductible: float | None
    amount_after_deductible: float | None
    exclusions: list[str]
    notes: str = Field(description="What was checked and what could not be verified")


class HistoryEvidence(BaseModel):
    status: ToolStatus
    claim_count: int | None = Field(ge=0, description="null when history is unavailable")
    claims_summary: list[str]
    notes: str


class DocumentEvidence(BaseModel):
    status: ToolStatus
    required: list[str]
    provided: list[str]
    missing: list[str]
    notes: str


# ---------------------------------------------------------------- Step 8: writer / evaluator
class CaseSummary(BaseModel):
    """Output of the Claim Writer (LLM)."""

    summary: str = Field(min_length=1)
    supported_facts: list[str] = Field(description="Only facts backed by claim text or tool evidence")
    open_questions: list[str] = Field(description="Missing or unverified items, incl. 'could not verify'")
    recommended_next_step: str = Field(
        min_length=1,
        description="A step for the human adjuster. Never an approval, rejection, or payout promise",
    )


class EvaluatorVerdict(BaseModel):
    """Output of the Evaluator / Checker (LLM)."""

    is_valid: bool
    issues: list[str]
    revision_instructions: str = Field(description="Empty string when is_valid is true")