import pytest
from pydantic import ValidationError

from src.models import (
    ALLOWED_CASE_STATUSES,
    CaseSummary,
    ClaimFacts,
    ClaimType,
    DocumentEvidence,
    EvaluatorVerdict,
    HistoryEvidence,
    PolicyEvidence,
    RiskDecision,
    RiskRoute,
    ToolStatus,
)


def _facts(**overrides):
    data = dict(
        claim_type="auto",
        claim_amount=1500.0,
        incident_date="2026-09-01",
        description="Rear-end collision at a traffic light",
        missing_fields=[],
    )
    data.update(overrides)
    return data


def test_claim_facts_valid():
    f = ClaimFacts(**_facts())
    assert f.claim_type is ClaimType.AUTO
    assert f.claim_amount == 1500.0


def test_claim_facts_allows_null_amount_and_date():
    f = ClaimFacts(**_facts(claim_amount=None, incident_date=None,
                            missing_fields=["claim_amount", "incident_date"]))
    assert f.claim_amount is None and f.incident_date is None


def test_claim_facts_rejects_negative_amount():
    with pytest.raises(ValidationError):
        ClaimFacts(**_facts(claim_amount=-5))


def test_claim_facts_rejects_bad_date():
    with pytest.raises(ValidationError):
        ClaimFacts(**_facts(incident_date="01/09/2026"))


def test_claim_facts_rejects_unknown_claim_type():
    with pytest.raises(ValidationError):
        ClaimFacts(**_facts(claim_type="spaceship"))


def test_claim_facts_rejects_empty_description():
    with pytest.raises(ValidationError):
        ClaimFacts(**_facts(description=""))


def test_tool_status_values_are_exactly_the_agreed_four():
    assert {s.value for s in ToolStatus} == {"ok", "not_found", "unavailable", "error"}


def test_evidence_rejects_invalid_status():
    with pytest.raises(ValidationError):
        HistoryEvidence(status="maybe", claim_count=0, claims_summary=[], notes="")


def test_unavailable_history_is_not_zero_claims():
    h = HistoryEvidence(status="unavailable", claim_count=None, claims_summary=[],
                        notes="could not verify")
    assert h.status is ToolStatus.UNAVAILABLE
    assert h.claim_count is None  # unknown, not 0


def test_history_rejects_negative_count():
    with pytest.raises(ValidationError):
        HistoryEvidence(status="ok", claim_count=-1, claims_summary=[], notes="")


def test_policy_and_document_evidence_build():
    PolicyEvidence(status="ok", policy_id="POL-001", policy_status="active", is_covered=True,
                   limit_amount=10000.0, deductible=500.0, amount_after_deductible=1000.0,
                   exclusions=[], notes="")
    DocumentEvidence(status="ok", required=["photo"], provided=[], missing=["photo"], notes="")


def test_risk_decision_defaults_and_route_enum():
    r = RiskDecision(route="human_review", risk_flags=["repeated_claims"])
    assert r.route is RiskRoute.HUMAN_REVIEW
    assert RiskDecision(route="routine").risk_flags == []


def test_case_summary_requires_next_step():
    with pytest.raises(ValidationError):
        CaseSummary(summary="x", supported_facts=[], open_questions=[], recommended_next_step="")


def test_evaluator_verdict_roundtrip_json():
    v = EvaluatorVerdict(is_valid=False, issues=["promises refund"],
                         revision_instructions="Remove the refund promise")
    assert EvaluatorVerdict.model_validate_json(v.model_dump_json()) == v


def test_no_approved_or_rejected_case_status():
    assert set(ALLOWED_CASE_STATUSES) == {"pending_human_review", "needs_more_info"}


@pytest.mark.parametrize("model", [ClaimFacts, PolicyEvidence, HistoryEvidence,
                                   DocumentEvidence, CaseSummary, EvaluatorVerdict])
def test_models_compatible_with_agents_sdk_strict_schema(model):
    """Offline check that the Agents SDK accepts the model as an output_type."""
    from agents import AgentOutputSchema

    AgentOutputSchema(model, strict_json_schema=True)