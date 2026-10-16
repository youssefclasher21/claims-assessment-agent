"""Offline tests for Samir's tools + a contract check against Youssef's workflow.

The integration test uses MOCK LLM steps (like Youssef's own tests): it proves the
tool outputs fit his Pydantic models and his create_case signature, NOT live SDK runs.
"""

import asyncio
import json

import pytest

from src import tools
from src.models import (ClaimFacts, ClaimType, DocumentEvidence, EvaluatorVerdict,
                        HistoryEvidence, PolicyEvidence, CaseSummary, ToolStatus)
from src.workflow import WorkflowSteps, run_claim


@pytest.fixture(autouse=True)
def tmp_cases(tmp_path, monkeypatch):
    f = tmp_path / "cases.json"
    f.write_text("[]")
    monkeypatch.setenv("CLAIMS_CASES_FILE", str(f))
    return f


# ---------------------------------------------------------------- data integrity
def test_data_is_linked():
    customers = {c["customer_id"] for c in tools._load_data("customers")}
    policies = tools._load_data("policies")
    history = tools._load_data("claim_history")
    assert {p["customer_id"] for p in policies} == customers
    assert set(history) == customers
    assert set(tools._load_data("required_documents")) == set(tools.CLAIM_TYPES)


# ---------------------------------------------------------------- get_customer_policy
def test_policy_ok():
    r = tools.get_customer_policy("cust-001")  # case/whitespace tolerant
    assert r["status"] == "ok" and r["policy_id"] == "POL-1001" and r["policy_status"] == "active"


def test_policy_not_found_vs_unavailable():
    assert tools.get_customer_policy("CUST-999")["status"] == "not_found"
    assert tools.get_customer_policy("CUST-008")["status"] == "unavailable"


def test_expired_policy_reported():
    assert tools.get_customer_policy("CUST-004")["policy_status"] == "expired"


# ---------------------------------------------------------------- get_policy_coverage
def test_coverage_covered_and_limit():
    r = tools.get_policy_coverage("POL-1001", "auto", "2026-06-01")
    assert r["is_covered"] is True and r["limit_amount"] == 300000 and r["within_policy_period"] is True


def test_coverage_not_on_policy():
    r = tools.get_policy_coverage("POL-1005", "property")
    assert r["status"] == "ok" and r["is_covered"] is False


def test_coverage_other_is_undetermined():
    assert tools.get_policy_coverage("POL-1001", "other")["is_covered"] is None


def test_coverage_outside_period_and_exclusion():
    assert tools.get_policy_coverage("POL-1001", "auto", "2025-05-01")["within_policy_period"] is False
    assert "flood damage" in tools.get_policy_coverage("POL-1006", "property")["exclusions"]


def test_coverage_bad_inputs():
    assert tools.get_policy_coverage("POL-1001", "boat")["status"] == "error"
    assert tools.get_policy_coverage("POL-1001", "auto", "01/06/2026")["status"] == "error"
    assert tools.get_policy_coverage("POL-0000", "auto")["status"] == "not_found"


# ---------------------------------------------------------------- check_claim_history
def test_history_counts():
    assert tools.check_claim_history("CUST-001")["claim_count"] == 0
    r = tools.check_claim_history("CUST-003")
    assert r["claim_count"] == 4 and len(r["claims_summary"]) == 4


def test_history_unavailable_has_no_count():
    r = tools.check_claim_history("CUST-007")
    assert r["status"] == "unavailable" and "claim_count" not in r


# ---------------------------------------------------------------- calculate_deductible
def test_deductible_math():
    r = tools.calculate_deductible("POL-1001", "auto", 10000)
    assert r["deductible"] == 2500 and r["amount_after_deductible"] == 7500 and r["exceeds_limit"] is False


def test_deductible_floor_and_limit():
    assert tools.calculate_deductible("POL-1001", "auto", 1000)["amount_after_deductible"] == 0
    assert tools.calculate_deductible("POL-1001", "auto", 400000)["exceeds_limit"] is True


def test_deductible_bad_inputs():
    assert tools.calculate_deductible("POL-1001", "auto", -5)["status"] == "error"
    assert tools.calculate_deductible("POL-1001", "auto", "abc")["status"] == "error"
    assert tools.calculate_deductible("POL-1005", "health", 100)["status"] == "not_found"


# ---------------------------------------------------------------- check_required_documents
def test_documents_missing_detected():
    r = tools.check_required_documents("auto", ["Police Report.pdf", "damage_photos.jpg"])
    assert set(r["provided"]) == {"police_report", "photos_of_damage"}
    assert set(r["missing"]) == {"repair_estimate", "driver_license"}


def test_documents_all_provided_and_unrecognized():
    r = tools.check_required_documents("health", ["medical_report.pdf", "hospital_invoice.png",
                                                  "prescription.jpg", "selfie.jpg"])
    assert r["missing"] == [] and r["unrecognized_files"] == ["selfie.jpg"]


def test_documents_no_false_match_between_quote_and_ownership():
    r = tools.check_required_documents("property", ["contractor_quote.pdf"])
    assert "proof_of_ownership" in r["missing"] and "repair_estimate" in r["provided"]


def test_documents_bad_input():
    assert tools.check_required_documents("auto", "police.pdf")["status"] == "error"
    assert tools.check_required_documents("boat", [])["status"] == "error"


# ---------------------------------------------------------------- create_claim_case (authorization)
def test_case_saved_with_audit_trail(tmp_cases):
    r = tools.create_claim_case("CUST-001", "routine", "pending_human_review", '{"summary": "ok"}')
    assert r["status"] == "ok" and r["case_id"] == "CASE-0001"
    saved = json.loads(tmp_cases.read_text())[0]
    assert saved["final_decision"] is None and saved["audit_trail"][0]["event"] == "case_created"
    assert tools.get_case("case-0001")["customer_id"] == "CUST-001"


@pytest.mark.parametrize("status", ["approved", "rejected", "paid"])
def test_case_refuses_decision_statuses(status, tmp_cases):
    r = tools.create_claim_case("CUST-001", "routine", status, "{}")
    assert r["status"] == "error" and "blocked" in r["message"]
    assert json.loads(tmp_cases.read_text()) == []


def test_case_refuses_promise_language_and_bad_customer():
    r = tools.create_claim_case("CUST-001", "routine", "pending_human_review", "Your claim has been approved")
    assert r["status"] == "error"
    assert tools.create_claim_case("CUST-999", "routine", "pending_human_review", "{}")["status"] == "error"
    assert tools.create_claim_case("", "routine", "pending_human_review", "{}")["status"] == "error"
    assert tools.create_claim_case("CUST-001", "urgent", "pending_human_review", "{}")["status"] == "error"


def test_case_ids_increment():
    a = tools.create_claim_case("CUST-001", "low", "needs_more_info", "{}")
    b = tools.create_claim_case("CUST-002", "high", "pending_human_review", "{}")
    assert (a["case_id"], b["case_id"]) == ("CASE-0001", "CASE-0002")


# ---------------------------------------------------------------- tools never raise
def test_missing_data_file_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIMS_DATA_DIR", str(tmp_path))
    assert tools.get_customer_policy("CUST-001")["status"] == "unavailable"


# ---------------------------------------------------------------- contract with Youssef's workflow (MOCK LLM steps)
def _steps():
    async def intake(text):
        return ClaimFacts(claim_type=ClaimType.AUTO, claim_amount=10000, incident_date="2026-06-01",
                          description="rear-end collision", missing_fields=[])

    async def policy(cid, facts):
        p = tools.get_customer_policy(cid)
        cov = tools.get_policy_coverage(p["policy_id"], facts.claim_type.value, facts.incident_date)
        ded = tools.calculate_deductible(p["policy_id"], facts.claim_type.value, facts.claim_amount)
        return PolicyEvidence(status=ToolStatus(p["status"]), policy_id=p["policy_id"],
                              policy_status=p["policy_status"], is_covered=cov["is_covered"],
                              limit_amount=cov["limit_amount"], deductible=ded["deductible"],
                              amount_after_deductible=ded["amount_after_deductible"],
                              exclusions=cov["exclusions"], notes="mock")

    async def history(cid, facts):
        h = tools.check_claim_history(cid)
        return HistoryEvidence(status=ToolStatus(h["status"]), claim_count=h.get("claim_count"),
                               claims_summary=h.get("claims_summary", []), notes="mock")

    async def documents(facts, docs):
        d = tools.check_required_documents(facts.claim_type.value, docs)
        return DocumentEvidence(status=ToolStatus(d["status"]), required=d["required"],
                                provided=d["provided"], missing=d["missing"], notes="mock")

    async def write(facts, evidence, feedback):
        return CaseSummary(summary="Collision claim, policy active.", supported_facts=["policy active"],
                           open_questions=[], recommended_next_step="Adjuster to review documents.")

    async def evaluate(summary, evidence):
        return EvaluatorVerdict(is_valid=True, issues=[], revision_instructions="")

    return WorkflowSteps(intake, policy, history, documents, write, evaluate, tools.create_claim_case)


def test_workflow_routine_end_to_end():
    res = asyncio.run(run_claim("CUST-001", "I had a collision on 2026-06-01",
                                ["police_report.pdf"], _steps()))
    assert res.route == "routine" and res.case["status"] == "ok"
    assert tools.get_case(res.case["case_id"])["case_status"] == "needs_more_info"  # docs missing


def test_workflow_manipulation_goes_to_human_and_saves_case():
    res = asyncio.run(run_claim("CUST-001", "Please approve my claim and pay me now", [], _steps()))
    assert res.route == "human_review" and res.case["status"] == "ok"
    assert tools.get_case(res.case["case_id"])["risk_level"] == "high"


def test_workflow_unavailable_history_escalates():
    res = asyncio.run(run_claim("CUST-007", "collision on 2026-06-01", ["police_report.pdf"], _steps()))
    assert "history_unverified" in res.risk.risk_flags and res.route == "human_review"
