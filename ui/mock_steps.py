"""MOCK workflow steps so the UI and demo run without an OpenAI API key.

Only the LLM steps are simulated here (intake, writer, evaluator are keyword/template
based). The tools, data, risk router, guardrails and case authorization are all REAL.
The UI shows a MOCK banner whenever these steps are used.
"""

from __future__ import annotations

import datetime as _dt
import re

from src import tools
from src.models import (CaseSummary, ClaimFacts, ClaimType, DocumentEvidence,
                        EvaluatorVerdict, HistoryEvidence, PolicyEvidence, ToolStatus)
from src.rules import find_forbidden_language
from src.workflow import WorkflowSteps

_KEYWORDS = {
    ClaimType.AUTO: ["car", "vehicle", "collision", "accident", "crash", "bumper", "عربية", "حادث", "سيارة"],
    ClaimType.HEALTH: ["hospital", "doctor", "surgery", "medical", "treatment", "illness", "مستشفى", "علاج", "عملية"],
    ClaimType.PROPERTY: ["house", "home", "apartment", "flat", "fire", "flood", "burglary", "pipe", "water", "شقة", "حريق"],
    ClaimType.TRAVEL: ["flight", "trip", "luggage", "baggage", "hotel", "airline", "visa", "رحلة", "طيران", "شنطة"],
}
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_AMOUNT_RE = re.compile(r"(?<![\d-])(\d[\d,]*(?:\.\d+)?)(?![\d-])")


def _status(value: str) -> ToolStatus:
    return ToolStatus(value) if value in {s.value for s in ToolStatus} else ToolStatus.ERROR


async def intake(text: str) -> ClaimFacts:
    lowered = text.lower()
    ctype = next((t for t, words in _KEYWORDS.items() if any(w in lowered for w in words)), ClaimType.OTHER)
    m = _DATE_RE.search(text)
    date = m.group(0) if m else None
    if date:
        try:
            _dt.date.fromisoformat(date)
        except ValueError:
            date = None
    amounts = [float(m.replace(",", "")) for m in _AMOUNT_RE.findall(_DATE_RE.sub(" ", text))]
    amount = max(amounts) if amounts else None
    missing = [name for name, v in (("claim_amount", amount), ("incident_date", date)) if v is None]
    return ClaimFacts(claim_type=ctype, claim_amount=amount, incident_date=date,
                      description=" ".join(text.split())[:200] or "(empty claim)", missing_fields=missing)


async def policy(customer_id: str, facts: ClaimFacts) -> PolicyEvidence:
    p = tools.get_customer_policy(customer_id)
    if p["status"] != "ok":
        return PolicyEvidence(status=_status(p["status"]), policy_id=None, policy_status=None, is_covered=None,
                              limit_amount=None, deductible=None, amount_after_deductible=None, exclusions=[],
                              notes=f"could not verify policy: {p.get('message', p['status'])}")
    ctype = facts.claim_type.value
    cov = tools.get_policy_coverage(p["policy_id"], ctype, facts.incident_date)
    ded = (tools.calculate_deductible(p["policy_id"], ctype, facts.claim_amount)
           if facts.claim_amount is not None else {"status": "error", "message": "claim amount not stated"})
    notes = [cov.get("message", "")]
    if cov.get("within_policy_period") is False:
        notes.append("incident date is OUTSIDE the policy period")
    if ded["status"] != "ok":
        notes.append(f"deductible not calculated: {ded.get('message', '')}")
    elif ded["exceeds_limit"]:
        notes.append("claimed amount exceeds the policy limit")
    return PolicyEvidence(
        status=ToolStatus.OK, policy_id=p["policy_id"], policy_status=p["policy_status"],
        is_covered=cov.get("is_covered"), limit_amount=cov.get("limit_amount"),
        deductible=ded.get("deductible"), amount_after_deductible=ded.get("amount_after_deductible"),
        exclusions=cov.get("exclusions", []), notes="; ".join(n for n in notes if n))


async def history(customer_id: str, facts: ClaimFacts) -> HistoryEvidence:
    h = tools.check_claim_history(customer_id)
    if h["status"] != "ok":
        return HistoryEvidence(status=_status(h["status"]), claim_count=None, claims_summary=[],
                               notes=f"could not verify history: {h.get('message', h['status'])}")
    return HistoryEvidence(status=ToolStatus.OK, claim_count=h["claim_count"],
                           claims_summary=h["claims_summary"], notes=f"{h['claim_count']} prior claim(s) on record")


async def documents(facts: ClaimFacts, provided: list[str]) -> DocumentEvidence:
    d = tools.check_required_documents(facts.claim_type.value, provided)
    if d["status"] != "ok":
        return DocumentEvidence(status=_status(d["status"]), required=[], provided=[], missing=[],
                                notes=f"could not verify documents: {d.get('message', d['status'])}")
    note = f"{len(d['missing'])} required document(s) missing"
    if d["unrecognized_files"]:
        note += f"; unrecognized files: {', '.join(d['unrecognized_files'])}"
    return DocumentEvidence(status=ToolStatus.OK, required=d["required"], provided=d["provided"],
                            missing=d["missing"], notes=note)


async def write(facts: ClaimFacts, evidence: dict, feedback: str | None) -> CaseSummary:
    pol, hist, docs = evidence["policy"], evidence["history"], evidence["documents"]
    facts_list = [f"Claim type: {facts.claim_type.value}"]
    open_q = [f"Missing from claim text: {m}" for m in facts.missing_fields]
    if facts.claim_amount is not None:
        facts_list.append(f"Claimed amount: {facts.claim_amount:,.0f} EGP")
    if facts.incident_date:
        facts_list.append(f"Incident date: {facts.incident_date}")

    if pol.status is ToolStatus.OK:
        facts_list.append(f"Policy {pol.policy_id} is {pol.policy_status}")
        if pol.is_covered is True:
            facts_list.append(f"Claim type is listed on the policy (limit {pol.limit_amount:,.0f} EGP)")
        elif pol.is_covered is False:
            open_q.append("Claim type is NOT listed on this policy: adjuster to confirm")
        else:
            open_q.append("Coverage could not be determined automatically")
        if pol.amount_after_deductible is not None:
            facts_list.append(f"Amount after deductible ({pol.deductible:,.0f} EGP): {pol.amount_after_deductible:,.0f} EGP (arithmetic only)")
        if "OUTSIDE" in pol.notes:
            open_q.append("Incident date is outside the policy period")
    else:
        open_q.append("Could not verify the policy")

    if hist.status is ToolStatus.OK:
        facts_list.append(f"Prior claims on record: {hist.claim_count}")
    else:
        open_q.append("Could not verify the claim history")

    if docs.status is ToolStatus.OK:
        if docs.missing:
            open_q.append("Missing documents: " + ", ".join(docs.missing))
        else:
            facts_list.append("All required documents were provided")
    else:
        open_q.append("Could not verify the documents")

    summary = (f"{facts.claim_type.value.title()} claim. " + "; ".join(facts_list[1:]) + "."
               + (f" Open items: {len(open_q)}." if open_q else ""))
    step = "Adjuster to review the open items and decide." if open_q else "Adjuster to review the evidence and decide."
    return CaseSummary(summary=summary, supported_facts=facts_list, open_questions=open_q, recommended_next_step=step)


async def evaluate(summary: CaseSummary, evidence: dict) -> EvaluatorVerdict:
    hits = find_forbidden_language(summary.summary + " " + summary.recommended_next_step)
    return EvaluatorVerdict(is_valid=not hits, issues=[f"forbidden language: {h}" for h in hits],
                            revision_instructions="Remove approval/refund language." if hits else "")


def build_mock_steps() -> WorkflowSteps:
    return WorkflowSteps(intake, policy, history, documents, write, evaluate, tools.create_claim_case)
