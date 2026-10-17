"""Deterministic Python rules: risk routing, language checks, retry limits.

Owner: Youssef. Nothing here calls an LLM. This is the "Python decides what
is allowed" layer from the brief (code-orchestrated decisions).
"""

from __future__ import annotations

import os
import re

from src.models import (
    ALLOWED_CASE_STATUSES,
    ClaimFacts,
    HistoryEvidence,
    PolicyEvidence,
    RiskDecision,
    RiskRoute,
    ToolStatus,
)

# ------------------------------------------------------------------ constants
HIGH_AMOUNT_THRESHOLD = 50_000.0      # claims above this go to a human
REPEATED_CLAIMS_THRESHOLD = 3         # this many prior claims = suspicious


def get_max_revisions() -> int:
    """Max evaluator-optimizer revisions. Read from env, bounded to 1..5."""
    try:
        value = int(os.getenv("MAX_REVISIONS", "2"))
    except ValueError:
        value = 2
    return max(1, min(value, 5))


# ------------------------------------------------------------------ language checks
# Phrases the system must never output: it prepares an assessment, it never
# approves, rejects, or promises money.
_FORBIDDEN_PATTERNS = [
    r"\b(claim|request)\s+(is|has been|was|will be)\s+(approved|accepted|rejected|denied|declined)\b",
    r"\bwe\s+(have\s+)?(approve|approved|reject|rejected|deny|denied)\b",
    r"\b(you|your)\s+(will|'ll)\s+(be\s+)?(refund|reimburs|paid|compensat)\w*",
    r"\byou\s+will\s+receive\b",
    r"\b(guarantee|guaranteed|promise|promised)\b",
    r"\b(full|complete)\s+(refund|payout|payment|coverage)\b",
    r"\b(approved|rejected)\s+for\s+payment\b",
    r"\bpayout\s+(of|is|will)\b",
    r"\bpayable\b",
    r"\bwill\s+be\s+(paid|refunded|reimbursed|compensated)\b",
    r"\bleaving\s+[$]?\s*[\d,]+(?:\.\d+)?\s*(?:EGP|USD|dollars)?\s+(?:after|payable)\b",
]
_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in _FORBIDDEN_PATTERNS]


def find_forbidden_language(text: str) -> list[str]:
    """Return the matched forbidden snippets (empty list = clean)."""
    if not text:
        return []
    hits: list[str] = []
    for rx in _FORBIDDEN_RE:
        m = rx.search(text)
        if m:
            hits.append(m.group(0))
    return hits


# Input-side manipulation / unauthorized-request patterns (code layer of the
# input guardrail; the LLM classifier is added on top later).
_MANIPULATION_PATTERNS = [
    r"\bmark\s+(my|the|this)\s+claim\s+(as\s+)?(approved|accepted|paid)\b",
    r"\bapprove\s+(my|this|the)\s+claim\b",
    r"\b(change|update|modify|increase|extend|cancel)\s+(my|the)\s+(policy|coverage|deductible|limit)\b",
    r"\bignore\s+(all\s+|any\s+)?(previous|prior|above)\s+(instructions|rules)\b",
    r"\bdisregard\s+(all\s+|your\s+)?(instructions|rules|guidelines)\b",
    r"\bsystem\s+prompt\b",
    r"\byou\s+are\s+now\b",
    r"\bpay\s+me\b",
    r"\bset\s+(the\s+)?(status|decision)\s+to\b",
]
_MANIPULATION_RE = [re.compile(p, re.IGNORECASE) for p in _MANIPULATION_PATTERNS]


def detect_manipulation(text: str) -> list[str]:
    """Return matched manipulation snippets (empty list = clean)."""
    if not text:
        return []
    return [m.group(0) for rx in _MANIPULATION_RE if (m := rx.search(text))]


# ------------------------------------------------------------------ risk router
def decide_risk_route(
    facts: ClaimFacts,
    history: HistoryEvidence | None = None,
    policy: PolicyEvidence | None = None,
    manipulation_hits: list[str] | None = None,
) -> RiskDecision:
    """Deterministic routine vs human_review decision.

    Any single high-risk signal routes to human review. Unavailable evidence
    is never treated as a clean result: it is recorded as a flag too.
    """
    flags: list[str] = []
    reasons: list[str] = []

    if manipulation_hits:
        flags.append("manipulation_attempt")
        reasons.append(f"Input contained manipulation language: {manipulation_hits}")

    if facts.claim_amount is not None and facts.claim_amount > HIGH_AMOUNT_THRESHOLD:
        flags.append("high_amount")
        reasons.append(
            f"Claim amount {facts.claim_amount:,.2f} exceeds {HIGH_AMOUNT_THRESHOLD:,.0f}"
        )

    if history is not None:
        if history.status is ToolStatus.OK:
            if (history.claim_count or 0) >= REPEATED_CLAIMS_THRESHOLD:
                flags.append("repeated_claims")
                reasons.append(f"{history.claim_count} prior claims on record")
        elif history.status in (ToolStatus.UNAVAILABLE, ToolStatus.ERROR):
            flags.append("history_unverified")
            reasons.append("Claim history could not be verified")

    if policy is not None:
        if policy.status in (ToolStatus.UNAVAILABLE, ToolStatus.ERROR):
            flags.append("policy_unverified")
            reasons.append("Policy could not be verified")
        elif policy.status is ToolStatus.OK and policy.policy_status not in (None, "active"):
            flags.append("policy_not_active")
            reasons.append(f"Policy status is '{policy.policy_status}'")

    route = RiskRoute.HUMAN_REVIEW if flags else RiskRoute.ROUTINE
    return RiskDecision(route=route, risk_flags=flags, reasons=reasons)


def pre_route_checks(facts: ClaimFacts, claim_text: str) -> RiskDecision:
    """Step 4 of the flow: runs right after intake, before any worker."""
    return decide_risk_route(facts, manipulation_hits=detect_manipulation(claim_text))


# ------------------------------------------------------------------ authorization
def validate_case_payload(customer_id: str, case_status: str, assessment_text: str) -> list[str]:
    """Code-side authorization for create_claim_case. Returns a list of
    refusal reasons; an empty list means the call is allowed."""
    reasons: list[str] = []
    if case_status not in ALLOWED_CASE_STATUSES:
        reasons.append(f"case_status '{case_status}' is not allowed")
    if not customer_id or not customer_id.strip():
        reasons.append("customer_id is empty")
    hits = find_forbidden_language(assessment_text)
    if hits:
        reasons.append(f"assessment contains approval/rejection/payout language: {hits}")
    return reasons


# ------------------------------------------------------------------ evaluator loop bound
def should_continue_revising(revisions_done: int, is_valid: bool) -> bool:
    """Stop when valid or when the maximum revision count is reached."""
    if is_valid:
        return False
    return revisions_done < get_max_revisions()