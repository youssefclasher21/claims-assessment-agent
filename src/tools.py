"""Domain tools for the claims agent. Owner: Samir.

Six tools, fixed interfaces. Every result is a plain dict with a ``status``
key that uses Youssef's ToolStatus vocabulary (src/models.py):

    ok           the check ran and returned data
    not_found    we looked and nothing exists (a real negative finding)
    unavailable  we could NOT check (never a negative finding)
    error        bad input, or the call was refused

Tools never raise. A crash inside a tool becomes status "error".
All data is synthetic and lives in data/*.json.

Agents SDK wiring (for Youssef):
    from src.tools import build_agent_tools
    t = build_agent_tools()
    build_policy_worker(t["policy"])
    build_history_worker(t["history"])
    build_documentation_worker(t["documents"])
``create_claim_case`` is deliberately NOT given to any LLM agent: the
workflow calls it directly, and it enforces src/rules.py authorization.
"""

from __future__ import annotations

import datetime as _dt
import functools
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from src.models import ToolStatus
from src.rules import validate_case_payload

CLAIM_TYPES = ("auto", "health", "property", "travel", "other")
RISK_LEVELS = ("low", "medium", "high", "routine", "human_review")
CURRENCY = "EGP"

_CASES_LOCK = threading.Lock()


class _DataUnavailable(Exception):
    """A data file is missing or unreadable -> status 'unavailable'."""


# ------------------------------------------------------------------ helpers
def _data_dir() -> Path:
    return Path(os.getenv("CLAIMS_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")


def _cases_path() -> Path:
    return Path(os.getenv("CLAIMS_CASES_FILE") or _data_dir() / "cases.json")


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _DataUnavailable(f"{path.name} could not be read") from exc


def _load_data(name: str) -> Any:
    return _load(_data_dir() / f"{name}.json")


def _result(status: ToolStatus, **fields: Any) -> dict:
    return {"status": status.value, **fields}


def _norm_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _norm_claim_type(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_tool(fn):
    """Tools never raise: unexpected failures become status 'error'."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _DataUnavailable as exc:
            return _result(ToolStatus.UNAVAILABLE, message=f"could not verify: {exc}")
        except Exception as exc:  # noqa: BLE001 - tools must not crash the workflow
            return _result(ToolStatus.ERROR, message=f"{type(exc).__name__}: {exc}")

    return wrapper


def _customer(customer_id: str) -> dict | None:
    cid = _norm_id(customer_id)
    return next((c for c in _load_data("customers") if c["customer_id"] == cid), None)


def _policy_by_id(policy_id: str) -> dict | None:
    pid = _norm_id(policy_id)
    return next((p for p in _load_data("policies") if p["policy_id"] == pid), None)


def _bad_claim_type(claim_type: str) -> dict | None:
    if _norm_claim_type(claim_type) not in CLAIM_TYPES:
        return _result(
            ToolStatus.ERROR,
            message=f"unknown claim_type '{claim_type}'. Allowed: {', '.join(CLAIM_TYPES)}",
        )
    return None


# ------------------------------------------------------------------ tool 1
@_safe_tool
def get_customer_policy(customer_id: str) -> dict:
    """Look up the customer's insurance policy (id, type, status, validity dates).

    Args:
        customer_id: The customer id, e.g. "CUST-001".
    """
    customer = _customer(customer_id)
    if customer is None:
        return _result(ToolStatus.NOT_FOUND, customer_id=_norm_id(customer_id),
                       message="no customer with this id")
    if "policy" in customer.get("simulate_unavailable", []):
        return _result(ToolStatus.UNAVAILABLE, customer_id=customer["customer_id"],
                       message="could not verify: policy system is unavailable")
    policy = next((p for p in _load_data("policies")
                   if p["customer_id"] == customer["customer_id"]), None)
    if policy is None:
        return _result(ToolStatus.NOT_FOUND, customer_id=customer["customer_id"],
                       message="customer has no policy on record")
    return _result(
        ToolStatus.OK,
        customer_id=customer["customer_id"],
        customer_name=customer["name"],
        policy_id=policy["policy_id"],
        policy_type=policy["policy_type"],
        policy_status=policy["status"],
        start_date=policy["start_date"],
        end_date=policy["end_date"],
    )


# ------------------------------------------------------------------ tool 2
@_safe_tool
def get_policy_coverage(policy_id: str, claim_type: str, incident_date: str | None = None) -> dict:
    """Check whether a policy covers a claim type, its limit and its exclusions.

    Args:
        policy_id: The policy id, e.g. "POL-1001".
        claim_type: One of auto, health, property, travel, other.
        incident_date: Optional incident date YYYY-MM-DD, to check it falls inside the policy period.
    """
    bad = _bad_claim_type(claim_type)
    if bad:
        return bad
    ctype = _norm_claim_type(claim_type)
    policy = _policy_by_id(policy_id)
    if policy is None:
        return _result(ToolStatus.NOT_FOUND, policy_id=_norm_id(policy_id), message="no policy with this id")

    within = None
    if incident_date:
        try:
            d = _dt.date.fromisoformat(incident_date)
        except ValueError:
            return _result(ToolStatus.ERROR, message="incident_date must be YYYY-MM-DD")
        within = _dt.date.fromisoformat(policy["start_date"]) <= d <= _dt.date.fromisoformat(policy["end_date"])

    base = dict(policy_id=policy["policy_id"], claim_type=ctype,
                policy_status=policy["status"], within_policy_period=within)

    if ctype == "other":
        return _result(ToolStatus.OK, **base, is_covered=None, limit_amount=None, exclusions=[],
                       message="claim type 'other' cannot be checked automatically; needs manual coverage review")
    cov = policy["coverages"].get(ctype)
    if cov is None:
        return _result(ToolStatus.OK, **base, is_covered=False, limit_amount=0.0, exclusions=[],
                       message=f"this policy ({policy['policy_type']}) does not include {ctype} coverage")
    return _result(ToolStatus.OK, **base, is_covered=bool(cov["covered"]),
                   limit_amount=float(cov["limit_amount"]), exclusions=list(cov["exclusions"]),
                   message="coverage terms retrieved")


# ------------------------------------------------------------------ tool 3
@_safe_tool
def check_claim_history(customer_id: str) -> dict:
    """List the customer's prior claims and how many there are.

    Args:
        customer_id: The customer id, e.g. "CUST-001".
    """
    customer = _customer(customer_id)
    if customer is None:
        return _result(ToolStatus.NOT_FOUND, customer_id=_norm_id(customer_id),
                       message="no customer with this id")
    if "history" in customer.get("simulate_unavailable", []):
        return _result(ToolStatus.UNAVAILABLE, customer_id=customer["customer_id"],
                       message="could not verify: claims history system is unavailable")
    claims = _load_data("claim_history").get(customer["customer_id"], [])
    summary = [f"{c['claim_id']}: {c['claim_type']}, {c['amount']:,.0f} {CURRENCY} on {c['date']} ({c['outcome']})"
               for c in claims]
    return _result(ToolStatus.OK, customer_id=customer["customer_id"], claim_count=len(claims),
                   claims_summary=summary, claims=claims)


# ------------------------------------------------------------------ tool 4
@_safe_tool
def calculate_deductible(policy_id: str, claim_type: str, claim_amount: float) -> dict:
    """Calculate the deductible and the amount left after it. Arithmetic only, not a payout decision.

    Args:
        policy_id: The policy id, e.g. "POL-1001".
        claim_type: One of auto, health, property, travel, other.
        claim_amount: The claimed amount in EGP (must be 0 or more).
    """
    bad = _bad_claim_type(claim_type)
    if bad:
        return bad
    try:
        amount = float(claim_amount)
    except (TypeError, ValueError):
        return _result(ToolStatus.ERROR, message="claim_amount must be a number")
    if amount < 0:
        return _result(ToolStatus.ERROR, message="claim_amount must be 0 or more")
    ctype = _norm_claim_type(claim_type)
    policy = _policy_by_id(policy_id)
    if policy is None:
        return _result(ToolStatus.NOT_FOUND, policy_id=_norm_id(policy_id), message="no policy with this id")
    cov = policy["coverages"].get(ctype)
    if cov is None:
        return _result(ToolStatus.NOT_FOUND, policy_id=policy["policy_id"], claim_type=ctype,
                       message="no deductible defined: this claim type is not on the policy")
    deductible = float(cov["deductible"])
    return _result(
        ToolStatus.OK, policy_id=policy["policy_id"], claim_type=ctype, currency=CURRENCY,
        claim_amount=amount, deductible=deductible,
        amount_after_deductible=round(max(amount - deductible, 0.0), 2),
        limit_amount=float(cov["limit_amount"]), exceeds_limit=amount > float(cov["limit_amount"]),
        message="arithmetic only; not a payout estimate or approval",
    )


# ------------------------------------------------------------------ tool 5
def _normalize_doc(name: str) -> str:
    base = str(name).replace("\\", "/").split("/")[-1].lower()
    base = re.sub(r"\.[a-z0-9]{2,5}$", "", base)
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")


@_safe_tool
def check_required_documents(claim_type: str, provided_documents: list[str]) -> dict:
    """Compare the documents the customer submitted with the documents required for the claim type.

    Args:
        claim_type: One of auto, health, property, travel, other.
        provided_documents: Names or filenames of submitted documents, e.g. ["police_report.pdf"].
    """
    bad = _bad_claim_type(claim_type)
    if bad:
        return bad
    if not isinstance(provided_documents, list) or not all(isinstance(x, str) for x in provided_documents):
        return _result(ToolStatus.ERROR, message="provided_documents must be a list of strings")
    ctype = _norm_claim_type(claim_type)
    spec = _load_data("required_documents")[ctype]
    normalized = {f: _normalize_doc(f) for f in provided_documents}

    provided, missing, used = [], [], set()
    for doc in spec:
        keys = [doc["name"], *doc.get("aliases", [])]
        hit = next((f for f, n in normalized.items() if any(k in n for k in keys)), None)
        if hit is None:
            missing.append(doc["name"])
        else:
            provided.append(doc["name"])
            used.add(hit)
    return _result(
        ToolStatus.OK, claim_type=ctype, required=[d["name"] for d in spec],
        provided=provided, missing=missing,
        unrecognized_files=[f for f in provided_documents if f not in used],
    )


# ------------------------------------------------------------------ tool 6
@_safe_tool
def create_claim_case(customer_id: str, risk_level: str, case_status: str, assessment_json: str) -> dict:
    """Save a claim case for human review. Code-side authorization (src/rules.py) is enforced here.

    Args:
        customer_id: The customer id, e.g. "CUST-001".
        risk_level: One of low, medium, high, routine, human_review.
        case_status: pending_human_review or needs_more_info. Approval/rejection is not allowed.
        assessment_json: The assessment (risk decision or case summary) as JSON text.
    """
    cid = _norm_id(customer_id)
    reasons = validate_case_payload(cid, case_status, assessment_json)
    if str(risk_level).strip().lower() not in RISK_LEVELS:
        reasons.append(f"risk_level '{risk_level}' is not allowed")
    if not str(assessment_json or "").strip():
        reasons.append("assessment is empty")
    if cid and _customer(cid) is None:
        reasons.append(f"unknown customer_id '{cid}'")
    if reasons:
        return _result(ToolStatus.ERROR, message="blocked: " + "; ".join(reasons), reasons=reasons)

    try:
        assessment: Any = json.loads(assessment_json)
    except json.JSONDecodeError:
        assessment = assessment_json

    path = _cases_path()
    with _CASES_LOCK:
        cases = _load(path) if path.exists() else []
        case_id = f"CASE-{len(cases) + 1:04d}"
        now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        cases.append({
            "case_id": case_id, "customer_id": cid,
            "risk_level": str(risk_level).strip().lower(), "case_status": case_status,
            "assessment": assessment, "created_at": now,
            "final_decision": None, "decision_owner": "human_adjuster",
            "audit_trail": [{"timestamp": now, "actor": "claims_agent", "event": "case_created",
                             "detail": f"status={case_status}, risk={risk_level}"}],
        })
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    return _result(ToolStatus.OK, case_id=case_id, case_status=case_status,
                   message="case saved and waiting for a human adjuster")


# ------------------------------------------------------------------ read helpers (UI, not agent tools)
def list_cases() -> list[dict]:
    path = _cases_path()
    return _load(path) if path.exists() else []


def get_case(case_id: str) -> dict | None:
    return next((c for c in list_cases() if c["case_id"] == _norm_id(case_id)), None)


# ------------------------------------------------------------------ human-only action (NOT an agent tool)
HUMAN_DECISIONS = ("approved", "rejected", "needs_more_info")


def record_human_decision(case_id: str, decision: str, adjuster: str, note: str = "") -> dict:
    """Record the human adjuster's final decision on a case (Human-in-the-loop).

    Only the UI calls this. It is never exposed to any agent, so the system itself
    can never approve or reject a claim.
    """
    decision = str(decision or "").strip().lower()
    if decision not in HUMAN_DECISIONS:
        return _result(ToolStatus.ERROR, message=f"decision must be one of {', '.join(HUMAN_DECISIONS)}")
    if not str(adjuster or "").strip():
        return _result(ToolStatus.ERROR, message="adjuster name is required")
    if decision != "approved" and not str(note or "").strip():
        return _result(ToolStatus.ERROR, message="a note is required for reject / request more info")
    path = _cases_path()
    try:
        with _CASES_LOCK:
            cases = _load(path) if path.exists() else []
            case = next((c for c in cases if c["case_id"] == _norm_id(case_id)), None)
            if case is None:
                return _result(ToolStatus.NOT_FOUND, message="no case with this id")
            if case.get("final_decision") in ("approved", "rejected"):
                return _result(ToolStatus.ERROR, message="this case already has a final human decision")
            now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
            case["final_decision"] = decision
            case["case_status"] = "needs_more_info" if decision == "needs_more_info" else "closed_by_adjuster"
            case["audit_trail"].append({"timestamp": now, "actor": f"human_adjuster:{adjuster.strip()}",
                                        "event": "human_decision", "detail": f"{decision}. {note.strip()}".strip()})
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
    except _DataUnavailable as exc:
        return _result(ToolStatus.UNAVAILABLE, message=f"could not verify: {exc}")
    return _result(ToolStatus.OK, case_id=case["case_id"], final_decision=decision,
                   message="decision recorded in the audit trail")


# ------------------------------------------------------------------ Agents SDK wiring
def build_agent_tools() -> dict[str, list]:
    """Wrap the five read-only tools as Agents SDK function tools, grouped per worker."""
    from agents import function_tool  # imported lazily so tests run without the SDK

    return {
        "policy": [function_tool(f) for f in (get_customer_policy, get_policy_coverage, calculate_deductible)],
        "history": [function_tool(check_claim_history)],
        "documents": [function_tool(check_required_documents)],
    }
