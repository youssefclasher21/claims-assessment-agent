"""Workflow orchestration. Owner: Youssef.

Code-orchestrated steps (deterministic Python): input screening, risk routing,
parallel gathering, bounded revision loop, case-creation authorization,
validation of the LLM orchestrator's worker plan.
LLM-orchestrated steps (via the injected `steps`): intake, worker plan,
worker evidence, writing, evaluating.

`steps` is a WorkflowSteps object holding async callables. Real runs inject
Agents SDK calls; tests inject clearly-labelled MOCK steps. A mock run is NOT
proof of live SDK execution.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agents import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

from src.models import (
    CaseSummary,
    ClaimFacts,
    DocumentEvidence,
    EvaluatorVerdict,
    HistoryEvidence,
    PolicyEvidence,
    RiskDecision,
    RiskRoute,
    ToolStatus,
)
from src.orchestration import normalize_plan
from src.rules import (
    detect_manipulation,
    find_forbidden_language,
    get_max_revisions,
    pre_route_checks,
    decide_risk_route,
    should_continue_revising,
)


@dataclass
class WorkflowSteps:
    intake: Callable[[str], Awaitable[ClaimFacts]]
    policy: Callable[[str, ClaimFacts], Awaitable[PolicyEvidence]]
    history: Callable[[str, ClaimFacts], Awaitable[HistoryEvidence]]
    documents: Callable[[ClaimFacts, list[str]], Awaitable[DocumentEvidence]]
    write: Callable[[ClaimFacts, dict, str | None], Awaitable[CaseSummary]]
    evaluate: Callable[[CaseSummary, dict], Awaitable[EvaluatorVerdict]]
    create_case: Callable[[str, str, str, str], dict]  # tool: customer_id, risk, status, json
    plan: Callable[[ClaimFacts], Awaitable[Any]] | None = None  # LLM orchestrator (optional)


@dataclass
class WorkflowResult:
    route: str = "routine"
    stage_log: list[str] = field(default_factory=list)
    facts: ClaimFacts | None = None
    risk: RiskDecision | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    summary: CaseSummary | None = None
    revisions: int = 0
    evaluator_concerns: list[str] = field(default_factory=list)
    case: dict | None = None
    error: str | None = None
    blocked_reason: str | None = None


def _unavailable(model_cls, **extra):
    base = dict(status=ToolStatus.UNAVAILABLE, notes="could not verify: worker failed")
    base.update(extra)
    return model_cls(**base)


async def _safe(coro, fallback):
    """Run one worker; a failure becomes 'unavailable' evidence, never a crash."""
    try:
        return await coro
    except Exception:  # noqa: BLE001 - deliberate: recover gracefully
        return fallback


async def _const(value):
    return value


async def gather_evidence(customer_id, facts, documents, steps: WorkflowSteps,
                          selected: list[str] | None = None) -> dict:
    """Parallelization: the selected independent workers run via asyncio.gather.

    A worker the orchestrator did not select is marked 'not checked' (unavailable),
    never reported as a clean/negative finding.
    """
    selected = list(selected) if selected else ["policy", "history", "documentation"]
    skip_note = "not checked: the orchestrator did not select this worker"

    policy_fb = _unavailable(
        PolicyEvidence, policy_id=None, policy_status=None, is_covered=None,
        limit_amount=None, deductible=None, amount_after_deductible=None, exclusions=[])
    history_fb = _unavailable(HistoryEvidence, claim_count=None, claims_summary=[])
    docs_fb = _unavailable(DocumentEvidence, required=[], provided=[], missing=[])

    def pick(name, make_coro, fallback):
        if name in selected:
            return _safe(make_coro(), fallback)
        return _const(fallback.model_copy(update={"notes": skip_note}))

    policy, history, docs = await asyncio.gather(
        pick("policy", lambda: steps.policy(customer_id, facts), policy_fb),
        pick("history", lambda: steps.history(customer_id, facts), history_fb),
        pick("documentation", lambda: steps.documents(facts, documents), docs_fb),
    )
    return {"policy": policy, "history": history, "documents": docs}


async def run_claim(customer_id: str, claim_text: str, documents: list[str],
                    steps: WorkflowSteps) -> WorkflowResult:
    res = WorkflowResult()
    log = res.stage_log

    # 1. Input guardrail (code layer)
    hits = detect_manipulation(claim_text)
    log.append(f"input_guardrail: {'FLAGGED ' + str(hits) if hits else 'clean'}")

    # 2. Intake (LLM) -> structured ClaimFacts  [prompt chaining step 1]
    try:
        res.facts = await steps.intake(claim_text)
    except InputGuardrailTripwireTriggered:
        res.route = RiskRoute.HUMAN_REVIEW.value
        res.blocked_reason = "input guardrail blocked this claim text (manipulation or unauthorized request)"
        log.append("input_guardrail(SDK): TRIPPED -> claim blocked, escalate to human")
        try:
            res.case = steps.create_case(
                customer_id, "high", "pending_human_review",
                json.dumps({"blocked_by": "input_guardrail", "note": "claim text blocked before processing"}))
            if res.case.get("status") == "ok":
                log.append("blocked path: case created for human review")
            else:
                log.append(f"blocked path: case not created ({res.case.get('message')})")
        except Exception:  # noqa: BLE001
            log.append("blocked path: case creation failed")
        return res
    except Exception as exc:  # noqa: BLE001
        res.error = f"intake failed: {exc}"
        res.route = RiskRoute.HUMAN_REVIEW.value
        log.append("intake: FAILED -> escalate to human")
        return res
    log.append("intake: ok")

    # 3. Risk router (deterministic Python)
    res.risk = pre_route_checks(res.facts, claim_text)
    log.append(f"router: {res.risk.route.value} flags={res.risk.risk_flags}")
    if res.risk.route is RiskRoute.HUMAN_REVIEW:
        res.route = RiskRoute.HUMAN_REVIEW.value
        res.case = steps.create_case(customer_id, "high", "pending_human_review",
                                     res.risk.model_dump_json())
        log.append("high-risk path: case created for human review")
        return res

    # 4. LLM-orchestrated decision: which workers does this claim need?
    #    The plan is validated by Python (normalize_plan); the LLM never bypasses it.
    selected = None
    plan_step = getattr(steps, "plan", None)
    if plan_step is not None:
        try:
            plan = await plan_step(res.facts)
        except Exception:  # noqa: BLE001 - orchestrator failure falls back to all workers
            plan = None
        selected, notes = normalize_plan(plan)
        reason = plan.reasoning if plan is not None else "n/a"
        log.append(f"orchestrator(LLM): workers={selected} reason={reason}")
        for note in notes:
            log.append(f"orchestrator(Python validation): {note}")

    # 5. Parallel evidence workers  [parallelization]
    res.evidence = await gather_evidence(customer_id, res.facts, documents, steps, selected)
    log.append("workers: policy/history/documents gathered in parallel")

    # Re-run the router with real evidence (still Python)
    res.risk = decide_risk_route(res.facts, res.evidence["history"], res.evidence["policy"],
                                 detect_manipulation(claim_text))
    log.append(f"router(post-evidence): {res.risk.route.value} flags={res.risk.risk_flags}")
    res.route = res.risk.route.value

    # 6. Writer <-> Evaluator loop (bounded)  [evaluator-optimizer]
    feedback = None
    verdict = None
    revisions = 0
    while True:
        try:
            res.summary = await steps.write(res.facts, res.evidence, feedback)
            verdict = await steps.evaluate(res.summary, res.evidence)
        except OutputGuardrailTripwireTriggered as exc:
            res.summary = None  # the rejected draft is never shown
            _gr = getattr(getattr(exc, "guardrail_result", None), "output", None)
            log.append(f"output_guardrail hits: {getattr(_gr, 'output_info', None)}")
            log.append(f"output_guardrail(SDK): TRIPPED, draft rejected (revision {revisions})")
            if should_continue_revising(revisions, False):
                revisions += 1
                feedback = ("The output guardrail rejected the draft because it contained approval, "
                            "refund or payout wording. Rewrite it using neutral facts only. "
                            f"Rejected phrases: {getattr(_gr, 'output_info', None)}. "
                            "Do not state any amount remaining after the deductible.")
                continue
            res.error = "output guardrail rejected every draft"
            res.route = RiskRoute.HUMAN_REVIEW.value
            log.append("output_guardrail(SDK): max revisions reached -> escalate to human")
            try:
                res.case = steps.create_case(customer_id, "high", "pending_human_review",
                                             res.risk.model_dump_json())
                log.append("failure path: case created for human review")
            except Exception:  # noqa: BLE001
                log.append("failure path: case creation failed")
            return res
        except Exception as exc:  # noqa: BLE001 - model failure must not crash the workflow
            res.error = f"writer/evaluator failed: {exc}"
            res.route = RiskRoute.HUMAN_REVIEW.value
            res.summary = None  # never show an unvalidated draft
            log.append("writer/evaluator: FAILED -> escalate to human")
            try:
                res.case = steps.create_case(customer_id, "high", "pending_human_review",
                                             res.risk.model_dump_json())
                log.append("failure path: case created for human review")
            except Exception:  # noqa: BLE001
                log.append("failure path: case creation failed")
            return res
        code_issues = find_forbidden_language(res.summary.summary + " " + res.summary.recommended_next_step)
        is_valid = verdict.is_valid and not code_issues
        log.append(f"evaluator: valid={is_valid} (revision {revisions})")
        if not should_continue_revising(revisions, is_valid):
            break
        revisions += 1
        feedback = "; ".join(verdict.issues + code_issues) + " | " + verdict.revision_instructions
    res.revisions = revisions
    if res.summary is not None and res.risk.risk_flags:
        res.summary.open_questions.append(
            "Escalated for human review by the risk rules: " + ", ".join(res.risk.risk_flags))
    if not is_valid:
        res.evaluator_concerns = verdict.issues + code_issues
        res.route = RiskRoute.HUMAN_REVIEW.value
        log.append(f"max revisions ({get_max_revisions()}) reached -> escalate with concerns")

    # 7. Protected action (authorization enforced inside the tool via rules.py)
    status = ("pending_human_review" if res.route == RiskRoute.HUMAN_REVIEW.value else "needs_more_info" if (res.facts.missing_fields or res.evidence["documents"].missing) else "pending_human_review")
    res.case = steps.create_case(customer_id, res.route, status, res.summary.model_dump_json())
    if res.case.get("status") != "ok":
        res.blocked_reason = res.case.get("message")
        log.append(f"create_claim_case: BLOCKED/failed ({res.blocked_reason})")
    else:
        log.append("create_claim_case: saved")
    return res