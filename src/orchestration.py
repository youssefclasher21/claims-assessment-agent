"""LLM-orchestrated planning step. Owner: Youssef.

The Claim Orchestrator (LLM) proposes which evidence workers a claim needs.
Python then validates the plan: unknown names are dropped, mandatory workers
are always added, and a failed/missing plan falls back to running every worker.
The LLM never bypasses these rules.
"""

from __future__ import annotations

from agents import Agent, ModelSettings
from pydantic import BaseModel, Field

from src.claim_agents import _DATA_RULE, get_model, out_type

ALL_WORKERS = ("policy", "history", "documentation")
MANDATORY_WORKERS = ("policy", "documentation")


class WorkerPlan(BaseModel):
    """Structured plan returned by the Claim Orchestrator."""

    workers: list[str] = Field(
        default_factory=list,
        description="Worker names to run. Allowed: policy, history, documentation.",
    )
    reasoning: str = Field(default="", description="One short sentence explaining the choice.")


def normalize_plan(plan: WorkerPlan | None) -> tuple[list[str], list[str]]:
    """Deterministic Python validation of the LLM plan.

    Returns (selected_workers_in_canonical_order, correction_notes).
    plan=None means the orchestrator failed: run every worker (safest).
    """
    if plan is None:
        return list(ALL_WORKERS), ["orchestrator unavailable: running all workers"]

    notes: list[str] = []
    requested = {str(w).strip().lower() for w in plan.workers}

    unknown = sorted(requested - set(ALL_WORKERS))
    for name in unknown:
        notes.append(f"dropped unknown worker '{name}'")
    requested &= set(ALL_WORKERS)

    for name in MANDATORY_WORKERS:
        if name not in requested:
            notes.append(f"added mandatory worker '{name}'")
            requested.add(name)

    return [w for w in ALL_WORKERS if w in requested], notes


def build_plan_agent() -> Agent:
    """Claim Orchestrator: decides which workers this claim needs."""
    return Agent(
        name="Claim Orchestrator",
        instructions=(
            "You coordinate evidence gathering for ONE insurance claim. Given the extracted "
            "claim facts, choose which workers to run from: policy (policy, coverage, "
            "deductible), history (prior claims), documentation (required documents). "
            "Return the worker names in `workers` and one short sentence in `reasoning`. "
            "Do not answer coverage questions yourself. " + _DATA_RULE
        ),
        model=get_model(),
        model_settings=ModelSettings(temperature=0),
        output_type=out_type(WorkerPlan),
    )