from src.claim_agents import (
    build_claim_writer,
    build_documentation_worker,
    build_evaluator,
    build_history_worker,
    build_intake_agent,
    build_orchestrator,
    build_policy_worker,
)
from src.models import CaseSummary, ClaimFacts, EvaluatorVerdict


def test_seven_agents_build_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    policy = build_policy_worker()
    history = build_history_worker()
    docs = build_documentation_worker()
    build_intake_agent()
    build_claim_writer()
    build_evaluator()
    orch = build_orchestrator({"policy": policy, "history": history, "documentation": docs})
    assert {t.name for t in orch.tools} == {
        "run_policy", "run_history", "run_documentation"
    }


def test_structured_output_types(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert build_intake_agent().output_type is ClaimFacts
    assert build_claim_writer().output_type is CaseSummary
    assert build_evaluator().output_type is EvaluatorVerdict


def test_agents_forbid_approval_in_instructions(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    for agent in (build_claim_writer(), build_evaluator(), build_intake_agent()):
        assert "never approve or reject" in agent.instructions