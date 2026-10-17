"""Runs every demo scenario through the workflow with MOCK LLM steps.

Tools, data, rules and guardrails are real; only intake/writer/evaluator are simulated.
"""

import asyncio
import json

import pytest

from src import tools
from src.workflow import run_claim
from ui.mock_steps import build_mock_steps

SCENARIOS = json.loads((tools._data_dir() / "demo_scenarios.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def tmp_cases(tmp_path, monkeypatch):
    f = tmp_path / "cases.json"
    f.write_text("[]")
    monkeypatch.setenv("CLAIMS_CASES_FILE", str(f))


def test_there_are_12_scenarios_with_unique_ids():
    assert len(SCENARIOS) == 12 and len({s["id"] for s in SCENARIOS}) == 12


@pytest.mark.parametrize("s", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario(s):
    res = asyncio.run(run_claim(s["customer_id"], s["claim_text"], s["documents"], build_mock_steps()))
    exp = s["expected"]
    assert res.route == exp["route"], res.stage_log
    assert set(exp["risk_flags"]) <= set(res.risk.risk_flags)
    if not exp["risk_flags"]:
        assert res.risk.risk_flags == []
    assert res.case["status"] == "ok", res.blocked_reason
    case = tools.get_case(res.case["case_id"])
    assert case["case_status"] == exp["case_status"]
    assert case["risk_level"] == exp["risk_level"]
    assert case["final_decision"] is None  # only a human can decide


def test_gap_scenario_flags_uncovered_claim_type():
    s = next(x for x in SCENARIOS if x["id"] == "S10")
    res = asyncio.run(run_claim(s["customer_id"], s["claim_text"], s["documents"], build_mock_steps()))
    assert any("NOT listed" in q for q in res.summary.open_questions)
