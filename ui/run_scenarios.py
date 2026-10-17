"""Run all 12 demo scenarios and print expected vs actual.

    python -m ui.run_scenarios            # MOCK LLM steps (no API key)
    python -m ui.run_scenarios --live     # needs OPENAI_API_KEY + src/live_steps.py

Cases are written to a temporary file, so data/cases.json is not touched.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

from src import tools
from src.workflow import run_claim


def main(live: bool = False) -> int:
    os.environ["CLAIMS_CASES_FILE"] = os.path.join(tempfile.mkdtemp(), "cases.json")
    if live:
        from src.live_steps import build_live_steps  # Youssef's module

        make_steps = build_live_steps
    else:
        from ui.mock_steps import build_mock_steps

        make_steps = build_mock_steps
    scenarios = json.loads((tools._data_dir() / "demo_scenarios.json").read_text(encoding="utf-8"))
    print(f"Mode: {'LIVE' if live else 'MOCK (LLM steps simulated; tools, rules and data are real)'}\n")
    failed = 0
    for s in scenarios:
        res = asyncio.run(run_claim(s["customer_id"], s["claim_text"], s["documents"], make_steps()))
        exp = s["expected"]
        case = tools.get_case(res.case["case_id"]) if res.case and res.case.get("status") == "ok" else None
        ok = (res.route == exp["route"] and set(exp["risk_flags"]) <= set(res.risk.risk_flags if res.risk else [])
              and case is not None and case["case_status"] == exp["case_status"])
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {s['id']}  {s['title']}")
        print(f"      expected route={exp['route']} flags={exp['risk_flags']} status={exp['case_status']}")
        print(f"      actual   route={res.route} flags={res.risk.risk_flags if res.risk else None} "
              f"status={case['case_status'] if case else None}")
    print(f"\n{len(scenarios) - failed}/{len(scenarios)} scenarios matched")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(live="--live" in sys.argv))
