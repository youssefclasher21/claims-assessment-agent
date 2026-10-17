"""LIVE smoke test: one real Intake Agent call. Needs a key in .env."""

import asyncio

from src.claim_agents import build_intake_agent
from src.llm_config import configure_llm
from src.llm_json import run_structured
from src.models import ClaimFacts


async def main():
    print("provider:", configure_llm())
    facts = await run_structured(
        build_intake_agent(),
        "My car was rear-ended at a traffic light on 2026-09-01. The repair estimate is 1500 USD.",
        ClaimFacts,
    )
    print(type(facts).__name__)
    print(facts)


asyncio.run(main())