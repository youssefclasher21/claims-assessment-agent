"""LIVE smoke test: one real Intake Agent call. Needs a key in .env."""

import asyncio

from agents import Runner

from src.claim_agents import build_intake_agent
from src.llm_config import configure_llm


async def main():
    print("provider:", configure_llm())
    result = await Runner.run(
        build_intake_agent(),
        "My car was rear-ended at a traffic light on 2026-09-01. The repair estimate is 1500 USD.",
    )
    print(type(result.final_output).__name__)
    print(result.final_output)


asyncio.run(main())