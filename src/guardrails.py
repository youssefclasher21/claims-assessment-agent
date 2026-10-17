"""SDK guardrails that wrap our deterministic Python rules. Owner: Youssef.

Input guardrail  -> attached to the Intake Agent (first agent of a run).
Output guardrail -> attached to the Claim Writer.
Neither calls an LLM: src/rules.py decides, the SDK runs the check, records it
in the trace and raises a tripwire exception when it fires.
"""

from __future__ import annotations

import re

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
)
from pydantic import BaseModel

from src.rules import detect_manipulation, find_forbidden_language

_TAG_RE = re.compile(r"<claim_text>(.*?)</claim_text>", re.DOTALL | re.IGNORECASE)
_PROMPT_TAIL = "\n\nReply with ONLY"


def _as_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, list):
        return " ".join(_as_text(v.get("content") if isinstance(v, dict) else v) for v in value)
    return str(value)


def extract_claim_text(input_text: str) -> str:
    """Screen only the customer's claim, never our own instructions."""
    m = _TAG_RE.search(input_text)
    if m:
        return m.group(1)
    return input_text.split(_PROMPT_TAIL)[0]


def check_claim_input(input_text: str) -> list[str]:
    return detect_manipulation(extract_claim_text(input_text))


def check_summary_output(output_text: str) -> list[str]:
    return find_forbidden_language(output_text)


@input_guardrail
async def claim_input_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    hits = check_claim_input(_as_text(input))
    return GuardrailFunctionOutput(
        output_info={"manipulation_hits": hits}, tripwire_triggered=bool(hits)
    )


@output_guardrail
async def claim_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output
) -> GuardrailFunctionOutput:
    hits = check_summary_output(_as_text(output))
    return GuardrailFunctionOutput(
        output_info={"forbidden_hits": hits}, tripwire_triggered=bool(hits)
    )