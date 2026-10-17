"""Structured output for providers whose strict JSON mode is unreliable.

The agent runs WITHOUT output_type; the prompt asks for JSON and Pydantic
validates it (one retry with the validation error as feedback)."""

from __future__ import annotations

import json
import re

from agents import Agent, Runner
from pydantic import BaseModel, ValidationError


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found in model output")
    return m.group(0)


async def run_structured(agent: Agent, user_input: str, model_cls: type[BaseModel]):
    """Run `agent` (built without output_type) and return a validated model_cls."""
    schema = json.dumps(model_cls.model_json_schema())
    prompt = (
        f"{user_input}\n\nReply with ONLY one JSON object matching this JSON schema, "
        f"no markdown, no extra text:\n{schema}"
    )
    last_err = None
    for _ in range(2):
        result = await Runner.run(agent, prompt)
        try:
            return model_cls.model_validate_json(_extract_json(str(result.final_output)))
        except (ValueError, ValidationError) as exc:
            last_err = exc
            prompt += f"\n\nYour previous reply was invalid: {exc}. Return corrected JSON only."
    raise ValueError(f"structured output failed validation: {last_err}")