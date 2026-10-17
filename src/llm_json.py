"""Structured output for providers whose strict JSON mode is unreliable.

The agent runs WITHOUT output_type; the prompt asks for JSON and Pydantic
validates it. If the model replies with the JSON schema itself (or invalid
JSON), the validation error is fed back and the call is retried (max 3 tries)."""

from __future__ import annotations

import json
import re

from agents import Agent, Runner
from pydantic import BaseModel, ValidationError

MAX_ATTEMPTS = 3


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found in model output")
    return m.group(0)


def _describe_fields(model_cls: type[BaseModel]) -> str:
    lines = []
    for name, info in model_cls.model_fields.items():
        required = "required" if info.is_required() else "optional"
        desc = f" - {info.description}" if info.description else ""
        lines.append(f'- "{name}" ({required}){desc}')
    return "\n".join(lines)


def _parse(text: str, model_cls: type[BaseModel]):
    data = json.loads(_extract_json(text))
    if isinstance(data, dict) and ("properties" in data or "$defs" in data):
        raise ValueError(
            "you returned the JSON schema itself; return one JSON object with real values instead"
        )
    return model_cls.model_validate(data)


async def run_structured(agent: Agent, user_input: str, model_cls: type[BaseModel]):
    """Run `agent` (built without output_type) and return a validated model_cls."""
    schema = json.dumps(model_cls.model_json_schema())
    prompt = (
        f"{user_input}\n\n"
        "Reply with ONLY one JSON object containing real VALUES for these fields "
        "(no markdown, no extra text). Do NOT copy or echo the schema below; "
        "it only describes the allowed format.\n"
        f"Fields:\n{_describe_fields(model_cls)}\n\n"
        f"Format reference (do not return this): {schema}"
    )
    last_err = None
    for _ in range(MAX_ATTEMPTS):
        result = await Runner.run(agent, prompt)
        try:
            return _parse(str(result.final_output), model_cls)
        except (ValueError, ValidationError) as exc:
            last_err = exc
            prompt += (
                f"\n\nYour previous reply was invalid: {str(exc)[:500]}. "
                "Return ONE corrected JSON object with real values only."
            )
    raise ValueError(f"structured output failed validation: {last_err}")