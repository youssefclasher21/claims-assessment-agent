"""LLM provider setup. Owner: Youssef.

Default is OpenAI. For a free provider set LLM_PROVIDER=gemini or groq in .env.
Non-OpenAI providers use the chat-completions API and cannot export traces to
OpenAI, so SDK trace export is disabled for them (never silently claimed).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

PROVIDERS = {
    "openai": (None, "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}


def configure_llm() -> str:
    """Load .env and configure the Agents SDK client. Returns provider name."""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in PROVIDERS:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'. Use: {list(PROVIDERS)}")
    base_url, key_env = PROVIDERS[provider]
    key = os.getenv(key_env)
    if not key:
        raise RuntimeError(f"{key_env} is not set in .env")

    if provider != "openai":
        from agents import (set_default_openai_api, set_default_openai_client,
                            set_tracing_disabled)
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base_url, api_key=key)
        set_default_openai_client(client, use_for_tracing=False)
        set_default_openai_api("chat_completions")
        set_tracing_disabled(True)
    return provider