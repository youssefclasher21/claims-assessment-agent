"""Local trace collector for the Agents SDK. Owner: Youssef.

Providers other than OpenAI cannot export traces to the OpenAI dashboard, so we
register our own TracingProcessor. It records every agent, tool (function) and
model-call span of a `trace()` and can print them in order for the demo.
"""

from __future__ import annotations

import threading

from agents.tracing import TracingProcessor


def _short(value, limit: int = 120) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


class LocalTraceProcessor(TracingProcessor):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.workflows: list[str] = []
        self.spans: list[dict] = []

    # --- TracingProcessor interface -------------------------------------
    def on_trace_start(self, trace) -> None:
        with self._lock:
            self.workflows.append(getattr(trace, "name", "workflow"))

    def on_trace_end(self, trace) -> None:
        pass

    def on_span_start(self, span) -> None:
        pass

    def on_span_end(self, span) -> None:
        try:
            data = span.span_data.export()
        except Exception:  # noqa: BLE001 - tracing must never break the workflow
            return
        kind = data.get("type", "other")
        if kind not in ("agent", "function", "generation"):
            return
        record = {
            "kind": kind,
            "name": data.get("name") or data.get("model") or kind,
            "started_at": str(getattr(span, "started_at", "") or ""),
            "error": bool(getattr(span, "error", None)),
        }
        if kind == "function":
            record["detail"] = _short(data.get("output"))
        with self._lock:
            self.spans.append(record)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    # --- helpers for the demo / UI ---------------------------------------
    def reset(self) -> None:
        with self._lock:
            self.workflows.clear()
            self.spans.clear()

    def format(self) -> str:
        """Human-readable trace: workflow name, then spans in start order."""
        with self._lock:
            spans = sorted(self.spans, key=lambda s: s["started_at"])
            workflows = list(self.workflows)
        label = {"agent": "AGENT", "function": "TOOL ", "generation": "LLM  "}
        lines = [f"workflow: {', '.join(workflows) or '(none)'}"]
        for s in spans:
            err = "  [ERROR]" if s["error"] else ""
            detail = f"  -> {s['detail']}" if s.get("detail") else ""
            lines.append(f"  {label[s['kind']]} {s['name']}{detail}{err}")
        lines.append(
            f"summary: {sum(s['kind'] == 'agent' for s in spans)} agent spans, "
            f"{sum(s['kind'] == 'function' for s in spans)} tool calls, "
            f"{sum(s['kind'] == 'generation' for s in spans)} model calls"
        )
        return "\n".join(lines)


LOCAL_TRACER = LocalTraceProcessor()