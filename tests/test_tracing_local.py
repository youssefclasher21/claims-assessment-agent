from types import SimpleNamespace

from src.tracing_local import LocalTraceProcessor


def _span(data, started="2026-10-17T10:00:00"):
    return SimpleNamespace(
        span_data=SimpleNamespace(export=lambda: data), started_at=started, error=None
    )


def test_records_agent_tool_and_model_spans_in_order():
    t = LocalTraceProcessor()
    t.on_trace_start(SimpleNamespace(name="Claim assessment workflow"))
    t.on_span_end(_span({"type": "function", "name": "get_customer_policy", "output": "{'status': 'ok'}"}, "2026-10-17T10:00:02"))
    t.on_span_end(_span({"type": "agent", "name": "Policy Worker"}, "2026-10-17T10:00:01"))
    t.on_span_end(_span({"type": "generation", "model": "some-model"}, "2026-10-17T10:00:03"))
    out = t.format()
    assert "Claim assessment workflow" in out
    assert out.index("Policy Worker") < out.index("get_customer_policy")
    assert "1 agent spans, 1 tool calls, 1 model calls" in out


def test_ignores_unknown_span_types_and_bad_data():
    t = LocalTraceProcessor()
    t.on_span_end(_span({"type": "custom", "name": "x"}))
    t.on_span_end(SimpleNamespace(span_data=SimpleNamespace(export=lambda: 1 / 0)))
    assert t.spans == []


def test_reset_clears_everything():
    t = LocalTraceProcessor()
    t.on_trace_start(SimpleNamespace(name="w"))
    t.on_span_end(_span({"type": "agent", "name": "A"}))
    t.reset()
    assert t.spans == [] and t.workflows == []