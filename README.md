# AI Insurance Claims Assessment Agent

DEPI Agentic AI final project (Project 2). All data is **synthetic**.
The system prepares a structured assessment for a **human adjuster**. It never approves or rejects a claim.

> Principle: the LLM decides what to do; Python decides what is allowed.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then fill OPENAI_API_KEY (never commit .env)
python -m pytest -v
 python -m ui.app
```

## Code-orchestrated vs LLM-orchestrated

| Decision | Style | Where |
|---|---|---|
| Input manipulation screening | Code | `src/rules.py` `detect_manipulation` |
| Routine vs high-risk routing | Code | `src/rules.py` `decide_risk_route`, `src/workflow.py` |
| Parallel evidence gathering | Code | `src/workflow.py` `gather_evidence` (`asyncio.gather`) |
| Max revision count | Code | `src/rules.py` `should_continue_revising` (`MAX_REVISIONS`, 1 to 5) |
| `create_claim_case` authorization | Code | `src/rules.py` `validate_case_payload` |
| Forbidden approval/refund language | Code | `src/rules.py` `find_forbidden_language` |
| Claim facts extraction and type | LLM | Intake Agent |
| Which workers to call | LLM | Claim Orchestrator |
| Case summary writing | LLM | Claim Writer |
| Is the draft supported by evidence | LLM + code check | Evaluator + `find_forbidden_language` |

## Workflow patterns

- **Prompt chaining:** Intake (`ClaimFacts`) -> evidence -> Claim Writer (`CaseSummary`).
- **Routing:** deterministic risk router sends suspicious claims to human review.
- **Parallelization:** policy, history, document workers via `asyncio.gather`.
- **Orchestrator-workers:** Claim Orchestrator exposes three workers as tools.
- **Evaluator-optimizer:** Claim Writer <-> Evaluator, bounded by `MAX_REVISIONS`; if still invalid, escalate to a human with the concerns attached.

Evidence statuses are `ok | not_found | unavailable | error`. `unavailable` means "could not verify" and is never a negative finding.

## Verification status (honest)

| Item | Status |
|---|---|
| Pydantic models | PASS (offline tests) |
| Risk router, language guards, bounded loop | PASS (offline tests) |
| Agents build with correct output types | PASS (construction only, no model call) |
| Workflow logic | PASS with **MOCK** steps. Not proof of live SDK execution |
| Live Agents SDK runs, LLM guardrails, `trace()`, `SQLiteSession` | **BLOCKED**: no OpenAI API key available |
| Domain tools, data, Gradio, 12 demo scenarios | NOT VERIFIED yet |
| OCR bonus | Deferred |
