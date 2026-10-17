# Demo guide (Samir: tools, data and interface)

All data is synthetic. The assistant prepares an assessment for a human adjuster; it never approves or rejects a claim.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install gradio
python -m pytest -v
```

## Run the interface

```powershell
python -m ui.app
```

Open the local link (usually http://127.0.0.1:7860).

| Mode | What is real | What you need |
|---|---|---|
| **Mock** (default) | Tools, data, risk rules, guardrails, case authorization, human decision | Nothing |
| **Live** | Everything, including the LLM agents | `OPENAI_API_KEY` in `.env` and `src/live_steps.py` exposing `build_live_steps()` |

In Mock mode the intake, writer and evaluator steps are simulated, and the UI shows a MOCK banner.

## How to use the interface

1. **Submit claim** tab: pick a demo scenario (or write your own claim), press **Analyze claim**.
2. Read the route, risk flags, evidence and case summary. The **Workflow trace** shows each stage.
3. **Adjuster review** tab: press **Refresh cases**, select the case, enter your name and press
   **Approve**, **Reject** or **Request more information**. Reject and Request more information need a note.
   The decision is added to the case audit trail.

## Run all scenarios from the terminal

```powershell
python -m ui.run_scenarios
```

It prints expected vs actual for every scenario and does not touch `data/cases.json`.

## The 12 demo scenarios

| ID | Customer | Scenario | Route | Risk flags | Case status |
|---|---|---|---|---|---|
| S01 | CUST-001 | Routine auto claim, all documents | `routine` | - | `pending_human_review` |
| S02 | CUST-001 | Routine claim with missing documents | `routine` | - | `needs_more_info` |
| S03 | CUST-002 | Routine health claim | `routine` | - | `pending_human_review` |
| S04 | CUST-002 | Vague claim: amount and date missing | `routine` | - | `needs_more_info` |
| S05 | CUST-001 | Manipulation: 'approve my claim' | `human_review` | `manipulation_attempt` | `pending_human_review` |
| S06 | CUST-009 | Unauthorized policy change request | `human_review` | `manipulation_attempt` | `pending_human_review` |
| S07 | CUST-006 | High-amount claim | `human_review` | `high_amount` | `pending_human_review` |
| S08 | CUST-003 | Repeated claims (4 prior) | `human_review` | `repeated_claims` | `pending_human_review` |
| S09 | CUST-004 | Expired policy | `human_review` | `policy_not_active` | `pending_human_review` |
| S10 | CUST-005 | Claim type not on the policy | `routine` | - | `pending_human_review` |
| S11 | CUST-007 | Claims-history system unavailable | `human_review` | `history_unverified` | `pending_human_review` |
| S12 | CUST-008 | Policy system unavailable | `human_review` | `policy_unverified` | `pending_human_review` |

## Suggested recording order

1. **S01 - Routine auto claim, all documents**: Normal path: three workers run in parallel, the summary is checked, and the case waits for the adjuster.
2. **S02 - Routine claim with missing documents**: The agent lists exactly which documents are missing and marks the case needs_more_info.
3. **S03 - Routine health claim**: Different claim type, different required documents and a different policy.
4. **S04 - Vague claim: amount and date missing**: The intake step uses null for unknown amount and date instead of inventing them.
5. **S05 - Manipulation: 'approve my claim'**: Python catches the manipulation before any worker runs and sends the claim straight to a human.
6. **S06 - Unauthorized policy change request**: The system has no tool that edits policies, and requests to change one are flagged.
7. **S07 - High-amount claim**: Claims above 50,000 EGP skip the workers and go to a human with the risk flag recorded.
8. **S08 - Repeated claims (4 prior)**: The claim looks routine at first; the history worker finds 4 prior claims and the router escalates it.
9. **S09 - Expired policy**: The policy is expired, so a human must look at the claim.
10. **S10 - Claim type not on the policy**: This customer only has auto cover. The summary flags the gap as an open question; the human decides.
11. **S11 - Claims-history system unavailable**: When a system is down we say 'could not verify' and never treat it as a clean record.
12. **S12 - Policy system unavailable**: The workflow does not crash: the missing evidence is recorded and the case goes to a human.

Finish the recording in the **Adjuster review** tab: open a saved case, show the audit trail, and record a human decision.
This is the human-in-the-loop proof: no agent or tool can set an approval or rejection.

## Test data map

| Customer | Policy | Used for |
|---|---|---|
| CUST-001 | auto, active, 0 prior claims | routine path, missing documents, manipulation |
| CUST-002 | health, active, 1 prior claim | health claims, vague claim |
| CUST-003 | property, active, 4 prior claims | repeated claims |
| CUST-004 | travel, **expired** | expired policy |
| CUST-005 | auto only | claim type not on the policy |
| CUST-006 | property, flood excluded | high amount |
| CUST-007 | travel, history system **unavailable** | tool failure |
| CUST-008 | health, policy system **unavailable** | tool failure |
| CUST-009 | auto, 2 prior claims | policy-change request |

## Known limits

- Uploaded files are only checked by **file name**. Reading their content (OCR / multimodal) is the optional bonus.
- Mock mode does not prove the live agents work. Live runs need an API key and Youssef's `src/live_steps.py`.
