# Orchestration (Dify / n8n) — visual pipeline + human-abstention approval

The grader's stages are already separate scripts; `scripts/pipeline.py` chains
them into one run that emits a **machine-readable manifest** and an **exit code a
workflow can branch on**. An orchestrator (n8n or Dify) only needs to shell out
to it and route on the result — the **abstain queue becomes a human-approval
node**, closing the selective-grading loop (grade → auto-publish the confident
ones → send the rest to a human).

## The contract `pipeline.py` exposes

```
python scripts/pipeline.py --assign <ASSIGN> --engine numeric|general \
    [--from <stage> --to <stage>] [--ingest] [--memory <store>] \
    [--manifest <path>] <LLM flags>
```

- **Stages** — numeric: `preprocess → score → report`; general: `score → report`
  (`--ingest` appends a task-bank archival stage). `--from/--to` run one stage,
  so each workflow node can own a single step.
- **Manifest** (`workspace/<ASSIGN>/run_manifest.json`) — per-stage
  `status/returncode/duration_s` plus a `review` summary:
  ```json
  {
    "assignment": "HW2", "engine": "numeric", "status": "needs_review",
    "stages": [{"stage": "preprocess", "status": "ok", "duration_s": 12.3}, ...],
    "review": {"total": 38, "auto": 33, "abstain": 5,
               "review_queue": [{"student_id": "anon-007",
                                 "reasons": ["low_confidence:Q3"], "confidence": 0.41}]}
  }
  ```
- **Exit code** — the routing signal:
  | code | meaning | workflow action |
  |---|---|---|
  | `0`  | done, all AUTO            | auto-publish / auto-ingest |
  | `10` | done, some **ABSTAIN**    | route to the **human-approval** node |
  | `1`  | a stage failed            | stop / alert |

## n8n

Import `n8n_grading_workflow.json` (Workflows → Import from File). Flow:

```
Start → Config(assign,engine) → Run pipeline → Read manifest → Parse manifest
      → IF abstain>0 ─true→  HUMAN APPROVAL (Wait: resume-webhook) → Archive after approval
                    └false→  Auto-archive (all AUTO)
```

- **Run pipeline** appends `|| true` so the node never hard-fails on exit `10`;
  the parsed manifest (`review.abstain`) is the branch condition — robust and
  explicit.
- **HUMAN APPROVAL** is an n8n `Wait` node (`resume: webhook`). The incoming item
  carries `review_queue` (ids + reasons). A grader inspects
  `workspace/<ASSIGN>/review_queue/*.json`, makes corrections, then calls the
  node's resume URL to continue into the archival stage.
- Set `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` as workflow/container env vars.
  Mount the repo at `/app` (the commands `cd /app`).

## Dify

Dify runs hosted, so it orchestrates over **HTTP** rather than shelling out: wrap
`pipeline.py` behind a tiny HTTP service (e.g. FastAPI `POST /grade` →
`run_pipeline(...)` returning the manifest JSON), then in a Dify **Workflow**:

```
Start → HTTP Request (POST /grade {assign, engine})
      → Code/IF on response.review.abstain
          >0 → Human-in-the-loop (Dify approval / external review link) → HTTP POST /ingest
          =0 → HTTP POST /ingest
      → End
```

The mapping is identical: the manifest's `review.abstain` is the branch, the
abstain queue is the approval payload. Only the transport differs (HTTP vs.
Execute Command). The grader stays the single source of truth; the orchestrator
only routes.

## Why the orchestrator stays thin

All grading logic — determinism-first execution, localization, physics/field
invariants, the confidence gate — lives in the scripts and is unit-tested. The
workflow contributes only *routing and human approval*. That keeps the pipeline
runnable head-less (CI, cron) and the visual workflow a thin, swappable shell.
