# JN Grader — Grading Workbench (web UI)

A small Flask web app that drives the whole grading loop from the browser:

1. **Upload** an assignment's files — student submissions (multiple `.ipynb` or a
   `.zip`), plus optional `reference.ipynb` / `description.txt` / `config.yaml` —
   laid out under `datasets/<ASSIGN>/`.
2. **Start grading** — kicks off `scripts/pipeline.py` as a background run; the
   panel polls live status (per-stage + log tail) until it finishes.
3. **Review** — browse scores + feedback + deterministic diagnostics (error
   class, first divergence, failed physics/plot invariants) and **work the
   abstention queue**: approve or override flagged submissions. Each decision is
   persisted to `workspace/<ASSIGN>/decisions/`, the visual form of the
   pipeline's human-approval node.

It reads/writes live under `datasets/` and `workspace/` — **no database required**
(the Postgres task bank stays optional). API keys are taken from the **server's**
environment (`LLM_API_KEY`), never sent from the browser.

## Run

```bash
pip install -r webapp/requirements.txt          # just Flask
set -a; source .env; set +a                      # so the server has LLM_API_KEY for grading
python webapp/app.py                             # http://127.0.0.1:5000
# custom dirs / host / port:
JN_WORKSPACE=workspace JN_DATASETS=datasets JN_HOST=0.0.0.0 JN_PORT=8080 python webapp/app.py
```

In the **New / Upload** panel: enter an assignment key, choose files, pick the
engine + provider/base-url/model, then **Upload** and **▶ Start grading**. Watch
the run status, then click the assignment to review. Use **needs review only** to
jump straight to the abstention queue; **Approve** accepts the AI score or
**Override** records an adjusted score + note.

## Layout

```
webapp/
├── app.py              # Flask routes (thin) over the layers below
├── data.py             # reads scored JSON, normalizes both engines, persists decisions
├── files.py            # upload ingest: lay out submissions/reference/config (path-safe)
├── runner.py           # background pipeline runs + status (process factory injectable)
├── requirements.txt    # Flask
└── static/index.html   # single-page workbench (vanilla JS + CSS, no build step)
```
All four backend modules are Flask-free and unit-tested (`tests/test_webapp*.py`).

## API

| Method + path | Purpose |
|---|---|
| `GET /api/datasets` | uploaded assignments + input status + how many are graded |
| `POST /api/assignments/<a>/upload` | multipart: `submissions` (×N or .zip) + optional `reference`/`description`/`config` |
| `POST /api/assignments/<a>/grade` | start a background pipeline run (`{engine, llm}`) |
| `GET /api/assignments/<a>/run` | live run status (stages + log tail + review summary) |
| `GET /api/assignments` | graded assignments + summary (n, avg %, #abstain) |
| `GET /api/assignments/<a>` | submission rows (sorted) + run manifest + decisions |
| `GET /api/assignments/<a>/submissions/<sid>` | full detail: per-item scores, feedback, diagnostics, failed invariants |
| `POST /api/assignments/<a>/submissions/<sid>/decision` | record a human decision (`approve` / `override` + `final_score`/`note`) |

Decisions are written to `workspace/<ASSIGN>/decisions/<sid>.json` — read them
back from any export/report step to apply the human-reviewed grades.

## Notes on the grading run

- `grade` shells out to `scripts/pipeline.py` (one run per assignment; a second
  request while one is in flight returns `409`). Exit code → status: `0` done,
  `10` done-with-review-queue, else failed.
- The browser never sees API keys — the pipeline subprocess inherits the
  server's `LLM_API_KEY`. Provider / base-url / model are passed per run.

## Notes

- Both engines are normalized: the general grader's `problems[]` and the numeric
  grader's `Q*` + `diagnostics` (+ failed physics/field invariants) render in the
  same submission detail view.
- The data layer (`data.py`) has no Flask dependency and is covered by
  `tests/test_webapp.py` (the Flask routes are smoke-tested with `app.test_client`).
