# JN Grader

**Grade Jupyter Notebook homework for any course — with reproducible execution,
reference-grounded LLM judgment, and a queryable results database.**

You provide, per assignment: a **problem statement**, a **correct reference
solution** (a notebook), and a folder of **student submissions**. JN Grader
executes the reference to obtain ground-truth answers, grades every submission
against it, and produces per-student Markdown reports plus rows in a Postgres
task bank you can query across assignments.

It is course-agnostic: nothing about a specific assignment is hard-coded — all
assignment knowledge lives in per-assignment files you drop in.

See [`architecture.md`](architecture.md) for the design and component status.

---

## Two grading engines

| When the answer is… | Engine | Entry script |
|---|---|---|
| **one number per question**, checkable with a tolerance | **Numeric autograder** | `score_notebooks.py` (+ `preprocess.py`, `run_tests.py`) |
| **matrices / derivations / multi-part**, or "submit a link" | **General grader** | `score_general.py` |

Both execute the reference solution for ground truth, share the same inputs
layout, and write the same reports and database records.

- **Numeric autograder** runs the notebook, compares the computed value to the
  reference within tolerance (hard pass/fail), then an LLM scores the *method*
  for partial credit.
- **General grader** runs the reference, captures its computed answers + written
  explanation, and an LLM scores each problem against that ground truth — judging
  both correctness and explanation quality, with per-problem points.

---

## Directory layout (one folder per assignment)

```
datasets/<ASSIGN>/
├── description.txt     # problem statement   → gen_config.py
├── reference.ipynb     # correct solution    → executed for ground-truth answers
├── config.yaml         # grading config      → auto-generated, then reviewed
└── submissions/        # student *.ipynb     → the grader's input

workspace/<ASSIGN>/     # outputs (git-ignored, regenerable)
└── scored/   reports/   processed/
```

> Keep `reference.ipynb` / `description.txt` **outside** `submissions/` — every
> `*.ipynb` in the input folder is treated as a student submission.

---

## Prerequisites

- **Docker** (recommended): `Dockerfile` + `docker-compose.yml` bundle Python, a
  Jupyter kernel, and Postgres + pgvector, and run student code as a non-root
  container user. (Or run the scripts directly with Python ≥ 3.10 and
  `pip install -r scripts/requirements.txt`.)
- **An LLM API key** in a `.env` file at the repo root:
  - OpenAI-compatible provider (default — DeepSeek, Qwen, SiliconFlow, …):
    `LLM_API_KEY=...`
  - and/or Anthropic (Claude): `ANTHROPIC_API_KEY=...` (used with `--provider anthropic`).

```bash
printf 'LLM_API_KEY=%s\n' '<your-key>' > .env     # .env is git-ignored
```

---

## Grade an assignment (general grader)

Drop `description.txt` + `reference.ipynb` into `datasets/<ASSIGN>/`, then:

```bash
cd 471Grader
ASSIGN=my-assignment        # the folder name and database key
LLM="--base-url https://api.deepseek.com --model deepseek-chat"   # your provider

# 1) LLM turns the description into a grading config (REVIEW the result)
docker compose run --rm grader python scripts/gen_config.py \
  datasets/$ASSIGN/description.txt --key $ASSIGN \
  --output datasets/$ASSIGN/config.yaml $LLM

# 2) Grade every submission against the reference
docker compose run --rm grader python scripts/score_general.py \
  datasets/$ASSIGN/submissions \
  --reference datasets/$ASSIGN/reference.ipynb \
  --config datasets/$ASSIGN/config.yaml \
  --description datasets/$ASSIGN/description.txt \
  --output workspace/$ASSIGN/scored $LLM

# 3) Markdown reports (per student + class summary)
docker compose run --rm grader python scripts/report.py \
  workspace/$ASSIGN/scored --output workspace/$ASSIGN/reports

# 4) Archive into the task bank (--max-score = the config's max_score)
docker compose up -d db
docker compose run --rm grader python scripts/db_ingest.py --key $ASSIGN \
  --title "$ASSIGN" --max-score 100 \
  --description datasets/$ASSIGN/description.txt \
  --reference datasets/$ASSIGN/reference.ipynb \
  --scored workspace/$ASSIGN/scored
```

Reports land in `workspace/$ASSIGN/reports/` (`summary.md` + one file per student).

## Grade an assignment (numeric autograder)

For "one numeric answer per question" assignments. The config uses a `questions:`
schema with per-question markers (how each question is labeled in the notebooks)
and a numerical tolerance — see [config.yaml](#configyaml).

```bash
# 1) Execute + autograde (expected answers derived by running the reference)
docker compose run --rm grader python scripts/preprocess.py \
  datasets/$ASSIGN/submissions --output workspace/$ASSIGN/processed \
  --reference datasets/$ASSIGN/reference.ipynb --config datasets/$ASSIGN/config.yaml

# 2) (optional) generate a process rubric from the description
docker compose run --rm grader python scripts/generate_rubric.py \
  datasets/$ASSIGN/description.txt --output workspace/$ASSIGN/rubric.yaml $LLM

# 3) LLM process scoring (numeric result score enforced from autograde)
docker compose run --rm grader python scripts/score_notebooks.py \
  workspace/$ASSIGN/processed --output workspace/$ASSIGN/scored \
  --reference datasets/$ASSIGN/reference.ipynb \
  --rubric workspace/$ASSIGN/rubric.yaml $LLM

# 4) reports + 5) archive — same as steps 3–4 above (add --ir workspace/$ASSIGN/processed)
```

Debug a single notebook's autograde:

```bash
docker compose run --rm grader python scripts/run_tests.py \
  datasets/$ASSIGN/submissions/<one>.ipynb \
  --reference datasets/$ASSIGN/reference.ipynb --config datasets/$ASSIGN/config.yaml
```

---

## config.yaml

`gen_config.py` writes the config from `description.txt`; always review it before
grading.

**General grader** — `problems:` schema:

```yaml
assignment: my-assignment
max_score: 100
criteria:                       # cross-cutting, applied to every problem
  - "Explanation & interpretation: explain the approach in markdown between code
     cells, comparable to the reference; deduct for code with little explanation."
problems:
  - {name: P1, points: 10, type: link, desc: "..."}   # link → full marks if a URL is present
  - {name: P2, points: 45, type: llm,  desc: "..."}   # llm  → graded against the reference
  - {name: P3, points: 45, type: llm,  desc: "..."}
```

**Numeric autograder** — `questions:` schema:

```yaml
questions:
  - {name: Q1, marker: '<student-side regex>', ref_marker: '<reference section>'}
  - {name: Q2, marker: '...',                   ref_marker: '...'}
rtol: 0.02
atol: 1.0e-8
answer_var: u                   # variable in the reference holding the answer
# expected:                     # optional hard-coded fallback if no runnable reference
```

## Reference oracle

Ground truth comes from **executing the reference solution**, not from hard-coded
values:

- **Numeric**: `run_tests.py --reference` runs the reference and reads the
  expected answer per question. Most robust convention: the reference defines
  `ANSWERS = {"Q1": ..., ...}`. Otherwise it snapshots `answer_var` at each
  question's section. Falls back to `config.yaml: expected:`.
- **General**: `score_general.py` executes the reference, captures its computed
  arrays and full markdown, and gives both to the LLM as the answer key and the
  expected level of explanation.

## LLM providers

Both graders use `scripts/llm_client.py`:

- `--provider openai` (default) — any OpenAI-compatible endpoint via `--base-url`
  + `--model`; key from `LLM_API_KEY`.
- `--provider anthropic` — native Anthropic SDK; key from `ANTHROPIC_API_KEY`.

Only the process / per-problem score and feedback come from the model; the
numeric **result** score always comes from execution.

## Task bank (Postgres + pgvector)

Every assignment's results are archived into one database
(`assignments`, `submissions`, `results`, `problem_scores`, `diagnoses`;
`assignments.embedding vector(1024)` reserved for semantic similar-problem
grouping — embeddings pluggable, null by default).

```bash
docker compose up -d db
docker compose run --rm grader python scripts/db_query.py list                # all assignments
docker compose run --rm grader python scripts/db_query.py stats  --key $ASSIGN # per-problem averages
docker compose run --rm grader python scripts/db_query.py top    --key $ASSIGN --n 5 --lowest
docker compose run --rm grader python scripts/db_query.py errors --key $ASSIGN # numeric: failure rate
```

## Identity (anonymization)

Submissions are graded under anonymous ids (`anon-001`, …). `scripts/identity.py`
extracts the real **name / student number** from filenames and notebook headers
(best effort); they show up in reports and the database. When nothing is
detectable, the original filename is kept so you can map back manually.

## Adding a new assignment

1. Create `datasets/<ASSIGN>/`; put student notebooks in `submissions/`.
2. Add `description.txt` and a correct, runnable `reference.ipynb`.
3. `gen_config.py` → `config.yaml`; review points / `type` / markers.
4. Run the matching engine above. No code changes needed.

## Limitations

- **Truth depends on the reference** — it must run cleanly and follow the answer
  convention; a wrong reference yields wrong grades.
- **LLM judgment is advisory** — review the `top --lowest` queue for disputes.
- **Execution isn't fully sandboxed** — run via Docker; treat student code as
  untrusted.
- **Identity extraction is heuristic** — review before publishing grades.

## Repository layout

```
471Grader/
├── architecture.md                # full design + component status
├── SKILL.md                       # entry point when driven as a Claude Code Skill
├── Dockerfile / docker-compose.yml# containerized run + pgvector `db` service
├── scripts/
│   ├── gen_config.py              # description.txt → config.yaml (LLM)
│   ├── preprocess.py / run_tests.py / generate_rubric.py / score_notebooks.py   # numeric engine
│   ├── score_general.py           # reference-grounded per-problem LLM grader
│   ├── report.py                  # scored JSON → Markdown reports
│   ├── identity.py                # name / student-id extraction
│   ├── llm_client.py              # provider abstraction (openai / anthropic)
│   ├── db_common.py / db_ingest.py / db_query.py / db/schema.sql                # task bank
│   └── requirements.txt
├── datasets/<ASSIGN>/             # per-assignment inputs
└── workspace/<ASSIGN>/            # generated outputs (git-ignored)
```

## License

MIT
