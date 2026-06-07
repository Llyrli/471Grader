-- JN Grader task-bank schema (Postgres + pgvector).
-- Idempotent: safe to run on every ingest.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per assignment/problem set: the "task" + its grading standard.
CREATE TABLE IF NOT EXISTS assignments (
    id             SERIAL PRIMARY KEY,
    key            TEXT UNIQUE NOT NULL,          -- e.g. 'ME471-HW2'
    title          TEXT,
    description    TEXT,                           -- assignment_description.txt
    rubric         JSONB,                          -- generated process criteria
    expected       JSONB,                          -- {"Q1":[...], "Q2":[...], ...}
    reference_path TEXT,                            -- correct_sample.ipynb
    max_score      INT DEFAULT 30,
    embedding      vector(1024),                    -- nullable; for semantic grouping/dedup
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- One row per student submission to an assignment (anonymized id).
CREATE TABLE IF NOT EXISTS submissions (
    id               SERIAL PRIMARY KEY,
    assignment_id    INT NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
    student_id       TEXT NOT NULL,                -- anon-NNN
    source_file      TEXT,
    execution_status TEXT,
    created_at       TIMESTAMPTZ DEFAULT now(),
    UNIQUE (assignment_id, student_id)
);

-- One row per scored submission (result + process breakdown).
CREATE TABLE IF NOT EXISTS results (
    id            SERIAL PRIMARY KEY,
    submission_id INT NOT NULL UNIQUE REFERENCES submissions(id) ON DELETE CASCADE,
    q1_result INT, q1_process INT, q1_score INT,
    q2_result INT, q2_process INT, q2_score INT,
    q3_result INT, q3_process INT, q3_score INT,
    final_score INT,
    autograde   JSONB,                             -- per-question pass/fail + details
    scored_at   TIMESTAMPTZ
);

-- One row per question's written diagnosis (Q1/Q2/Q3/overall).
CREATE TABLE IF NOT EXISTS diagnoses (
    id         SERIAL PRIMARY KEY,
    result_id  INT NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    question   TEXT NOT NULL,                       -- Q1 | Q2 | Q3 | overall
    feedback   TEXT,
    UNIQUE (result_id, question)
);

-- Best-effort real identity extracted from filename/content (anon-NNN ↔ student).
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS student_name TEXT;
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS student_no TEXT;

-- General per-problem scores (for assignments graded by score_general.py, where
-- problem count/points differ from the HW2 q1/q2/q3 shape). results.q*_* stay
-- NULL for these; per-problem rows live here instead.
ALTER TABLE results ADD COLUMN IF NOT EXISTS max_score INT;

CREATE TABLE IF NOT EXISTS problem_scores (
    id         SERIAL PRIMARY KEY,
    result_id  INT NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,                       -- P1 / P2 / P3 ...
    max        INT,
    score      INT,
    feedback   TEXT,
    UNIQUE (result_id, name)
);

CREATE INDEX IF NOT EXISTS idx_submissions_assignment ON submissions(assignment_id);
CREATE INDEX IF NOT EXISTS idx_results_final ON results(final_score);
CREATE INDEX IF NOT EXISTS idx_problem_scores_result ON problem_scores(result_id);
