"""Ingest an assignment + its graded submissions into the Postgres task bank.

Applies the schema (idempotent), upserts the assignment (description, rubric,
derived expected answers, reference), then upserts every scored submission with
its result and per-question diagnoses.

Usage:
    python db_ingest.py \
        --key ME471-HW2 --title "ME471 HW2 (1D bar FEM)" \
        --description workspace/assignment_description.txt \
        --rubric workspace/rubric.yaml \
        --reference workspace/correct_sample.ipynb \
        --scored workspace/scored \
        --ir workspace/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from psycopg.types.json import Json

import db_common

logger = logging.getLogger("db_ingest")
QUESTIONS = ["Q1", "Q2", "Q3"]


def _load_expected(reference: Path | None) -> dict[str, list] | None:
    if reference is None:
        return None
    try:
        from run_tests import derive_expected
        exp = derive_expected(reference)
        return {q: v.tolist() for q, v in exp.items()}
    except Exception as exc:
        logger.warning("Could not derive expected from reference: %s", exc)
        return None


def upsert_assignment(conn, args) -> int:
    description = args.description.read_text(encoding="utf-8") if args.description else None
    rubric = yaml.safe_load(args.rubric.read_text(encoding="utf-8")) if args.rubric else None
    expected = _load_expected(args.reference)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO assignments (key, title, description, rubric, expected,
                                     reference_path, max_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (key) DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                rubric = EXCLUDED.rubric,
                expected = EXCLUDED.expected,
                reference_path = EXCLUDED.reference_path,
                max_score = EXCLUDED.max_score
            RETURNING id
            """,
            (
                args.key, args.title, description,
                Json(rubric) if rubric is not None else None,
                Json(expected) if expected is not None else None,
                str(args.reference) if args.reference else None,
                args.max_score,
            ),
        )
        assignment_id = cur.fetchone()[0]
    conn.commit()
    logger.info("Assignment '%s' upserted (id=%d)", args.key, assignment_id)
    return assignment_id


def ingest_one(conn, assignment_id: int, scored: dict[str, Any], ir: dict[str, Any] | None) -> None:
    sid = scored.get("student_id", "unknown")
    exec_status = scored.get("execution_status") or (ir or {}).get("execution_status")
    source_file = scored.get("source_file")
    if source_file is None and ir:
        files = ir.get("source_files") or []
        source_file = files[0] if files else None

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO submissions (assignment_id, student_id, source_file,
                                     execution_status, student_name, student_no)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (assignment_id, student_id) DO UPDATE SET
                source_file = EXCLUDED.source_file,
                execution_status = EXCLUDED.execution_status,
                student_name = EXCLUDED.student_name,
                student_no = EXCLUDED.student_no
            RETURNING id
            """,
            (assignment_id, sid, source_file, exec_status,
             scored.get("name"), scored.get("student_no")),
        )
        submission_id = cur.fetchone()[0]

        is_general = "problems" in scored
        cur.execute(
            """
            INSERT INTO results (submission_id,
                q1_result,q1_process,q1_score,
                q2_result,q2_process,q2_score,
                q3_result,q3_process,q3_score,
                final_score, max_score, autograde, scored_at)
            VALUES (%s, %s,%s,%s, %s,%s,%s, %s,%s,%s, %s, %s, %s, %s)
            ON CONFLICT (submission_id) DO UPDATE SET
                q1_result=EXCLUDED.q1_result, q1_process=EXCLUDED.q1_process, q1_score=EXCLUDED.q1_score,
                q2_result=EXCLUDED.q2_result, q2_process=EXCLUDED.q2_process, q2_score=EXCLUDED.q2_score,
                q3_result=EXCLUDED.q3_result, q3_process=EXCLUDED.q3_process, q3_score=EXCLUDED.q3_score,
                final_score=EXCLUDED.final_score, max_score=EXCLUDED.max_score,
                autograde=EXCLUDED.autograde, scored_at=EXCLUDED.scored_at
            RETURNING id
            """,
            (
                submission_id,
                scored.get("Q1_result_score"), scored.get("Q1_process_score"), scored.get("Q1_score"),
                scored.get("Q2_result_score"), scored.get("Q2_process_score"), scored.get("Q2_score"),
                scored.get("Q3_result_score"), scored.get("Q3_process_score"), scored.get("Q3_score"),
                scored.get("final_score"),
                scored.get("max_score", 30),
                Json(scored.get("autograde_detail")) if scored.get("autograde_detail") else None,
                scored.get("scored_at"),
            ),
        )
        result_id = cur.fetchone()[0]

        # General per-problem breakdown → problem_scores
        if is_general:
            for p in scored.get("problems", []):
                cur.execute(
                    """
                    INSERT INTO problem_scores (result_id, name, max, score, feedback)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (result_id, name) DO UPDATE SET
                        max=EXCLUDED.max, score=EXCLUDED.score, feedback=EXCLUDED.feedback
                    """,
                    (result_id, p.get("name"), p.get("max"), p.get("score"), p.get("feedback")),
                )

        # Written feedback (per question/problem + overall) → diagnoses
        feedback = scored.get("feedback", {}) or {}
        for q, text in feedback.items():
            if not text:
                continue
            cur.execute(
                """
                INSERT INTO diagnoses (result_id, question, feedback)
                VALUES (%s, %s, %s)
                ON CONFLICT (result_id, question) DO UPDATE SET feedback = EXCLUDED.feedback
                """,
                (result_id, q, text),
            )
    conn.commit()
    logger.info("  %s ingested (final=%s/%s)", sid,
                scored.get("final_score"), scored.get("max_score", 30))


def run(args) -> None:
    conn = db_common.connect()
    db_common.apply_schema(conn)
    assignment_id = upsert_assignment(conn, args)

    scored_files = sorted(args.scored.glob("*_scored.json"))
    if not scored_files:
        logger.warning("No *_scored.json in %s", args.scored)
        return

    ir_dir = args.ir
    for path in scored_files:
        scored = json.loads(path.read_text(encoding="utf-8"))
        sid = scored.get("student_id", path.stem.replace("_scored", ""))
        ir = None
        if ir_dir:
            ir_path = ir_dir / f"{sid}.json"
            if ir_path.exists():
                ir = json.loads(ir_path.read_text(encoding="utf-8"))
        ingest_one(conn, assignment_id, scored, ir)

    if getattr(args, "embed", False):
        try:
            from embeddings import Embedder, to_pgvector
            embedder = Embedder(provider=args.embed_provider, model=args.embed_model,
                                api_key=getattr(args, "api_key", None),
                                base_url=getattr(args, "base_url", None))
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(title,''), COALESCE(description,'') "
                            "FROM assignments WHERE id = %s", (assignment_id,))
                t, d = cur.fetchone()
                vec = embedder.embed_one(f"{t}\n\n{d}".strip())
                cur.execute("UPDATE assignments SET embedding = %s::vector WHERE id = %s",
                            (to_pgvector(vec), assignment_id))
            conn.commit()
            logger.info("Embedded assignment '%s' (provider=%s)", args.key, embedder.provider)
        except Exception as exc:
            logger.warning("Embedding skipped: %s", exc)

    conn.close()
    logger.info("Done. Ingested %d submission(s) for '%s'.", len(scored_files), args.key)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ingest an assignment + graded submissions into Postgres.")
    p.add_argument("--key", required=True, help="Assignment key, e.g. ME471-HW2")
    p.add_argument("--title", default=None)
    p.add_argument("--description", type=Path, default=None)
    p.add_argument("--rubric", type=Path, default=None)
    p.add_argument("--reference", type=Path, default=None,
                   help="Reference .ipynb — expected answers derived and stored")
    p.add_argument("--scored", type=Path, required=True, help="Dir with *_scored.json")
    p.add_argument("--ir", type=Path, default=None, help="Dir with IR anon-NNN.json (optional)")
    p.add_argument("--max-score", type=int, default=30)
    p.add_argument("--embed", action="store_true",
                   help="Compute + store the assignment embedding (pgvector) after ingest")
    p.add_argument("--embed-provider", choices=["hash", "openai"], default="hash")
    p.add_argument("--embed-model", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run(args)


if __name__ == "__main__":
    main()
