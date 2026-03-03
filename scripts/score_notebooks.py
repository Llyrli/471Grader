"""Score preprocessed notebook IR files for ME471 HW2.

Reads processed IR JSON files (output of preprocess.py), converts autograde
pass/fail into numeric scores, calls Claude for qualitative feedback, and
writes final scored JSON to the output directory.

Scoring rules:
  - Q1, Q2, Q3: pass → 10 pts, fail → 0 pts
  - final_score = Q1_score + Q2_score + Q3_score  (max 30)
  - LLM provides feedback only — it cannot change correctness scores.

Usage:
    python score_notebooks.py <processed_dir> --output <scored_dir>
    python score_notebooks.py workspace/processed --output workspace/scored
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger("score_notebooks")

POINTS_PER_QUESTION = 10
QUESTIONS = ["Q1", "Q2", "Q3"]
DEFAULT_MODEL = "claude-sonnet-4-6"

FEEDBACK_SYSTEM_PROMPT = """\
You are a teaching assistant for ME471 (Finite Element Methods).
Your job is to give brief, specific qualitative feedback on student code.
You must NOT change correctness scores — those are determined by automated tests.
Write in the same language the student used. Be concise."""

FEEDBACK_USER_TEMPLATE = """\
Student ID: {student_id}
Execution status: {exec_status}

Autograde results:
  Q1 (Problem 2.8):  {q1_status} — {q1_details}
  Q2 (Problem 2.11): {q2_status} — {q2_details}
  Q3 (Problem 2.15): {q3_status} — {q3_details}

Submission code:
{code_text}

Write feedback in this EXACT JSON format (no markdown, no extra keys):
{{
  "Q1": "<one sentence about Q1 code quality>",
  "Q2": "<one sentence about Q2 code quality>",
  "Q3": "<one sentence about Q3 code quality>",
  "overall": "<one to two sentences overall comment>"
}}"""


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_scores(autograde: dict[str, Any]) -> dict[str, int]:
    """Convert autograde pass/fail to numeric scores."""
    scores = {}
    for q in QUESTIONS:
        passed = autograde.get(q, {}).get("passed", False)
        scores[f"{q}_score"] = POINTS_PER_QUESTION if passed else 0
    scores["final_score"] = sum(scores[f"{q}_score"] for q in QUESTIONS)
    return scores


# ---------------------------------------------------------------------------
# LLM feedback
# ---------------------------------------------------------------------------

def get_feedback(
    client: anthropic.Anthropic,
    student_id: str,
    ir: dict[str, Any],
) -> dict[str, str]:
    """Call Claude to get qualitative feedback. Returns feedback dict."""
    autograde = ir.get("autograde", {})
    exec_status = ir.get("execution_status", "unknown")

    # Extract code cells only (strip markdown cells for brevity)
    sections = ir.get("content", {}).get("sections", [])
    code_parts = [s["text"] for s in sections if s.get("level") == 2 and s.get("text")]
    code_text = "\n\n---\n\n".join(code_parts[:3]) if code_parts else "(no code extracted)"

    def _status(q):
        ag = autograde.get(q, {})
        return ("PASS" if ag.get("passed") else "FAIL"), ag.get("details", "")

    q1_s, q1_d = _status("Q1")
    q2_s, q2_d = _status("Q2")
    q3_s, q3_d = _status("Q3")

    prompt = FEEDBACK_USER_TEMPLATE.format(
        student_id=student_id,
        exec_status=exec_status,
        q1_status=q1_s, q1_details=q1_d,
        q2_status=q2_s, q2_details=q2_d,
        q3_status=q3_s, q3_details=q3_d,
        code_text=code_text[:6000],  # cap to avoid token overflow
    )

    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=512,
            system=FEEDBACK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        feedback = json.loads(raw)
        # Ensure all required keys present
        for key in ["Q1", "Q2", "Q3", "overall"]:
            if key not in feedback:
                feedback[key] = ""
        return feedback
    except Exception as exc:
        logger.warning("LLM feedback failed for %s: %s", student_id, exc)
        return {q: "Feedback unavailable." for q in ["Q1", "Q2", "Q3", "overall"]}


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def score_one(
    ir_path: Path,
    output_dir: Path,
    client: anthropic.Anthropic,
) -> None:
    """Score a single processed IR file and write scored JSON."""
    with open(ir_path, encoding="utf-8") as f:
        ir = json.load(f)

    student_id = ir.get("student_id", ir_path.stem)
    autograde = ir.get("autograde", {})
    exec_status = ir.get("execution_status", "unknown")

    # If notebook never ran, skip LLM
    if exec_status != "success":
        logger.warning("  %s: execution_failed — skipping LLM feedback", student_id)
        feedback = {q: "Notebook did not execute." for q in ["Q1", "Q2", "Q3", "overall"]}
    else:
        logger.info("  %s: requesting LLM feedback …", student_id)
        feedback = get_feedback(client, student_id, ir)

    scores = compute_scores(autograde)

    scored = {
        "student_id": student_id,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "execution_status": exec_status,
        **scores,
        "feedback": feedback,
        "autograde_detail": autograde,
    }

    out_path = output_dir / f"{student_id}_scored.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    logger.info("  → %s  (final_score=%d/30)", out_path.name, scores["final_score"])


def run_batch(processed_dir: Path, output_dir: Path, api_key: str | None = None) -> None:
    """Score all processed IR files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ir_files = sorted(processed_dir.glob("anon-*.json"))
    if not ir_files:
        logger.warning("No processed IR files found in %s", processed_dir)
        return

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(
            "No API key found. Pass --api-key <key> or set ANTHROPIC_API_KEY."
        )
        raise SystemExit(1)

    client = anthropic.Anthropic(api_key=api_key)
    logger.info("Scoring %d submission(s) …", len(ir_files))

    for ir_path in ir_files:
        logger.info("[%s]", ir_path.name)
        try:
            score_one(ir_path, output_dir, client)
        except Exception as exc:
            logger.error("  Failed to score %s: %s", ir_path.name, exc)

    logger.info("Done. Results in %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score preprocessed notebook IRs for ME471 HW2.",
    )
    parser.add_argument("processed_dir", type=Path,
                        help="Directory containing anon-*.json IR files")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        dest="output_dir",
                        help="Directory to write scored JSON files")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not args.processed_dir.is_dir():
        logger.error("processed_dir does not exist: %s", args.processed_dir)
        raise SystemExit(1)
    run_batch(args.processed_dir, args.output_dir, api_key=args.api_key)


if __name__ == "__main__":
    main()
