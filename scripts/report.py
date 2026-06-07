"""Generate Markdown grading reports from scored notebook JSON files.

Consumes the scored JSON produced by score_notebooks.py and writes:
  - one per-submission Markdown report  (anon-NNN.md)
  - one class summary table             (summary.md)

This is the report stage of JN Grader (architecture.md §5.9 / §8).

Usage:
    python report.py <scored_dir> --output <reports_dir>

Example:
    python report.py workspace/scored --output workspace/reports
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("report")

QUESTIONS = ["Q1", "Q2", "Q3"]
MAX_TOTAL = 30


# ---------------------------------------------------------------------------
# Per-submission report
# ---------------------------------------------------------------------------

def _question_block(scored: dict[str, Any], q: str) -> str:
    """Render one question's result + process + diagnosis."""
    result = scored.get(f"{q}_result_score", 0)
    process = scored.get(f"{q}_process_score", 0)
    qi = scored.get(f"{q}_score", result + process)

    ag = scored.get("autograde_detail", {}).get(q, {})
    status = "pass" if ag.get("passed") else "fail"
    details = ag.get("details", "")

    feedback = scored.get("feedback", {}).get(q, "")

    lines = [
        f"### {q} — {qi}/10  (result {result}/3 + process {process}/7)",
        f"- **自动判分:** {status}" + (f"  ({details})" if details else ""),
    ]
    if feedback:
        lines.append(f"- **诊断:** {feedback}")
    return "\n".join(lines)


def _identity_label(scored: dict[str, Any]) -> str:
    """'Name (student_no)' / 'Name' / 'student_no' / '' from a scored record."""
    name = (scored.get("name") or "").strip()
    no = (scored.get("student_no") or "").strip()
    if name and no:
        return f"{name} ({no})"
    return name or no or ""


def render_submission_general(scored: dict[str, Any]) -> str:
    """Render a report for the general per-problem format (has 'problems')."""
    sid = scored.get("student_id", "unknown")
    final = scored.get("final_score", 0)
    mx = scored.get("max_score", 0)
    ident = _identity_label(scored)
    head = f"## {sid}" + (f" — {ident}" if ident else "") + f" — {final}/{mx}"
    parts = [
        head,
        f"- 作业: {scored.get('assignment', '')}",
        f"- 来源文件: {scored.get('source_file', '')}",
        f"- 评分时间: {scored.get('scored_at', 'n/a')}",
        "",
    ]
    for p in scored.get("problems", []):
        parts.append(f"### {p['name']} — {p['score']}/{p['max']}")
        if p.get("feedback"):
            parts.append(f"- {p['feedback']}")
        parts.append("")
    overall = scored.get("feedback", {}).get("overall", "")
    if overall:
        parts += ["### 总体评价", overall, ""]
    return "\n".join(parts)


def render_submission(scored: dict[str, Any]) -> str:
    """Render a full Markdown report for one scored submission."""
    if "problems" in scored:
        return render_submission_general(scored)
    sid = scored.get("student_id", "unknown")
    final = scored.get("final_score", 0)
    exec_status = scored.get("execution_status", "unknown")

    parts = [
        f"## {sid} — {final}/{MAX_TOTAL}",
        f"- 执行状态: `{exec_status}`",
        f"- 评分时间: {scored.get('scored_at', 'n/a')}",
        "",
    ]
    for q in QUESTIONS:
        parts.append(_question_block(scored, q))
        parts.append("")

    overall = scored.get("feedback", {}).get("overall", "")
    if overall:
        parts.append("### 总体评价")
        parts.append(overall)
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Class summary
# ---------------------------------------------------------------------------

def render_summary_general(records: list[dict[str, Any]]) -> str:
    """Class summary for the general per-problem format."""
    probs = [p["name"] for p in records[0].get("problems", [])]
    mx = records[0].get("max_score", 0)
    header = "| 学生 | 姓名/学号 | " + " | ".join(probs) + " | 总分 | 来源文件 |"
    sep = "|" + "---|" * (len(probs) + 4)
    lines = [f"# 班级评分汇总", "", f"共 {len(records)} 份提交,满分 {mx}。", "", header, sep]
    finals = []
    for r in sorted(records, key=lambda x: x.get("student_id", "")):
        finals.append(r.get("final_score", 0))
        by = {p["name"]: p["score"] for p in r.get("problems", [])}
        row = " | ".join(str(by.get(p, "")) for p in probs)
        lines.append(
            f"| {r.get('student_id','?')} | {_identity_label(r) or '—'} | {row} "
            f"| **{r.get('final_score',0)}** | {r.get('source_file','')} |"
        )
    if finals:
        mean = sum(finals) / len(finals)
        lines += ["", f"- 平均分: {mean:.1f}/{mx}", f"- 最高 / 最低: {max(finals)} / {min(finals)}"]
    return "\n".join(lines)


def render_summary(records: list[dict[str, Any]]) -> str:
    """Render a class-level summary table over all scored submissions."""
    if records and "problems" in records[0]:
        return render_summary_general(records)
    lines = [
        "# 班级评分汇总",
        "",
        f"共 {len(records)} 份提交,满分 {MAX_TOTAL}。",
        "",
        "| 学生 | Q1 | Q2 | Q3 | 总分 | 执行状态 |",
        "|---|---|---|---|---|---|",
    ]
    finals: list[int] = []
    for r in sorted(records, key=lambda x: x.get("student_id", "")):
        sid = r.get("student_id", "?")
        final = r.get("final_score", 0)
        finals.append(final)
        lines.append(
            f"| {sid} "
            f"| {r.get('Q1_score', 0)} "
            f"| {r.get('Q2_score', 0)} "
            f"| {r.get('Q3_score', 0)} "
            f"| **{final}** "
            f"| `{r.get('execution_status', '?')}` |"
        )

    if finals:
        mean = sum(finals) / len(finals)
        lines += [
            "",
            f"- 平均分: {mean:.1f}/{MAX_TOTAL}",
            f"- 最高 / 最低: {max(finals)} / {min(finals)}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def run(scored_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(scored_dir.glob("*_scored.json"))
    if not files:
        logger.warning("No *_scored.json files found in %s", scored_dir)
        return

    records: list[dict[str, Any]] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                scored = json.load(f)
        except Exception as exc:
            logger.error("Could not read %s: %s", path.name, exc)
            continue

        records.append(scored)
        sid = scored.get("student_id", path.stem)
        out_path = output_dir / f"{sid}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(render_submission(scored))
        logger.info("  → %s  (final=%s/%s)", out_path.name,
                    scored.get("final_score", "?"), scored.get("max_score", MAX_TOTAL))

    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(render_summary(records))
    logger.info("Summary → %s  (%d submissions)", summary_path.name, len(records))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown reports from scored notebook JSON.",
    )
    parser.add_argument("scored_dir", type=Path,
                        help="Directory containing *_scored.json files")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        dest="output_dir",
                        help="Directory to write Markdown reports")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not args.scored_dir.is_dir():
        logger.error("scored_dir does not exist: %s", args.scored_dir)
        raise SystemExit(1)
    run(args.scored_dir, args.output_dir)


if __name__ == "__main__":
    main()
