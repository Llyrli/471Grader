"""End-to-end grading pipeline orchestrator (the backbone an n8n/Dify node calls).

Chains the existing per-stage scripts for one assignment into a single run, in
the conventional layout:

    datasets/<ASSIGN>/{submissions/, reference.ipynb, config.yaml, description.txt}
    workspace/<ASSIGN>/{processed/, scored/, reports/, review_queue/}

Two engines (same as the two graders):
  - numeric : preprocess → score_notebooks → report   (+ optional ingest)
  - general : score_general → report                   (+ optional ingest)

What makes this orchestration-friendly (vs. a shell script):
  1. Emits a machine-readable RUN MANIFEST (JSON): per-stage status/returncode/
     duration + an AUTO/ABSTAIN review summary, so a workflow can branch on it.
  2. Exit code encodes the decision a workflow needs:
       0  → done, everything AUTO (safe to publish/ingest automatically)
       10 → done, but N submissions ABSTAINED → route to the HUMAN-APPROVAL node
       1  → a stage failed (stop the workflow / alert)
  3. `--from/--to` run a single stage, so each workflow node can own one step.

Usage:
    python pipeline.py --assign HW2 --engine numeric $LLM
    python pipeline.py --assign HW3 --engine general --ingest $LLM
    python pipeline.py --assign HW2 --engine numeric --from report --to report
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("pipeline")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

# Stage order per engine. Each engine's stages are named; --from/--to slice them.
ENGINE_STAGES = {
    "numeric": ["preprocess", "score", "report"],
    "general": ["score", "report"],
}

EXIT_OK = 0
EXIT_STAGE_FAILED = 1
EXIT_NEEDS_REVIEW = 10


# ---------------------------------------------------------------------------
# Paths + stage command construction
# ---------------------------------------------------------------------------

class Paths:
    """Conventional input/output paths for an assignment."""

    def __init__(self, assign: str, repo: Path = REPO_ROOT):
        self.assign = assign
        self.dataset = repo / "datasets" / assign
        self.submissions = self.dataset / "submissions"
        self.reference = self.dataset / "reference.ipynb"
        self.config = self.dataset / "config.yaml"
        self.description = self.dataset / "description.txt"
        ws = repo / "workspace" / assign
        self.processed = ws / "processed"
        self.scored = ws / "scored"
        self.reports = ws / "reports"
        self.review_queue = ws / "review_queue"
        self.manifest = ws / "run_manifest.json"


def _llm_flags(opts: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if opts.provider:
        flags += ["--provider", opts.provider]
    if opts.model:
        flags += ["--model", opts.model]
    if opts.base_url:
        flags += ["--base-url", opts.base_url]
    if opts.api_key:
        flags += ["--api-key", opts.api_key]
    return flags


def build_stage_cmd(stage: str, engine: str, p: Paths, opts: argparse.Namespace) -> list[str]:
    """Argv for one stage. Pure (no I/O) so it's unit-testable."""
    py = sys.executable
    llm = _llm_flags(opts)
    if engine == "numeric":
        if stage == "preprocess":
            cmd = [py, str(SCRIPTS / "preprocess.py"), str(p.submissions),
                   "--output", str(p.processed)]
            if p.reference.exists():
                cmd += ["--reference", str(p.reference)]
            if p.config.exists():
                cmd += ["--config", str(p.config)]
            return cmd
        if stage == "score":
            cmd = [py, str(SCRIPTS / "score_notebooks.py"), str(p.processed),
                   "--output", str(p.scored), "--review-queue", str(p.review_queue)]
            if p.reference.exists():
                cmd += ["--reference", str(p.reference)]
            if opts.rubric:
                cmd += ["--rubric", str(opts.rubric)]
            if opts.memory:
                cmd += ["--memory", str(opts.memory)]
            return cmd + llm
        if stage == "report":
            return [py, str(SCRIPTS / "report.py"), str(p.scored), "--output", str(p.reports)]
    elif engine == "general":
        if stage == "score":
            cmd = [py, str(SCRIPTS / "score_general.py"), str(p.submissions),
                   "--reference", str(p.reference), "--config", str(p.config),
                   "--output", str(p.scored), "--review-queue", str(p.review_queue)]
            if p.description.exists():
                cmd += ["--description", str(p.description)]
            if opts.memory:
                cmd += ["--memory", str(opts.memory)]
            return cmd + llm
        if stage == "report":
            return [py, str(SCRIPTS / "report.py"), str(p.scored), "--output", str(p.reports)]
    if stage == "ingest":
        cmd = [py, str(SCRIPTS / "db_ingest.py"), "--key", p.assign, "--title", p.assign,
               "--scored", str(p.scored)]
        if p.reference.exists():
            cmd += ["--reference", str(p.reference)]
        if p.description.exists():
            cmd += ["--description", str(p.description)]
        if engine == "numeric" and p.processed.exists():
            cmd += ["--ir", str(p.processed)]
        if opts.max_score is not None:
            cmd += ["--max-score", str(opts.max_score)]
        return cmd
    raise ValueError(f"unknown stage {stage!r} for engine {engine!r}")


def select_stages(engine: str, from_stage: str | None, to_stage: str | None) -> list[str]:
    stages = ENGINE_STAGES[engine]
    i = stages.index(from_stage) if from_stage else 0
    j = stages.index(to_stage) + 1 if to_stage else len(stages)
    if i > j:
        raise ValueError(f"--from {from_stage} comes after --to {to_stage}")
    return stages[i:j]


# ---------------------------------------------------------------------------
# Review summary (the human-approval gate signal)
# ---------------------------------------------------------------------------

def summarize_review(scored_dir: Path) -> dict:
    """Read scored JSON and split AUTO vs ABSTAIN.

    The numeric engine's selective-grading gate writes a `status` field; the
    general engine has no gate, so its records count as AUTO. Returns counts plus
    the list of abstained ids (the human-review work-list)."""
    total = auto = abstain = 0
    review: list[dict] = []
    for f in sorted(scored_dir.glob("*_scored.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        total += 1
        if rec.get("status") == "ABSTAIN":
            abstain += 1
            review.append({"student_id": rec.get("student_id", f.stem),
                           "reasons": rec.get("review_reasons", []),
                           "confidence": rec.get("confidence")})
        else:
            auto += 1
    return {"total": total, "auto": auto, "abstain": abstain, "review_queue": review}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _default_runner(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def run_pipeline(
    assign: str,
    engine: str,
    opts: argparse.Namespace,
    runner: Callable[[list[str]], tuple[int, str, str]] = _default_runner,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Run the selected stages, returning a manifest dict. `runner`/`clock` are
    injectable for testing. Stops at the first failing stage."""
    p = Paths(assign)
    stages = select_stages(engine, opts.from_stage, opts.to_stage)
    # Optional archival stage, appended only when the run reaches `report`.
    if getattr(opts, "ingest", False) and "report" in stages:
        stages = stages + ["ingest"]
    manifest = {
        "assignment": assign,
        "engine": engine,
        "stages": [],
        "status": "running",
    }
    for stage in stages:
        cmd = build_stage_cmd(stage, engine, p, opts)
        logger.info("▶ stage %s: %s", stage, " ".join(cmd[1:]))
        t0 = clock()
        rc, out, err = runner(cmd)
        dur = round(clock() - t0, 2)
        entry = {"stage": stage, "returncode": rc, "duration_s": dur,
                 "status": "ok" if rc == 0 else "failed"}
        if rc != 0:
            entry["stderr_tail"] = (err or "").strip().splitlines()[-5:]
        manifest["stages"].append(entry)
        if rc != 0:
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage
            return manifest

    # Stages succeeded → compute the review summary if scoring ran.
    ran = {s["stage"] for s in manifest["stages"]}
    if "score" in ran or p.scored.exists():
        manifest["review"] = summarize_review(p.scored)
    manifest["status"] = "needs_review" if manifest.get("review", {}).get("abstain") else "done"
    return manifest


def exit_code_for(manifest: dict) -> int:
    if manifest.get("status") == "failed":
        return EXIT_STAGE_FAILED
    if manifest.get("review", {}).get("abstain"):
        return EXIT_NEEDS_REVIEW
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end grading pipeline orchestrator.")
    p.add_argument("--assign", required=True, help="Assignment key = datasets/<ASSIGN> folder")
    p.add_argument("--engine", choices=["numeric", "general"], required=True)
    p.add_argument("--from", dest="from_stage", default=None, help="Start at this stage")
    p.add_argument("--to", dest="to_stage", default=None, help="Stop after this stage")
    p.add_argument("--rubric", type=Path, default=None, help="(numeric) rubric.yaml")
    p.add_argument("--memory", type=Path, default=None, help="Program-memory store to inject")
    p.add_argument("--ingest", action="store_true", help="Append a db_ingest stage after report")
    p.add_argument("--max-score", type=int, default=None, dest="max_score",
                   help="(ingest) assignment max score stored in the task bank")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Where to write the run manifest JSON (default: workspace/<ASSIGN>/run_manifest.json)")
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    opts = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if opts.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    manifest = run_pipeline(opts.assign, opts.engine, opts)

    out = opts.manifest or Paths(opts.assign).manifest
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rev = manifest.get("review", {})
    logger.info("Manifest → %s", out)
    if manifest["status"] == "failed":
        logger.error("Pipeline FAILED at stage '%s'", manifest.get("failed_stage"))
    else:
        logger.info("Pipeline %s — graded %d (auto=%d, abstain=%d)",
                    manifest["status"], rev.get("total", 0),
                    rev.get("auto", 0), rev.get("abstain", 0))
        if rev.get("abstain"):
            logger.info("⚑ %d submission(s) need human approval: %s",
                        rev["abstain"], [r["student_id"] for r in rev["review_queue"]])
    return exit_code_for(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
