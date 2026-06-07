"""
Preprocess .ipynb homework submissions into IR JSON format.

Usage:
    python preprocess.py <input_dir> --output <output_dir>
    python preprocess.py --help

Requires Python 3.10+.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import nbclient
import nbformat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".ipynb"}
NOTEBOOK_TIMEOUT = 120  # seconds per notebook

logger = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Section(NamedTuple):
    """A section parsed from notebook cells."""
    heading: str
    level: int
    text: str


@dataclass
class ProcessingLogEntry:
    """A single entry in the processing log."""
    step: str
    status: str  # success | warning | error | skipped
    duration_ms: int = 0
    message: str = ""


@dataclass
class ProcessingContext:
    """Mutable accumulator for processing a single file."""
    log: list[ProcessingLogEntry] = field(default_factory=list)

    def add_log(
        self,
        step: str,
        status: str,
        duration_ms: int = 0,
        message: str = "",
    ) -> None:
        self.log.append(ProcessingLogEntry(
            step=step, status=status,
            duration_ms=duration_ms, message=message,
        ))


# ---------------------------------------------------------------------------
# Notebook extraction
# ---------------------------------------------------------------------------

def extract_ipynb(file_path: Path) -> tuple[str, list[Section]]:
    """Parse a .ipynb file and extract full text and sections.

    Returns (full_text, sections) where full_text includes markdown + code.
    """
    nb = nbformat.read(str(file_path), as_version=4)

    lines: list[str] = []
    sections: list[Section] = []

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            text = cell.source.strip()
            if text:
                lines.append(text)
                sections.append(Section(heading=text[:80], level=1, text=text))
        elif cell.cell_type == "code":
            code = cell.source.strip()
            if code:
                lines.append(f"```python\n{code}\n```")
                sections.append(Section(heading="", level=2, text=code))

    full_text = "\n\n".join(lines)
    return full_text, sections


# ---------------------------------------------------------------------------
# Notebook execution
# ---------------------------------------------------------------------------

def execute_notebook(file_path: Path) -> tuple[str, dict | None]:
    """Execute a notebook using nbclient.

    Returns (status, error_info).
      status:     'success' | 'execution_failed'
      error_info: None on success, else {'type': ..., 'message': ...}
    """
    nb = nbformat.read(str(file_path), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=NOTEBOOK_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(file_path.parent)}},
    )
    try:
        client.execute()
        return "success", None
    except Exception as exc:
        return "execution_failed", {
            "type": type(exc).__name__,
            "message": str(exc),
        }


# ---------------------------------------------------------------------------
# Autograde test integration
# ---------------------------------------------------------------------------

def run_autograde_tests(
    file_path: Path,
    reference: Path | None = None,
    config: Path | None = None,
) -> dict[str, Any]:
    """Run autograde tests via scripts/run_tests.py.

    Returns dict with per-question pass/fail results. ``reference`` enables the
    reference oracle (derive expected answers by executing it); ``config`` selects
    a per-assignment config.yaml (question markers, tolerance, ...).
    """
    run_tests_script = Path(__file__).parent / "run_tests.py"
    cmd = [sys.executable, str(run_tests_script), str(file_path)]
    if reference is not None:
        cmd += ["--reference", str(reference)]
    if config is not None:
        cmd += ["--config", str(config)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        msg = result.stderr.strip() or "run_tests.py failed"
        return {q: {"passed": False, "details": msg} for q in ["Q1", "Q2", "Q3"]}
    except Exception as exc:
        return {q: {"passed": False, "details": str(exc)} for q in ["Q1", "Q2", "Q3"]}


def _format_autograde_block(ag: dict[str, Any]) -> str:
    """Format autograde results as the required text block."""
    lines = ["[AUTOGRADE TEST RESULTS]"]
    for q in ["Q1", "Q2", "Q3"]:
        result = ag.get(q, {})
        status = "pass" if result.get("passed") else "fail"
        details = result.get("details", "")
        line = f"{q}: {status}"
        if details:
            line += f" ({details})"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IR assembly
# ---------------------------------------------------------------------------

def build_notebook_ir(
    student_id: str,
    source_files: list[str],
    full_text: str,
    sections: list[Section],
    execution_status: str,
    execution_error: dict | None,
    autograde: dict[str, Any],
    processing_log: list[ProcessingLogEntry],
) -> dict[str, Any]:
    """Assemble IR JSON dict for a notebook submission."""
    full_text_with_ag = full_text + "\n\n" + _format_autograde_block(autograde)
    return {
        "student_id": student_id,
        "submission_type": "notebook",
        "source_files": source_files,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "execution_status": execution_status,
        "execution_error": execution_error,
        "metadata": {
            "cell_count": len(sections),
            "language": "python",
        },
        "content": {
            "full_text": full_text_with_ag,
            "sections": [
                {"heading": s.heading, "level": s.level, "text": s.text}
                for s in sections
            ],
        },
        "autograde": autograde,
        "processing_log": [
            {
                "step": e.step,
                "status": e.status,
                "duration_ms": e.duration_ms,
                "message": e.message,
            }
            for e in processing_log
        ],
    }


def _build_error_ir(
    student_id: str,
    source_files: list[str],
    error_type: str,
    error_message: str,
    log_entries: list[ProcessingLogEntry],
) -> dict[str, Any]:
    """Build a minimal IR for a submission that failed preprocessing."""
    failed_ag = {
        q: {"passed": False, "details": "execution_failed"}
        for q in ["Q1", "Q2", "Q3"]
    }
    return {
        "student_id": student_id,
        "submission_type": "notebook",
        "source_files": source_files,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "execution_status": error_type,
        "execution_error": {"type": error_type, "message": error_message},
        "metadata": {
            "cell_count": 0,
            "language": "python",
            "error_type": error_type,
        },
        "content": {"full_text": "", "sections": []},
        "autograde": failed_ag,
        "processing_log": [
            {
                "step": e.step,
                "status": e.status,
                "duration_ms": e.duration_ms,
                "message": e.message,
            }
            for e in log_entries
        ],
    }


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _timed(func, *args, **kwargs) -> tuple[Any, int]:
    """Run func and return (result, elapsed_ms)."""
    t0 = time.monotonic()
    result = func(*args, **kwargs)
    return result, int((time.monotonic() - t0) * 1000)


def process_ipynb_file(
    file_path: Path,
    student_id: str,
    reference: Path | None = None,
    config: Path | None = None,
) -> dict[str, Any] | None:
    """Process a single .ipynb file into an IR dict."""
    ctx = ProcessingContext()
    source_files = [f"raw/{file_path.name}"]

    # Step 1: Extract text
    try:
        (full_text, sections), dur = _timed(extract_ipynb, file_path)
        ctx.add_log("text_extraction", "success", duration_ms=dur,
                    message=f"{len(sections)} cells extracted")
    except Exception as exc:
        ctx.add_log("text_extraction", "error", message=str(exc))
        return _build_error_ir(student_id, source_files, "parse_failed", str(exc), ctx.log)

    # Step 2: Execute notebook
    (exec_status, exec_error), dur = _timed(execute_notebook, file_path)
    ctx.add_log(
        "notebook_execution",
        "success" if exec_status == "success" else "error",
        duration_ms=dur,
        message=str(exec_error) if exec_error else "",
    )
    logger.info("  Execution: %s", exec_status)

    # Step 3: Autograde tests (always run — uses exec()+Agg, independent of nbclient)
    autograde, dur = _timed(run_autograde_tests, file_path, reference, config)
    ctx.add_log("autograde_tests", "success", duration_ms=dur)

    # Step 4: Build IR
    ir = build_notebook_ir(
        student_id=student_id,
        source_files=source_files,
        full_text=full_text,
        sections=sections,
        execution_status=exec_status,
        execution_error=exec_error,
        autograde=autograde,
        processing_log=ctx.log,
    )
    return ir


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def generate_student_id(index: int) -> str:
    """Generate an anonymous student ID: anon-001, anon-002, etc."""
    return f"anon-{index:03d}"


def collect_files(input_dir: Path) -> list[Path]:
    """Collect all .ipynb files from input_dir (non-recursive)."""
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def run_batch(input_dir: Path, output_dir: Path, reference: Path | None = None,
              config: Path | None = None) -> None:
    """Preprocess all .ipynb files in input_dir, write IR to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(input_dir)
    if not files:
        logger.warning("No .ipynb files found in %s", input_dir)
        return

    logger.info("Found %d notebook(s) in %s", len(files), input_dir)
    success_count = 0
    fail_count = 0

    for idx, file_path in enumerate(files, start=1):
        student_id = generate_student_id(idx)
        logger.info("[%d/%d] Processing %s → %s",
                    idx, len(files), file_path.name, student_id)

        ir = process_ipynb_file(file_path, student_id, reference, config)

        if ir is not None:
            out_path = output_dir / f"{student_id}.json"
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(ir, f, ensure_ascii=False, indent=2)
                success_count += 1
                logger.info("  → Wrote %s", out_path.name)
            except OSError as exc:
                fail_count += 1
                logger.error("  → Failed to write %s: %s", out_path.name, exc)
        else:
            fail_count += 1
            logger.error("  → Failed to process %s", file_path.name)

    logger.info("=" * 60)
    logger.info("Preprocessing complete: %d OK, %d failed", success_count, fail_count)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess .ipynb submissions into IR JSON.",
    )
    parser.add_argument("input_dir", type=Path,
                        help="Directory containing .ipynb files")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        dest="output_dir",
                        help="Directory to write IR JSON files")
    parser.add_argument("--reference", "-r", type=Path, default=None,
                        help="Reference .ipynb — execute it to derive expected "
                             "answers (reference oracle) instead of hardcoded constants")
    parser.add_argument("--config", "-c", type=Path, default=None,
                        help="Per-assignment config.yaml (question markers, tolerance, ...)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug-level logging")
    return parser.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if not args.input_dir.is_dir():
        logger.error("Input directory does not exist: %s", args.input_dir)
        raise SystemExit(1)

    run_batch(input_dir=args.input_dir, output_dir=args.output_dir,
              reference=args.reference, config=args.config)


if __name__ == "__main__":
    main()
