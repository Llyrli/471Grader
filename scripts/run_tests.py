"""Autograde tests for ME471 HW2 notebook submissions.

Strategy:
  1. Scan cells for problem markers ("2.8", "2.11", "2.15") to identify
     which code cell belongs to Q1 / Q2 / Q3.
  2. Execute cells in order; after each identified problem cell,
     search the namespace for any numpy array matching the expected answer.
  3. Does NOT depend on variable names — works for any coding style.

Usage:
    python run_tests.py <notebook.ipynb>
Output: JSON with Q1/Q2/Q3 pass/fail to stdout.
"""

import argparse
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # suppress all plots before any pyplot import

import numpy as np
import nbformat
import yaml

# ---------------------------------------------------------------------------
# Expected answers  (update to match official solutions)
# ---------------------------------------------------------------------------

Q1_EXPECTED = np.array([0.0, 0.5, 1.0])          # Problem 2.8  (in)
Q2_EXPECTED = np.array([0.0, 0.015, 0.02])        # Problem 2.11 (m)
Q3_EXPECTED = np.array([0.0, 0.0, 0.002, 0.0])   # Problem 2.15 (m)

RTOL = 0.02   # 2% relative tolerance
ATOL = 1e-8

EXPECTED = {"Q1": Q1_EXPECTED, "Q2": Q2_EXPECTED, "Q3": Q3_EXPECTED}

# ---------------------------------------------------------------------------
# Reference oracle: derive expected answers by EXECUTING a reference solution,
# instead of relying on the hardcoded constants above.
#
# Answer-designation convention (either works):
#   1) The reference defines a top-level dict ``ANSWERS = {"Q1": ..., ...}``.
#   2) Otherwise: the answer for each problem is the nodal-displacement vector
#      ``u``, and problems are delimited by ``2_8`` / ``2_11`` / ``2_15``
#      markdown section headers (underscore form — avoids colliding with a
#      title cell that lists "2.8, 2.11, 2.15").
# ---------------------------------------------------------------------------

# Underscore-form section markers for the REFERENCE (disambiguates from the
# dotted "2.8" that may appear in a title/intro cell).
REFERENCE_SECTION_PATTERNS = {
    "Q1": re.compile(r"2_8\b"),
    "Q2": re.compile(r"2_11\b"),
    "Q3": re.compile(r"2_15\b"),
}
ANSWER_VAR = "u"  # convention: nodal displacements live in `u`


def _reference_section_starts(cells) -> list[tuple[str, int]]:
    """Return [(Q, cell_index)] for each problem's section header, in order."""
    seen: dict[str, int] = {}
    for i, cell in enumerate(cells):
        if cell.cell_type != "markdown":
            continue
        for q, pat in REFERENCE_SECTION_PATTERNS.items():
            if q not in seen and pat.search(cell.source):
                seen[q] = i
    return sorted(seen.items(), key=lambda kv: kv[1])


def derive_expected(ref_path: Path) -> dict[str, np.ndarray]:
    """Execute a reference notebook and return {Q: expected_array}.

    Prefers an explicit ``ANSWERS`` dict; otherwise snapshots ``u`` at each
    problem-section boundary. Raises ValueError if nothing can be derived.
    """
    nb = nbformat.read(str(ref_path), as_version=4)
    cells = nb.cells
    bounds = _reference_section_starts(cells)
    order = [q for q, _ in bounds]
    starts = {idx for _, idx in bounds}
    start_to_q = {idx: q for q, idx in bounds}

    ns: dict = {"__builtins__": __builtins__}
    snapshots: dict[str, np.ndarray] = {}
    prev_q: str | None = None

    for i, cell in enumerate(cells):
        # Crossing into a new section snapshots the previous section's answer.
        if i in starts:
            if prev_q is not None and prev_q not in snapshots and ANSWER_VAR in ns:
                snapshots[prev_q] = np.asarray(ns[ANSWER_VAR], dtype=float).ravel()
            prev_q = start_to_q[i]
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        err = _exec_safe(cell.source, ns)
        if err:
            # Reference must run cleanly; surface the failure loudly.
            raise ValueError(f"reference cell {i} failed: {err}")

    # Last section.
    if order and order[-1] not in snapshots and ANSWER_VAR in ns:
        snapshots[order[-1]] = np.asarray(ns[ANSWER_VAR], dtype=float).ravel()

    # Explicit ANSWERS dict takes precedence over the heuristic snapshots.
    explicit = ns.get("ANSWERS")
    expected: dict[str, np.ndarray] = dict(snapshots)
    if isinstance(explicit, dict):
        for q, val in explicit.items():
            try:
                expected[q] = np.asarray(val, dtype=float).ravel()
            except Exception:
                continue

    if not expected:
        raise ValueError(
            "could not derive any expected answers from reference "
            "(define ANSWERS={...} or use 2_8/2_11/2_15 sections with `u`)"
        )
    return expected

# ---------------------------------------------------------------------------
# Regex patterns to identify problem cells
# ---------------------------------------------------------------------------

PROBLEM_PATTERNS = {
    "Q1": re.compile(r"2[._]8\b|problem.{0,5}2\.8", re.IGNORECASE),
    "Q2": re.compile(r"2[._]11\b|problem.{0,5}2\.11", re.IGNORECASE),
    "Q3": re.compile(r"2[._]15\b|problem.{0,5}2\.15", re.IGNORECASE),
}

# Ordered question list (overridden by --config).
QUESTIONS = ["Q1", "Q2", "Q3"]


# ---------------------------------------------------------------------------
# Per-assignment config (overrides the HW2 defaults above)
# ---------------------------------------------------------------------------

def apply_config(path: Path) -> None:
    """Load a per-assignment config.yaml and override the module defaults.

    Schema (see datasets/<HW>/config.yaml):
        questions:
          - {name: Q1, marker: "2[._]8",  ref_marker: "2_8"}
          - {name: Q2, marker: "2[._]11", ref_marker: "2_11"}
        rtol: 0.02
        atol: 1.0e-8
        answer_var: u
        expected:                 # optional hardcoded fallback (no --reference)
          Q1: [0.0, 0.5, 1.0]
    """
    global QUESTIONS, PROBLEM_PATTERNS, REFERENCE_SECTION_PATTERNS
    global RTOL, ATOL, ANSWER_VAR, EXPECTED

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = cfg.get("questions") or []
    if not questions:
        raise ValueError(f"{path}: 'questions' is required")

    QUESTIONS = [q["name"] for q in questions]
    PROBLEM_PATTERNS = {
        q["name"]: re.compile(q["marker"], re.IGNORECASE)
        for q in questions if q.get("marker")
    }
    REFERENCE_SECTION_PATTERNS = {
        q["name"]: re.compile(q["ref_marker"])
        for q in questions if q.get("ref_marker")
    }
    if "rtol" in cfg:
        RTOL = float(cfg["rtol"])
    if "atol" in cfg:
        ATOL = float(cfg["atol"])
    if cfg.get("answer_var"):
        ANSWER_VAR = cfg["answer_var"]
    if isinstance(cfg.get("expected"), dict):
        EXPECTED = {k: np.asarray(v, dtype=float).ravel() for k, v in cfg["expected"].items()}


def _find_problem_cells(nb) -> dict[str, int]:
    """Return {Q_name: cell_index} by looking for 2.8/2.11/2.15 markers.

    Checks code cell source directly (e.g. '#2.8' comment) OR markdown
    cells (uses the next non-empty code cell as the problem cell).
    """
    mapping: dict[str, int] = {}
    cells = nb.cells
    for i, cell in enumerate(cells):
        src = cell.source
        for q, pat in PROBLEM_PATTERNS.items():
            if q in mapping:
                continue
            if not pat.search(src):
                continue
            if cell.cell_type == "code" and src.strip():
                mapping[q] = i
            elif cell.cell_type == "markdown":
                for j in range(i + 1, len(cells)):
                    if cells[j].cell_type == "code" and cells[j].source.strip():
                        mapping[q] = j
                        break
    return mapping


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _exec_safe(source: str, ns: dict) -> str | None:
    """Execute source in ns, capturing stdout. Returns error string or None."""
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(source, "<nb>", "exec"), ns)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _search_namespace(
    ns: dict, expected: np.ndarray
) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    """Search namespace for an array matching expected.

    Returns (matched_arr, best_abs_err, closest_arr):
      - matched_arr  : array that passes allclose, or None
      - best_abs_err : max absolute error of the closest shape-matching array
      - closest_arr  : the closest array (for fail details), or None
    """
    best_err = float("inf")
    closest: np.ndarray | None = None
    for key, val in ns.items():
        if key.startswith("_"):
            continue
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except Exception:
            continue
        if arr.shape != expected.shape:
            continue
        err = float(np.max(np.abs(arr - expected)))
        if err < best_err:
            best_err = err
            closest = arr
        if np.allclose(arr, expected, rtol=RTOL, atol=ATOL):
            return arr, err, arr
    return None, best_err, closest


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_tests(nb_path: Path, expected: dict[str, np.ndarray] | None = None) -> dict:
    expected = expected or EXPECTED
    nb = nbformat.read(str(nb_path), as_version=4)
    cells = nb.cells

    results = {q: {"passed": False, "details": ""} for q in QUESTIONS}

    problem_map = _find_problem_cells(nb)
    if not problem_map:
        for q in results:
            results[q]["details"] = "Could not identify problem cells (no 2.8/2.11/2.15 markers found)"
        return results

    # Only execute cells up to the last problem cell needed
    max_idx = max(problem_map.values())

    ns: dict = {"__builtins__": __builtins__}

    for i in range(max_idx + 1):
        cell = cells[i]
        if cell.cell_type != "code" or not cell.source.strip():
            continue

        err = _exec_safe(cell.source, ns)

        # Check if this cell is one of the problem cells
        for q, prob_idx in problem_map.items():
            if i != prob_idx:
                continue
            exp_arr = expected[q]
            if err:
                results[q] = {"passed": False, "details": f"execution_failed: {err}"}
            else:
                arr, abs_err, closest = _search_namespace(ns, exp_arr)
                if arr is not None:
                    results[q] = {
                        "passed": True,
                        "details": f"max_abs_err={abs_err:.4g}",
                    }
                else:
                    if closest is not None:
                        rel_err = abs_err / (np.max(np.abs(exp_arr)) + 1e-12)
                        results[q] = {
                            "passed": False,
                            "details": (
                                f"max_abs_err={abs_err:.4g}, "
                                f"max_rel_err={rel_err*100:.1f}%, "
                                f"got={closest.tolist()}, "
                                f"expected={exp_arr.tolist()}"
                            ),
                        }
                    else:
                        results[q] = {
                            "passed": False,
                            "details": f"no array of shape {exp_arr.shape} found in namespace",
                        }

    # Mark any question whose cell was not found
    for q in results:
        if q not in problem_map and not results[q]["details"]:
            results[q]["details"] = f"Cell for {q} not found in notebook"

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autograde a notebook against expected answers."
    )
    parser.add_argument("notebook", type=Path, help="Student .ipynb to grade")
    parser.add_argument(
        "--reference", "-r", type=Path, default=None,
        help="Reference .ipynb — execute it to DERIVE expected answers "
             "instead of using the hardcoded constants.",
    )
    parser.add_argument(
        "--config", "-c", type=Path, default=None,
        help="Per-assignment config.yaml (question markers, count, tolerance, "
             "answer_var, optional fallback expected).",
    )
    args = parser.parse_args()

    if args.config is not None:
        if not args.config.exists():
            print(f"config not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        apply_config(args.config)

    expected: dict[str, np.ndarray] | None = None
    if args.reference is not None:
        if not args.reference.exists():
            print(f"reference not found: {args.reference}", file=sys.stderr)
            sys.exit(1)
        try:
            expected = derive_expected(args.reference)
        except Exception as exc:  # fall back to hardcoded, but say so
            print(f"WARNING: could not derive from reference ({exc}); "
                  f"using hardcoded EXPECTED.", file=sys.stderr)
            expected = None

    print(json.dumps(run_tests(args.notebook, expected), indent=2))
