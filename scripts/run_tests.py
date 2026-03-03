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
# Regex patterns to identify problem cells
# ---------------------------------------------------------------------------

PROBLEM_PATTERNS = {
    "Q1": re.compile(r"2[._]8\b|problem.{0,5}2\.8", re.IGNORECASE),
    "Q2": re.compile(r"2[._]11\b|problem.{0,5}2\.11", re.IGNORECASE),
    "Q3": re.compile(r"2[._]15\b|problem.{0,5}2\.15", re.IGNORECASE),
}


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


def _search_namespace(ns: dict, expected: np.ndarray) -> np.ndarray | None:
    """Return the first numpy array in ns that matches expected, or None."""
    for key, val in ns.items():
        if key.startswith("_"):
            continue
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except Exception:
            continue
        if arr.shape == expected.shape and np.allclose(arr, expected, rtol=RTOL, atol=ATOL):
            return arr
    return None


def _arrays_of_shape(ns: dict, shape: tuple) -> list[str]:
    """Return names of arrays in ns with the given shape (for debug output)."""
    found = []
    for key, val in ns.items():
        if key.startswith("_"):
            continue
        try:
            arr = np.asarray(val, dtype=float).ravel()
            if arr.shape == shape:
                found.append(f"{key}={arr.tolist()}")
        except Exception:
            pass
    return found[:5]  # limit for readability


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_tests(nb_path: Path) -> dict:
    nb = nbformat.read(str(nb_path), as_version=4)
    cells = nb.cells

    results = {
        "Q1": {"passed": False, "details": ""},
        "Q2": {"passed": False, "details": ""},
        "Q3": {"passed": False, "details": ""},
    }

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
            expected = EXPECTED[q]
            if err:
                results[q] = {"passed": False, "details": f"execution_failed: {err}"}
            else:
                arr = _search_namespace(ns, expected)
                if arr is not None:
                    results[q] = {"passed": True, "details": f"u={arr.tolist()}"}
                else:
                    hint = _arrays_of_shape(ns, expected.shape)
                    results[q] = {
                        "passed": False,
                        "details": (
                            f"expected≈{expected.tolist()}; "
                            f"arrays of shape {expected.shape} found: "
                            f"{hint if hint else 'none'}"
                        ),
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
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <notebook.ipynb>", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(run_tests(Path(sys.argv[1])), indent=2))
