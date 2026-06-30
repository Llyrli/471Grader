"""Autograde + deterministic deviation localization for notebook submissions.

Two deterministic signals, both execution-based and style-agnostic (they compare
*values*, never code structure — so any implementation that computes the right
quantity passes):

  1. RESULT (pass/fail): does the student's final answer match the reference
     within tolerance?  (unchanged from the original autograder)

  2. DEVIATION LOCALIZATION (new): the reference defines an *ordered* list of
     intermediate-value CHECKPOINTS (e.g. global stiffness `Kg` → reduced system
     `K_free` → displacement `u`).  We execute the student notebook and, at each
     problem cell, search the namespace for an array matching each checkpoint by
     shape+value.  The FIRST checkpoint with no match is the deterministic locus
     of the logic error — "where the student first diverged from the reference".
     This is the signal the LLM is later restricted to explaining; it is NOT an
     AST/structure comparison (those are fragile for open-ended code).

If a checkpoint's intermediate is never materialized in the student namespace
(e.g. computed inline and discarded), that checkpoint is reported `located:
false` rather than failed — which downstream triggers low-confidence abstention.

Strategy:
  1. Scan cells for per-question markers to identify which code cell ends Q1/Q2/Q3.
  2. Execute cells in order; at each problem cell, match every checkpoint.
  3. Never depends on variable names — matches by array shape + value.

Usage:
    python run_tests.py <notebook.ipynb> [--reference <ref.ipynb>] [--config <config.yaml>]
Output: JSON with per-question pass/fail + checkpoint localization to stdout.
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

from physics_checks import run_physics_checks
from field_checks import capture_current_curves, run_field_checks

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
# Reference oracle: derive expected answers AND intermediate-value checkpoints
# by EXECUTING a reference solution, instead of relying on hardcoded constants.
#
# Answer-designation convention (either works):
#   1) The reference defines a top-level dict ``ANSWERS = {"Q1": ..., ...}``.
#   2) Otherwise: the answer for each problem is the nodal-displacement vector
#      ``u``, and problems are delimited by ``2_8`` / ``2_11`` / ``2_15``
#      markdown section headers (underscore form — avoids colliding with a
#      title cell that lists "2.8, 2.11, 2.15").
#
# Checkpoint convention (for deviation localization, optional):
#   config.yaml may declare an ordered ``checkpoints`` list per assignment, e.g.
#       checkpoints:
#         - {name: global_stiffness, ref_var: Kg}
#         - {name: bc_reduced,       ref_var: K_free}
#         - {name: displacement,     ref_var: u, is_answer: true}
#   Each checkpoint's reference value is snapshotted from the named variable at
#   the end of the corresponding problem section. Order encodes the FE pipeline
#   (assembly → BC → solve), so the first unmatched checkpoint localizes the
#   error. If no checkpoints are configured we fall back to a single answer-only
#   checkpoint, i.e. the original behavior.
# ---------------------------------------------------------------------------

# Underscore-form section markers for the REFERENCE (disambiguates from the
# dotted "2.8" that may appear in a title/intro cell).
REFERENCE_SECTION_PATTERNS = {
    "Q1": re.compile(r"2_8\b"),
    "Q2": re.compile(r"2_11\b"),
    "Q3": re.compile(r"2_15\b"),
}
ANSWER_VAR = "u"  # convention: nodal displacements live in `u`

# Per-question ordered checkpoint specs: {Q: [{name, ref_var, is_answer}, ...]}.
# Empty by default → answer-only behavior (back-compatible).
CHECKPOINTS: dict[str, list[dict]] = {}

# Per-question reference-free physics-invariant specs (see physics_checks.py).
# Empty by default → no physics checks (back-compatible).
PHYSICS_CHECKS: dict[str, list[dict]] = {}

# Per-question plot/field-comparison specs (see field_checks.py).
# Empty by default → no field checks (back-compatible).
FIELD_CHECKS: dict[str, list[dict]] = {}


def _checkpoints_for(q: str) -> list[dict]:
    """Ordered checkpoint specs for question ``q``.

    Falls back to a single answer checkpoint on ``ANSWER_VAR`` so that, even
    without an explicit ``checkpoints`` config, the localization machinery has a
    well-defined (degenerate) answer checkpoint to match.
    """
    specs = CHECKPOINTS.get(q)
    if specs:
        return specs
    return [{"name": "answer", "ref_var": ANSWER_VAR, "is_answer": True}]


def _needed_ref_vars() -> set[str]:
    """Every reference variable referenced by any checkpoint (plus ANSWER_VAR)."""
    names = {ANSWER_VAR}
    for specs in CHECKPOINTS.values():
        for c in specs:
            if c.get("ref_var"):
                names.add(c["ref_var"])
    return names


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


def derive_reference(ref_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Execute a reference notebook and snapshot all checkpoint variables.

    Returns ``{Q: {var_name: array}}`` — for each problem section, the value of
    every variable named by a checkpoint (and ANSWER_VAR), captured at the
    section boundary. An explicit top-level ``ANSWERS`` dict, if present,
    overrides the answer variable's snapshot.

    Raises ValueError if the reference fails to run or yields nothing usable.
    """
    nb = nbformat.read(str(ref_path), as_version=4)
    cells = nb.cells
    bounds = _reference_section_starts(cells)
    order = [q for q, _ in bounds]
    starts = {idx for _, idx in bounds}
    start_to_q = {idx: q for q, idx in bounds}
    needed = _needed_ref_vars()

    ns: dict = {"__builtins__": __builtins__}
    snapshots: dict[str, dict[str, np.ndarray]] = {}
    prev_q: str | None = None

    def _snapshot(q: str) -> None:
        bag = snapshots.setdefault(q, {})
        for var in needed:
            if var in bag or var not in ns:
                continue
            try:
                bag[var] = np.asarray(ns[var], dtype=float)
            except Exception:
                continue

    for i, cell in enumerate(cells):
        # Crossing into a new section snapshots the previous section's vars.
        if i in starts:
            if prev_q is not None:
                _snapshot(prev_q)
            prev_q = start_to_q[i]
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        err = _exec_safe(cell.source, ns)
        if err:
            raise ValueError(f"reference cell {i} failed: {err}")

    # Last section.
    if order:
        _snapshot(order[-1])

    # Explicit ANSWERS dict overrides the answer var per question.
    explicit = ns.get("ANSWERS")
    if isinstance(explicit, dict):
        for q, val in explicit.items():
            try:
                snapshots.setdefault(q, {})[ANSWER_VAR] = np.asarray(val, dtype=float)
            except Exception:
                continue

    if not any(snapshots.values()):
        raise ValueError(
            "could not derive any reference values "
            "(define ANSWERS={...} or use section markers with the checkpoint vars)"
        )
    return snapshots


def derive_expected(ref_path: Path) -> dict[str, np.ndarray]:
    """Backward-compatible: return {Q: answer_array} (raveled) from the reference."""
    snapshots = derive_reference(ref_path)
    expected: dict[str, np.ndarray] = {}
    for q, bag in snapshots.items():
        if ANSWER_VAR in bag:
            expected[q] = np.asarray(bag[ANSWER_VAR], dtype=float).ravel()
    if not expected:
        raise ValueError("could not derive any expected answers from reference")
    return expected


def build_checkpoints(ref_path: Path | None) -> dict[str, list[dict]]:
    """Build per-question ordered checkpoint targets with reference values.

    Returns ``{Q: [{name, expected (ndarray, raveled), is_answer}, ...]}``.

    With a reference, each checkpoint's expected value is the snapshot of its
    ``ref_var``; checkpoints whose variable was never materialized in the
    reference are dropped (can't localize what the reference itself didn't show).
    Without a reference, falls back to one answer checkpoint per question using
    the hardcoded/`config` EXPECTED.
    """
    snapshots: dict[str, dict[str, np.ndarray]] = {}
    if ref_path is not None:
        snapshots = derive_reference(ref_path)

    out: dict[str, list[dict]] = {}
    for q in QUESTIONS:
        targets: list[dict] = []
        for spec in _checkpoints_for(q):
            var = spec.get("ref_var", ANSWER_VAR)
            is_answer = bool(spec.get("is_answer")) or spec.get("name") == "answer"
            exp: np.ndarray | None = None
            if snapshots:
                bag = snapshots.get(q, {})
                if var in bag:
                    exp = np.asarray(bag[var], dtype=float).ravel()
            if exp is None and is_answer:
                # Answer checkpoint can fall back to EXPECTED even w/o reference.
                if q in EXPECTED:
                    exp = np.asarray(EXPECTED[q], dtype=float).ravel()
            if exp is None:
                continue  # reference never produced this intermediate; skip it
            targets.append({
                "name": spec.get("name", var),
                "expected": exp,
                "is_answer": is_answer,
            })
        # Guarantee at least one answer checkpoint if EXPECTED has the question.
        if not any(t["is_answer"] for t in targets) and targets:
            targets[-1]["is_answer"] = True
        out[q] = targets
    return out

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
        checkpoints:                 # optional, applied to every question
          - {name: global_stiffness, ref_var: Kg}
          - {name: bc_reduced,       ref_var: K_free}
          - {name: displacement,     ref_var: u, is_answer: true}
        expected:                    # optional hardcoded fallback (no --reference)
          Q1: [0.0, 0.5, 1.0]

    A question may also carry its own ``checkpoints:`` to override the top-level
    default for that question only.
    """
    global QUESTIONS, PROBLEM_PATTERNS, REFERENCE_SECTION_PATTERNS
    global RTOL, ATOL, ANSWER_VAR, EXPECTED, CHECKPOINTS, PHYSICS_CHECKS, FIELD_CHECKS

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

    # Checkpoints: top-level default + optional per-question override.
    default_cps = cfg.get("checkpoints") or []
    CHECKPOINTS = {}
    for q in questions:
        cps = q.get("checkpoints", default_cps)
        if cps:
            CHECKPOINTS[q["name"]] = list(cps)

    # Physics checks: same top-level-default + per-question-override pattern.
    default_phys = cfg.get("physics_checks") or []
    PHYSICS_CHECKS = {}
    for q in questions:
        phys = q.get("physics_checks", default_phys)
        if phys:
            PHYSICS_CHECKS[q["name"]] = list(phys)

    # Field/plot checks: same pattern.
    default_fields = cfg.get("field_checks") or []
    FIELD_CHECKS = {}
    for q in questions:
        fch = q.get("field_checks", default_fields)
        if fch:
            FIELD_CHECKS[q["name"]] = list(fch)


def _find_problem_cells(nb) -> dict[str, int]:
    """Return {Q_name: cell_index} by looking for per-question markers.

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
    """Search namespace for an array matching expected (by raveled shape+value).

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


def _match_checkpoint(ns: dict, expected: np.ndarray) -> dict:
    """Match a single checkpoint against the namespace.

    Returns a report dict:
      - located: bool   — an array of the right shape exists at all
      - matched: bool   — that array is within tolerance
      - max_abs_err / max_rel_err — diagnostics for the closest array
      - got / expected  — values, only when located but not matched
    """
    arr, abs_err, closest = _search_namespace(ns, expected)
    if arr is not None:
        return {"located": True, "matched": True, "max_abs_err": round(abs_err, 6)}
    if closest is not None:
        rel_err = abs_err / (np.max(np.abs(expected)) + 1e-12)
        return {
            "located": True,
            "matched": False,
            "max_abs_err": round(abs_err, 6),
            "max_rel_err": round(rel_err, 4),
            "got": closest.tolist(),
            "expected": expected.tolist(),
        }
    return {
        "located": False,
        "matched": False,
        "details": f"no array of shape {expected.shape} found in namespace",
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _grade_cells(nb_path: Path, checkpoints: dict[str, list[dict]]) -> dict:
    """Execute the student notebook and match checkpoints (UNTRUSTED code path).

    Runs student code via in-process exec(); intended to be invoked inside the
    sandbox worker (see run_tests). Returns the per-question results dict.
    """
    nb = nbformat.read(str(nb_path), as_version=4)
    cells = nb.cells

    results = {
        q: {"passed": False, "details": "", "checkpoints": [],
            "first_divergence": None, "physics": [], "fields": []}
        for q in QUESTIONS
    }

    problem_map = _find_problem_cells(nb)
    if not problem_map:
        for q in results:
            results[q]["details"] = "Could not identify problem cells (no markers found)"
        return results

    max_idx = max(problem_map.values())
    ns: dict = {"__builtins__": __builtins__}

    for i in range(max_idx + 1):
        cell = cells[i]
        if cell.cell_type != "code" or not cell.source.strip():
            continue

        err = _exec_safe(cell.source, ns)

        for q, prob_idx in problem_map.items():
            if i != prob_idx:
                continue
            targets = checkpoints.get(q, [])
            if err:
                results[q] = {
                    "passed": False,
                    "details": f"execution_failed: {err}",
                    "checkpoints": [],
                    "first_divergence": None,
                    "physics": [],
                    "fields": [],
                }
                continue
            results[q] = _evaluate_checkpoints(ns, targets)
            # Reference-free physics invariants, evaluated on the same namespace.
            results[q]["physics"] = run_physics_checks(ns, PHYSICS_CHECKS.get(q, []))
            # Plot/field comparison: capture drawn curves from matplotlib state.
            fld_specs = FIELD_CHECKS.get(q, [])
            results[q]["fields"] = (
                run_field_checks(capture_current_curves(), fld_specs) if fld_specs else []
            )

    for q in results:
        if q not in problem_map and not results[q]["details"]:
            results[q]["details"] = f"Cell for {q} not found in notebook"
            results[q]["checkpoints"] = []

    return results


def run_tests(
    nb_path: Path,
    checkpoints: dict[str, list[dict]] | None = None,
    sandbox_limits: dict | None = None,
) -> dict:
    """Autograde + localize deviations for one notebook (sandboxed).

    Student code runs inside a resource-bounded, wall-clock-limited sandbox
    (see sandbox.py). On any sandbox failure (timeout, OOM, CPU/file-size cap,
    crash) every question is marked failed with the reason — a runaway or
    malicious submission yields a deterministic failure, never a hung grader.

    ``checkpoints`` is {Q: [{name, expected, is_answer}, ...]} from
    build_checkpoints(); when None it is derived from EXPECTED (answer-only).
    ``sandbox_limits`` overrides timeout/mem/cpu/fsize (see sandbox.limits_from_config).

    Per-question result:
      {passed, details, checkpoints[], first_divergence}
    """
    if checkpoints is None:
        checkpoints = build_checkpoints(None)

    try:
        from sandbox import run_in_sandbox, SandboxError
    except Exception:  # sandbox module unavailable → run directly
        return _grade_cells(nb_path, checkpoints)

    try:
        return run_in_sandbox(
            _grade_cells, (nb_path, checkpoints), **(sandbox_limits or {})
        )
    except SandboxError as exc:
        return {
            q: {
                "passed": False,
                "details": f"sandboxed_execution_failed: {exc}",
                "checkpoints": [],
                "first_divergence": None,
            }
            for q in QUESTIONS
        }


def _evaluate_checkpoints(ns: dict, targets: list[dict]) -> dict:
    """Match every checkpoint in order; localize the first divergence.

    Localization is meaningful ONLY when the answer is wrong: if the final answer
    matches, there is no logic error to localize, and intermediate "mismatches"
    are either spurious shape collisions or legitimately-different formulations
    (e.g. a penalty method that never materializes a reduced system). In that
    case ``first_divergence`` is null and intermediate checkpoints are reported
    as informational only.
    """
    reports: list[dict] = []
    answer_passed = False
    answer_seen = False

    for t in targets:
        rep = _match_checkpoint(ns, t["expected"])
        reports.append({"name": t["name"], "is_answer": bool(t.get("is_answer")), **rep})
        if t.get("is_answer"):
            answer_seen = True
            answer_passed = rep["matched"]

    passed = answer_passed if answer_seen else all(r["matched"] for r in reports)

    first_divergence: str | None = None
    if not passed:
        # Wrong answer → localize. Prefer the first *located-but-wrong*
        # checkpoint in pipeline order (assembly → BC → solve); an unlocated
        # checkpoint can't confirm divergence, so it's only a fallback locus.
        for r in reports:
            if r.get("located", True) and not r["matched"]:
                first_divergence = r["name"]
                break
        if first_divergence is None:
            for r in reports:
                if not r["matched"]:
                    first_divergence = r["name"]
                    break

    details = _summarize(reports, passed, first_divergence)
    return {
        "passed": passed,
        "details": details,
        "checkpoints": reports,
        "first_divergence": first_divergence,
    }


def _summarize(reports: list[dict], passed: bool, first_divergence: str | None) -> str:
    """One-line human summary preserving the old details style for the answer."""
    if passed:
        ans = next((r for r in reports if r.get("matched")), None)
        if ans and "max_abs_err" in ans:
            return f"max_abs_err={ans['max_abs_err']:.4g}"
        return "passed"
    # Failed: describe where it first diverged.
    div = next((r for r in reports if r["name"] == first_divergence), None)
    if div is None:
        return "no checkpoints evaluable"
    if not div.get("located", True):
        return f"diverged at '{first_divergence}': not materialized in namespace"
    return (
        f"diverged at '{first_divergence}': "
        f"max_abs_err={div.get('max_abs_err')}, "
        f"max_rel_err={(div.get('max_rel_err') or 0) * 100:.1f}%, "
        f"got={div.get('got')}, expected={div.get('expected')}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autograde a notebook + localize deviations against a reference."
    )
    parser.add_argument("notebook", type=Path, help="Student .ipynb to grade")
    parser.add_argument(
        "--reference", "-r", type=Path, default=None,
        help="Reference .ipynb — execute it to DERIVE expected answers and "
             "intermediate checkpoints instead of using hardcoded constants.",
    )
    parser.add_argument(
        "--config", "-c", type=Path, default=None,
        help="Per-assignment config.yaml (markers, tolerance, answer_var, "
             "checkpoints, optional fallback expected).",
    )
    args = parser.parse_args()

    sandbox_limits: dict | None = None
    if args.config is not None:
        if not args.config.exists():
            print(f"config not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        apply_config(args.config)
        try:
            from sandbox import limits_from_config
            sandbox_limits = limits_from_config(
                yaml.safe_load(args.config.read_text(encoding="utf-8"))
            )
        except Exception:
            sandbox_limits = None

    ref_path: Path | None = None
    if args.reference is not None:
        if not args.reference.exists():
            print(f"reference not found: {args.reference}", file=sys.stderr)
            sys.exit(1)
        ref_path = args.reference

    try:
        checkpoints = build_checkpoints(ref_path)
    except Exception as exc:  # fall back to answer-only from EXPECTED
        print(f"WARNING: could not derive from reference ({exc}); "
              f"using hardcoded EXPECTED (answer-only).", file=sys.stderr)
        checkpoints = build_checkpoints(None)

    print(json.dumps(run_tests(args.notebook, checkpoints, sandbox_limits), indent=2))
