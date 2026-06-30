"""LLM fix → re-execute → verify by execution.

The LLM already proposes a textual `fix` for a located error. This module turns
that into a DETERMINISTIC check instead of a claim: ask the LLM for a corrected,
self-contained version of the failing code, RUN it in the sandbox, and let
execution decide whether the fix worked:

  - `fix_runs`     : the corrected code executes without error ("跑通").
  - `fix_verified` : it runs AND, when a target answer is known, now reproduces
                     the reference answer within tolerance. Without a target
                     (open-ended problems) `fix_verified` falls back to `fix_runs`.

This keeps the determinism-first contract: the LLM proposes, execution verifies —
a fix that "runs and matches" is confirmed by the kernel, not by the model's say-so.

Used by score_notebooks.py (`--verify-fixes`) for failed, located questions; the
core `run_and_check()` is engine-agnostic and unit-testable without any LLM.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("fix_verify")

DEFAULT_RTOL = 0.02
DEFAULT_ATOL = 1e-8


# ---------------------------------------------------------------------------
# Deterministic core: execute code (sandboxed) and check the answer
# ---------------------------------------------------------------------------

def _run_target(code: str, expected: list | None, rtol: float, atol: float,
                answer_var: str | None) -> dict:
    """Runs INSIDE the sandbox worker. Exec `code`, then check the answer.

    Returns {runs, error, matched} where matched is True/False/None (None = no
    expected target to compare against)."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np

    ns: dict = {"__builtins__": __builtins__}
    try:
        exec(compile(code, "<fix>", "exec"), ns)
    except Exception as exc:  # noqa: BLE001 — any student-code error is a "didn't run"
        return {"runs": False, "error": f"{type(exc).__name__}: {exc}", "matched": False}

    if expected is None:
        return {"runs": True, "error": None, "matched": None}

    exp = np.asarray(expected, dtype=float).ravel()

    def _matches(val) -> bool:
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except Exception:
            return False
        return arr.shape == exp.shape and bool(np.allclose(arr, exp, rtol=rtol, atol=atol))

    # Prefer a named answer var; otherwise scan the namespace for a matching array.
    if answer_var and answer_var in ns and _matches(ns[answer_var]):
        return {"runs": True, "error": None, "matched": True}
    for key, val in ns.items():
        if not key.startswith("_") and _matches(val):
            return {"runs": True, "error": None, "matched": True}
    return {"runs": True, "error": None, "matched": False}


def run_and_check(
    code: str,
    expected: list | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    answer_var: str | None = None,
    limits: dict | None = None,
) -> dict:
    """Execute `code` under the sandbox and report {runs, error, matched}.

    `expected` is a flat list (the reference answer); when given, `matched` says
    whether the executed code reproduces it within tolerance. Sandbox failures
    (timeout/OOM) are reported as runs=False rather than raised."""
    try:
        from sandbox import run_in_sandbox, SandboxError
    except Exception:  # no sandbox available → run directly
        return _run_target(code, expected, rtol, atol, answer_var)
    try:
        return run_in_sandbox(_run_target, (code, expected, rtol, atol, answer_var),
                              **(limits or {}))
    except SandboxError as exc:
        return {"runs": False, "error": f"sandbox: {exc}", "matched": False}


# ---------------------------------------------------------------------------
# LLM fix proposal
# ---------------------------------------------------------------------------

FIX_SYSTEM = """\
You repair a single Finite-Element homework question. You are given the student's
code and the DETERMINISTIC locus where it first diverged from the reference.
Make the SMALLEST change that fixes the located error, preserving the student's
overall approach and variable names. Output ONLY a complete, self-contained
Python program that computes the question's answer (it must run on its own:
include necessary imports and any setup). No markdown fences, no prose."""


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("python"):
            raw = raw[len("python"):]
        elif raw.startswith("py"):
            raw = raw[len("py"):]
    return raw.strip()


def propose_fix(client: Any, student_code: str, finding: str,
                reference_hint: str = "", prev_code: str | None = None,
                prev_problem: str | None = None, max_tokens: int = 1600) -> str:
    """Ask the LLM for corrected, self-contained code. Returns the code string.

    On a retry, the previous (failed) attempt and WHY it failed are fed back so
    the model produces a genuinely different correction, not the same one."""
    user = (
        f"[LOCATED ERROR]\n{finding}\n\n"
        f"[STUDENT CODE]\n{student_code}\n\n"
        + (f"[REFERENCE APPROACH — context only]\n{reference_hint}\n\n" if reference_hint else "")
    )
    if prev_code:
        user += (
            f"[YOUR PREVIOUS ATTEMPT — REJECTED BY EXECUTION: {prev_problem}]\n{prev_code}\n\n"
            "That attempt did not work. Produce a DIFFERENT corrected program that "
            "addresses the failure above.\n\n"
        )
    user += "Return the corrected, self-contained Python program."
    return _strip_code_fences(client.complete(FIX_SYSTEM, user, max_tokens=max_tokens))


DEFAULT_MAX_ITERATIONS = 3


def attempt_fix(
    client: Any,
    student_code: str,
    finding: str,
    expected: list | None,
    reference_hint: str = "",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    answer_var: str | None = None,
    limits: dict | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> dict:
    """Self-repair loop: propose → execute → if not verified, regenerate with the
    failure fed back — up to `max_iterations`. Execution is the judge each round.

    Returns {fix_attempted, fix_runs, fix_verified, iterations, exhausted,
    fix_error, corrected_code, attempts[]}. When the loop exhausts without a
    verified fix, `exhausted=True` (the caller routes these to human review)."""
    attempts: list[dict] = []
    prev_code: str | None = None
    prev_problem: str | None = None
    last_code = ""
    last_runs = False

    for i in range(1, max(1, max_iterations) + 1):
        try:
            code = propose_fix(client, student_code, finding, reference_hint,
                               prev_code=prev_code, prev_problem=prev_problem)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fix proposal failed (iter %d): %s", i, exc)
            attempts.append({"iteration": i, "runs": False, "verified": False,
                             "error": f"proposal_failed: {exc}"})
            break

        res = run_and_check(code, expected, rtol, atol, answer_var, limits)
        matched = res.get("matched")
        verified = bool(res["runs"]) if matched is None else bool(matched)
        last_code, last_runs = code, bool(res["runs"])
        attempts.append({"iteration": i, "runs": bool(res["runs"]),
                         "verified": verified, "error": res.get("error")})

        if verified:
            return {"fix_attempted": True, "fix_runs": True, "fix_verified": True,
                    "iterations": i, "exhausted": False, "fix_error": None,
                    "corrected_code": code, "attempts": attempts}

        # Feed the concrete failure back into the next round.
        prev_code = code
        prev_problem = (res.get("error") or
                        "it ran but the computed answer still did not match the reference")

    return {"fix_attempted": True, "fix_runs": last_runs, "fix_verified": False,
            "iterations": len(attempts), "exhausted": True,
            "fix_error": prev_problem, "corrected_code": last_code, "attempts": attempts}
