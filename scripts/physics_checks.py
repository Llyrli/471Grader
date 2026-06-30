"""Reference-free physics-plausibility checks (deterministic signals).

Beyond matching a student's intermediate values to a reference (run_tests.py
checkpoints), many errors violate a PHYSICAL INVARIANT that must hold regardless
of the specific numbers — so they can be checked WITHOUT a reference answer:

  - finite     : a result array has no NaN/Inf.
  - bounded    : |values| ≤ a sane cap (catches blow-ups).
  - symmetric  : a stiffness/mass matrix equals its transpose (M ≈ Mᵀ).
  - psd        : a stiffness matrix is positive semi-definite (min eigenvalue ≥ −tol).
  - dirichlet  : fixed DOFs of a solution vector are ≈ 0 (boundary condition applied).
  - residual   : the solved linear system is actually satisfied: ‖A·x − b‖ small.
  - net_sum    : a reaction/force vector sums to a target (global equilibrium, usually 0).

These are course-agnostic and config-driven (each assignment declares which
invariants apply and which variable holds each quantity). They are SUPPLEMENTARY
deterministic signals: they never override the answer's pass/fail, but they
(a) corroborate an error's class, (b) provide a locus when value-matching can't,
and (c) flag "answer matches yet an invariant is violated" contradictions that
route a submission to human review.

A check resolves its variable by name (with optional `candidates` aliases). If
no candidate is materialized in the namespace, the check is "na" (not
applicable) — NOT a violation; we never penalize what we cannot observe.

Config (per-assignment config.yaml, optional; top-level default + per-question
override, mirroring `checkpoints:`):

    physics_checks:
      - {name: stiffness_symmetric, type: symmetric, var: Kg, candidates: [K, Kglobal]}
      - {name: stiffness_psd,       type: psd,       var: Kg}
      - {name: displacement_finite, type: finite,    var: u}
      - {name: bc_fixed_zero,       type: dirichlet, var: u, dofs: [0]}
      - {name: solve_residual,      type: residual,  matrix: K_free, x: u_free, b: F_free}
      - {name: reaction_balance,    type: net_sum,   var: R, target: 0.0}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("physics_checks")

DEFAULT_TOL = 1e-6        # relative tolerance for equality-style invariants
DEFAULT_MAX_ABS = 1e12    # boundedness cap

PASS, FAIL, NA = "pass", "fail", "na"


# ---------------------------------------------------------------------------
# Variable resolution (name + aliases, never by structure to avoid false hits)
# ---------------------------------------------------------------------------

def _as_array(val: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(val, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _resolve(ns: dict, spec: dict, key: str = "var") -> np.ndarray | None:
    """Find the array for a spec field, trying the primary name then aliases."""
    names: list[str] = []
    primary = spec.get(key)
    if primary:
        names.append(primary)
    names.extend(spec.get("candidates", []) if key == "var" else [])
    for name in names:
        if name in ns:
            arr = _as_array(ns[name])
            if arr is not None:
                return arr
    return None


def _report(spec: dict, status: str, detail: str) -> dict:
    return {"name": spec.get("name", spec.get("type", "?")),
            "type": spec.get("type"), "status": status, "detail": detail}


# ---------------------------------------------------------------------------
# Individual invariants — each returns (status, detail)
# ---------------------------------------------------------------------------

def _check_finite(arr: np.ndarray, spec: dict) -> tuple[str, str]:
    if np.all(np.isfinite(arr)):
        return PASS, "all finite"
    n_bad = int(np.sum(~np.isfinite(arr)))
    return FAIL, f"{n_bad} non-finite value(s) (NaN/Inf)"


def _check_bounded(arr: np.ndarray, spec: dict) -> tuple[str, str]:
    cap = float(spec.get("max_abs", DEFAULT_MAX_ABS))
    mx = float(np.max(np.abs(arr))) if arr.size else 0.0
    if not np.isfinite(mx):
        return FAIL, "non-finite magnitude"
    return (PASS, f"max|x|={mx:.4g} ≤ {cap:.4g}") if mx <= cap else (FAIL, f"max|x|={mx:.4g} > {cap:.4g}")


def _check_symmetric(arr: np.ndarray, spec: dict) -> tuple[str, str]:
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return NA, f"not a square matrix (shape {arr.shape})"
    tol = float(spec.get("tol", DEFAULT_TOL))
    asym = float(np.max(np.abs(arr - arr.T)))
    scale = float(np.max(np.abs(arr))) + 1e-12
    return (PASS, f"asymmetry={asym:.3g}") if asym <= tol * scale else (FAIL, f"‖M−Mᵀ‖={asym:.3g} (rel {asym/scale:.2g})")


def _check_psd(arr: np.ndarray, spec: dict) -> tuple[str, str]:
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return NA, f"not a square matrix (shape {arr.shape})"
    tol = float(spec.get("tol", DEFAULT_TOL))
    sym = 0.5 * (arr + arr.T)  # symmetric part — eigvalsh needs symmetry
    try:
        w = np.linalg.eigvalsh(sym)
    except np.linalg.LinAlgError as exc:
        return NA, f"eig failed: {exc}"
    scale = float(np.max(np.abs(w))) + 1e-12
    min_eig = float(np.min(w))
    return (PASS, f"min eig={min_eig:.3g}") if min_eig >= -tol * scale else (FAIL, f"min eig={min_eig:.3g} < 0 (not PSD)")


def _check_dirichlet(arr: np.ndarray, spec: dict) -> tuple[str, str]:
    dofs = spec.get("dofs")
    if not dofs:
        return NA, "no dofs specified"
    flat = arr.ravel()
    tol = float(spec.get("tol", DEFAULT_TOL))
    scale = float(np.max(np.abs(flat))) + 1e-12
    bad = []
    for d in dofs:
        if d >= flat.size:
            return NA, f"dof {d} out of range (size {flat.size})"
        if abs(float(flat[d])) > tol * scale + DEFAULT_TOL:
            bad.append((d, float(flat[d])))
    if bad:
        return FAIL, "non-zero fixed DOF(s): " + ", ".join(f"u[{d}]={v:.3g}" for d, v in bad)
    return PASS, f"fixed DOFs {list(dofs)} ≈ 0"


def _check_residual(ns: dict, spec: dict) -> tuple[str, str]:
    A = _resolve(ns, spec, "matrix")
    x = _resolve(ns, spec, "x")
    b = _resolve(ns, spec, "b")
    if A is None or x is None or b is None:
        missing = [k for k, v in [("matrix", A), ("x", x), ("b", b)] if v is None]
        return NA, f"missing {', '.join(missing)}"
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[1] != x.size or A.shape[0] != b.size:
        return NA, f"shape mismatch A{A.shape} x{x.shape} b{b.shape}"
    tol = float(spec.get("tol", 1e-4))
    resid = float(np.max(np.abs(A @ x.ravel() - b.ravel())))
    scale = float(np.max(np.abs(b))) + 1e-12
    return (PASS, f"‖Ax−b‖∞={resid:.3g}") if resid <= tol * scale else (FAIL, f"‖Ax−b‖∞={resid:.3g} (rel {resid/scale:.2g}) — system not satisfied")


def _check_net_sum(arr: np.ndarray, spec: dict) -> tuple[str, str]:
    target = float(spec.get("target", 0.0))
    tol = float(spec.get("tol", 1e-4))
    s = float(np.sum(arr))
    scale = float(np.max(np.abs(arr))) + 1e-12
    return (PASS, f"Σ={s:.3g}≈{target:g}") if abs(s - target) <= tol * scale + DEFAULT_TOL else (FAIL, f"Σ={s:.3g} ≠ {target:g} (imbalance {s-target:.3g})")


# Checks that take a single resolved array vs. those that need the namespace.
_ARRAY_CHECKS = {
    "finite": _check_finite,
    "bounded": _check_bounded,
    "symmetric": _check_symmetric,
    "psd": _check_psd,
    "dirichlet": _check_dirichlet,
    "net_sum": _check_net_sum,
}
_NS_CHECKS = {"residual": _check_residual}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(ns: dict, spec: dict) -> dict:
    ctype = spec.get("type")
    if ctype in _NS_CHECKS:
        status, detail = _NS_CHECKS[ctype](ns, spec)
        return _report(spec, status, detail)
    fn = _ARRAY_CHECKS.get(ctype)
    if fn is None:
        return _report(spec, NA, f"unknown check type {ctype!r}")
    arr = _resolve(ns, spec)
    if arr is None:
        return _report(spec, NA, f"variable {spec.get('var')!r} not materialized")
    try:
        status, detail = fn(arr, spec)
    except Exception as exc:  # never let a check crash grading
        return _report(spec, NA, f"check error: {type(exc).__name__}: {exc}")
    return _report(spec, status, detail)


def run_physics_checks(ns: dict, specs: list[dict]) -> list[dict]:
    """Run all configured physics checks against a namespace. Returns one report
    per spec: {name, type, status (pass|fail|na), detail}. Empty specs → []."""
    return [run_one(ns, s) for s in (specs or [])]


def violations(reports: list[dict]) -> list[str]:
    """Names of checks that deterministically FAILED (na is not a violation)."""
    return [r["name"] for r in reports if r.get("status") == FAIL]
