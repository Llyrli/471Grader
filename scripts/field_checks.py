"""Plot / field comparison — deterministic signals from a notebook's GRAPHICAL
output.

A lot of FEM homework asks for a *plot* (displacement along a bar, a stress
field, a convergence curve). The numbers behind those plots often never land in
a named variable we can checkpoint — they're passed straight into `plt.plot(...)`.
This module recovers the plotted data from the live matplotlib figure state after
execution and checks it, extending the deterministic layer from arrays to plots.

Two kinds of checks, both course-agnostic and config-driven (mirroring
physics_checks.py — same `{name, type, status: pass|fail|na, detail}` report
shape, so they slot into the same findings/gate machinery):

  reference-free shape invariants (no answer key needed):
    - plot_present    : at least `min_curves` curves were drawn (catches "forgot
                        to plot" / an empty axes).
    - plot_bounded    : every plotted y is finite and within a cap.
    - plot_monotonic  : some curve's y is monotonic in a given direction (e.g.
                        displacement increases along a bar in tension).
    - plot_endpoints  : a curve's first/last y matches a target (e.g. fixed end = 0).

  value comparison (against an expected curve, e.g. derived from the reference):
    - plot_matches    : some plotted curve's y (or x) matches `expected` within
                        tolerance, order-independent across curves.

A check is `na` (ignored, never penalized) when no curve is available to test or
the selector is out of range — we never penalize a plot we cannot observe.

Config (config.yaml, optional; top-level default + per-question override):

    field_checks:
      - {name: plotted_something,  type: plot_present,   min_curves: 1}
      - {name: disp_increasing,    type: plot_monotonic, direction: nondecreasing}
      - {name: disp_fixed_end,     type: plot_endpoints, first: 0.0}
      - {name: disp_curve_matches, type: plot_matches,   expected: [0.0, 0.5, 1.0]}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("field_checks")

DEFAULT_TOL = 1e-6
DEFAULT_MAX_ABS = 1e12
PASS, FAIL, NA = "pass", "fail", "na"


# ---------------------------------------------------------------------------
# Capture plotted curves from the live matplotlib state
# ---------------------------------------------------------------------------

def capture_current_curves(max_curves: int = 50) -> list[dict]:
    """Extract every drawn line as {x, y, label} from all current figures.

    Reads matplotlib's global figure registry (Agg backend), so it must be
    called in the SAME process that executed the plotting code (e.g. the sandbox
    worker). Never raises — returns [] if matplotlib is unavailable or has no
    figures.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    curves: list[dict] = []
    try:
        for num in plt.get_fignums():
            fig = plt.figure(num)
            for ax in fig.axes:
                for line in ax.get_lines():
                    xy = line.get_xydata()
                    if xy is None or len(xy) == 0:
                        continue
                    arr = np.asarray(xy, dtype=float)
                    if arr.ndim != 2 or arr.shape[1] < 2:
                        continue
                    curves.append({
                        "x": [float(v) for v in arr[:, 0]],
                        "y": [float(v) for v in arr[:, 1]],
                        "label": str(line.get_label()),
                    })
                    if len(curves) >= max_curves:
                        return curves
    except Exception as exc:  # defensive: capture must never break grading
        logger.debug("curve capture failed: %s", exc)
    return curves


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report(spec: dict, status: str, detail: str) -> dict:
    return {"name": spec.get("name", spec.get("type", "?")),
            "type": spec.get("type"), "status": status, "detail": detail}


def _axis(curve: dict, which: str) -> np.ndarray:
    return np.asarray(curve.get(which, []), dtype=float)


def _selected(curves: list[dict], spec: dict) -> list[dict]:
    """Curves a check applies to: a specific `index`, else all curves."""
    idx = spec.get("index")
    if idx is None:
        return curves
    if 0 <= idx < len(curves):
        return [curves[idx]]
    return []  # out of range → caller returns na


# ---------------------------------------------------------------------------
# Individual checks — each returns (status, detail)
# ---------------------------------------------------------------------------

def _check_present(curves: list[dict], spec: dict) -> tuple[str, str]:
    need = int(spec.get("min_curves", 1))
    return (PASS, f"{len(curves)} curve(s) ≥ {need}") if len(curves) >= need \
        else (FAIL, f"only {len(curves)} curve(s) plotted, need ≥ {need}")


def _check_bounded(curves: list[dict], spec: dict) -> tuple[str, str]:
    sel = _selected(curves, spec)
    if not sel:
        return NA, "no curve to test"
    cap = float(spec.get("max_abs", DEFAULT_MAX_ABS))
    worst = 0.0
    for c in sel:
        y = _axis(c, "y")
        if y.size and not np.all(np.isfinite(y)):
            return FAIL, "non-finite plotted values"
        worst = max(worst, float(np.max(np.abs(y))) if y.size else 0.0)
    return (PASS, f"max|y|={worst:.4g} ≤ {cap:.4g}") if worst <= cap \
        else (FAIL, f"max|y|={worst:.4g} > {cap:.4g}")


def _is_monotonic(y: np.ndarray, direction: str) -> bool:
    d = np.diff(y)
    if direction in ("increasing", "strictly_increasing"):
        return bool(np.all(d > 0))
    if direction in ("decreasing", "strictly_decreasing"):
        return bool(np.all(d < 0))
    if direction == "nonincreasing":
        return bool(np.all(d <= 1e-12))
    return bool(np.all(d >= -1e-12))  # nondecreasing (default)


def _check_monotonic(curves: list[dict], spec: dict) -> tuple[str, str]:
    sel = _selected(curves, spec)
    if not sel:
        return NA, "no curve to test"
    direction = spec.get("direction", "nondecreasing")
    for c in sel:
        y = _axis(c, "y")
        if y.size >= 2 and _is_monotonic(y, direction):
            return PASS, f"a curve is {direction}"
    return FAIL, f"no plotted curve is {direction}"


def _check_endpoints(curves: list[dict], spec: dict) -> tuple[str, str]:
    sel = _selected(curves, spec)
    if not sel:
        return NA, "no curve to test"
    tol = float(spec.get("tol", 1e-4))
    first, last = spec.get("first"), spec.get("last")
    if first is None and last is None:
        return NA, "no endpoint target given"
    for c in sel:
        y = _axis(c, "y")
        if y.size == 0:
            continue
        scale = float(np.max(np.abs(y))) + 1e-12
        ok = True
        if first is not None:
            ok = ok and abs(float(y[0]) - float(first)) <= tol * scale + DEFAULT_TOL
        if last is not None:
            ok = ok and abs(float(y[-1]) - float(last)) <= tol * scale + DEFAULT_TOL
        if ok:
            tgt = ", ".join(f"{k}={v}" for k, v in (("first", first), ("last", last)) if v is not None)
            return PASS, f"endpoints match ({tgt})"
    return FAIL, f"no curve with first={first}, last={last}"


def _check_matches(curves: list[dict], spec: dict) -> tuple[str, str]:
    expected = spec.get("expected")
    if expected is None:
        return NA, "no expected curve given"
    exp = np.asarray(expected, dtype=float).ravel()
    which = spec.get("axis", "y")
    rtol = float(spec.get("rtol", spec.get("tol", 0.02)))
    atol = float(spec.get("atol", 1e-8))
    best = float("inf")
    for c in _selected(curves, spec):
        v = _axis(c, which)
        if v.shape != exp.shape:
            continue
        err = float(np.max(np.abs(v - exp)))
        best = min(best, err)
        if np.allclose(v, exp, rtol=rtol, atol=atol):
            return PASS, f"a plotted {which}-curve matches expected (max_abs_err={err:.3g})"
    if best == float("inf"):
        return NA, f"no plotted curve of shape {exp.shape} on axis {which}"
    return FAIL, f"closest plotted {which}-curve off by max_abs_err={best:.3g}"


_CHECKS = {
    "plot_present": _check_present,
    "plot_bounded": _check_bounded,
    "plot_monotonic": _check_monotonic,
    "plot_endpoints": _check_endpoints,
    "plot_matches": _check_matches,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(curves: list[dict], spec: dict) -> dict:
    fn = _CHECKS.get(spec.get("type"))
    if fn is None:
        return _report(spec, NA, f"unknown field-check type {spec.get('type')!r}")
    try:
        status, detail = fn(curves, spec)
    except Exception as exc:
        return _report(spec, NA, f"check error: {type(exc).__name__}: {exc}")
    return _report(spec, status, detail)


def run_field_checks(curves: list[dict], specs: list[dict]) -> list[dict]:
    """Run all configured field checks against captured curves. Empty specs → []."""
    return [run_one(curves, s) for s in (specs or [])]
