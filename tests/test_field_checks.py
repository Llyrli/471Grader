"""Unit tests for plot/field comparison checks (pure-curve logic, no matplotlib)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import field_checks as fc  # noqa: E402


def _curve(y, x=None):
    x = x if x is not None else list(range(len(y)))
    return {"x": list(x), "y": list(y), "label": "_line"}


def _status(curves, spec):
    return fc.run_one(curves, spec)["status"]


def test_plot_present():
    spec = {"type": "plot_present", "min_curves": 1}
    assert _status([_curve([0, 1, 2])], spec) == fc.PASS
    assert _status([], spec) == fc.FAIL
    assert _status([_curve([0]), _curve([1])], {"type": "plot_present", "min_curves": 3}) == fc.FAIL


def test_plot_monotonic_nondecreasing():
    spec = {"type": "plot_monotonic", "direction": "nondecreasing"}
    assert _status([_curve([0.0, 0.5, 1.0])], spec) == fc.PASS
    assert _status([_curve([0.0, 1.0, 0.5])], spec) == fc.FAIL
    # passes if ANY curve qualifies
    assert _status([_curve([2, 1, 0]), _curve([0, 1, 2])], spec) == fc.PASS
    # no curves → na
    assert _status([], spec) == fc.NA


def test_plot_monotonic_strict_increasing():
    spec = {"type": "plot_monotonic", "direction": "increasing"}
    assert _status([_curve([0.0, 1.0, 2.0])], spec) == fc.PASS
    assert _status([_curve([0.0, 0.0, 1.0])], spec) == fc.FAIL  # flat segment not strict


def test_plot_endpoints():
    spec = {"type": "plot_endpoints", "first": 0.0, "last": 1.0}
    assert _status([_curve([0.0, 0.5, 1.0])], spec) == fc.PASS
    assert _status([_curve([0.3, 0.5, 1.0])], spec) == fc.FAIL
    # no target → na
    assert _status([_curve([0.0, 1.0])], {"type": "plot_endpoints"}) == fc.NA


def test_plot_bounded():
    spec = {"type": "plot_bounded", "max_abs": 10.0}
    assert _status([_curve([1.0, -9.0])], spec) == fc.PASS
    assert _status([_curve([1.0, 1e6])], spec) == fc.FAIL
    assert _status([_curve([float("nan")])], spec) == fc.FAIL


def test_plot_matches_order_independent():
    spec = {"type": "plot_matches", "expected": [0.0, 0.5, 1.0], "rtol": 0.02}
    # second curve matches → pass
    assert _status([_curve([9, 9, 9]), _curve([0.0, 0.5, 1.0])], spec) == fc.PASS
    # close but outside tol → fail
    assert _status([_curve([0.0, 0.9, 1.0])], spec) == fc.FAIL
    # no curve of matching shape → na
    assert _status([_curve([0.0, 1.0])], spec) == fc.NA


def test_plot_matches_on_x_axis():
    spec = {"type": "plot_matches", "expected": [0.0, 1.0, 2.0], "axis": "x"}
    assert _status([_curve(y=[5, 6, 7], x=[0.0, 1.0, 2.0])], spec) == fc.PASS


def test_index_selector_out_of_range_is_na():
    spec = {"type": "plot_monotonic", "index": 5}
    assert _status([_curve([0, 1, 2])], spec) == fc.NA


def test_unknown_type_na_and_empty_specs():
    assert _status([_curve([1, 2])], {"type": "bogus"}) == fc.NA
    assert fc.run_field_checks([_curve([1, 2])], []) == []
    assert fc.run_field_checks([], None) == []


def test_capture_current_curves_reads_matplotlib():
    """The capture helper recovers plotted data from live figure state."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0, 2.0], [0.0, 0.5, 1.0])
    curves = fc.capture_current_curves()
    plt.close("all")
    assert len(curves) == 1
    assert curves[0]["y"] == [0.0, 0.5, 1.0]
    assert curves[0]["x"] == [0.0, 1.0, 2.0]
