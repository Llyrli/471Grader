"""Unit tests for reference-free physics-plausibility checks."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import physics_checks as pc  # noqa: E402


def _status(ns, spec):
    return pc.run_one(ns, spec)["status"]


def test_finite_pass_and_fail():
    assert _status({"u": np.array([0.0, 0.5, 1.0])}, {"type": "finite", "var": "u"}) == pc.PASS
    assert _status({"u": np.array([0.0, np.nan])}, {"type": "finite", "var": "u"}) == pc.FAIL
    assert _status({"u": np.array([np.inf])}, {"type": "finite", "var": "u"}) == pc.FAIL


def test_missing_variable_is_na_not_violation():
    rep = pc.run_one({}, {"type": "finite", "var": "u", "name": "fin"})
    assert rep["status"] == pc.NA
    assert pc.violations([rep]) == []  # NA never counts as a violation


def test_symmetric():
    K = np.array([[2.0, -1.0], [-1.0, 2.0]])
    assert _status({"K": K}, {"type": "symmetric", "var": "K"}) == pc.PASS
    Kbad = np.array([[2.0, -1.0], [-0.5, 2.0]])
    assert _status({"K": Kbad}, {"type": "symmetric", "var": "K"}) == pc.FAIL
    # Non-square → not applicable, not a failure.
    assert _status({"K": np.zeros((2, 3))}, {"type": "symmetric", "var": "K"}) == pc.NA


def test_psd():
    spd = np.array([[2.0, -1.0], [-1.0, 2.0]])     # eigenvalues 1, 3
    assert _status({"K": spd}, {"type": "psd", "var": "K"}) == pc.PASS
    indef = np.array([[0.0, 1.0], [1.0, 0.0]])     # eigenvalues -1, 1
    assert _status({"K": indef}, {"type": "psd", "var": "K"}) == pc.FAIL


def test_dirichlet_fixed_dof_zero():
    spec = {"type": "dirichlet", "var": "u", "dofs": [0]}
    assert _status({"u": np.array([0.0, 0.5, 1.0])}, spec) == pc.PASS
    assert _status({"u": np.array([0.3, 0.5, 1.0])}, spec) == pc.FAIL
    # dof out of range → na
    assert _status({"u": np.array([0.0])}, {"type": "dirichlet", "var": "u", "dofs": [5]}) == pc.NA


def test_residual_satisfied_system():
    A = np.array([[2.0, -1.0], [-1.0, 2.0]])
    x = np.array([1.0, 1.0])
    b = A @ x
    ns = {"A": A, "x": x, "b": b}
    spec = {"type": "residual", "matrix": "A", "x": "x", "b": "b"}
    assert _status(ns, spec) == pc.PASS
    ns_bad = {"A": A, "x": np.array([0.0, 0.0]), "b": b}
    assert _status(ns_bad, spec) == pc.FAIL
    # missing operand → na
    assert _status({"A": A, "x": x}, spec) == pc.NA


def test_net_sum_force_balance():
    spec = {"type": "net_sum", "var": "R", "target": 0.0}
    assert _status({"R": np.array([500.0, -500.0])}, spec) == pc.PASS
    assert _status({"R": np.array([500.0, -300.0])}, spec) == pc.FAIL


def test_bounded():
    spec = {"type": "bounded", "var": "u", "max_abs": 10.0}
    assert _status({"u": np.array([1.0, -9.9])}, spec) == pc.PASS
    assert _status({"u": np.array([1.0, 1e6])}, spec) == pc.FAIL


def test_candidate_alias_resolution():
    # primary name absent, alias present
    spec = {"type": "finite", "var": "Kg", "candidates": ["K", "Kglobal"]}
    assert _status({"Kglobal": np.array([1.0, 2.0])}, spec) == pc.PASS


def test_unknown_type_is_na():
    assert _status({"u": np.array([1.0])}, {"type": "bogus", "var": "u"}) == pc.NA


def test_run_physics_checks_and_violations():
    ns = {"K": np.array([[0.0, 1.0], [1.0, 0.0]]), "u": np.array([0.0, np.nan])}
    specs = [
        {"name": "sym", "type": "symmetric", "var": "K"},   # pass
        {"name": "psd", "type": "psd", "var": "K"},          # fail (indefinite)
        {"name": "fin", "type": "finite", "var": "u"},       # fail (nan)
        {"name": "missing", "type": "finite", "var": "zzz"},  # na
    ]
    reports = pc.run_physics_checks(ns, specs)
    assert len(reports) == 4
    assert set(pc.violations(reports)) == {"psd", "fin"}


def test_empty_specs():
    assert pc.run_physics_checks({"u": np.array([1.0])}, []) == []
    assert pc.run_physics_checks({}, None) == []
