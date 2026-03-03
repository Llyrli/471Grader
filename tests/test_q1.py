"""Pytest placeholder for Q1 – Problem 2.8 (two springs in series).

Expected: k1=k2=1000 lb/in, F=500 lb, node 1 fixed
  u = [0, 0.5, 1.0] in,  R1 = -500 lb
"""

import numpy as np
import pytest

Q1_EXPECTED = np.array([0.0, 0.5, 1.0])
RTOL = 0.0001


def test_q1_displacements(q1_u):
    """Node displacements must match the series-spring solution."""
    assert q1_u is not None, "Q1 displacement array not provided"
    assert np.allclose(q1_u, Q1_EXPECTED, rtol=RTOL), (
        f"u={q1_u.tolist()}, expected≈{Q1_EXPECTED.tolist()}"
    )
