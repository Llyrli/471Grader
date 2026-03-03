"""Pytest placeholder for Q3 – Problem 2.15 (branching spring network).

Expected: k=[500,500,1000] kN/m, F=4 kN at node 3, u1=u2=u4=0 (fixed)
  u = [0, 0, 0.002, 0] m
"""

import numpy as np
import pytest

Q3_EXPECTED = np.array([0.0, 0.0, 0.002, 0.0])
RTOL = 0.0001


def test_q3_displacements(q3_u):
    """Node displacements must match the branching-network solution."""
    assert q3_u is not None, "Q3 displacement array not provided"
    assert np.allclose(q3_u, Q3_EXPECTED, rtol=RTOL), (
        f"u={q3_u.tolist()}, expected≈{Q3_EXPECTED.tolist()}"
    )
