"""Pytest placeholder for Q2 – Problem 2.11 (prescribed displacement).

Expected: k1=1000 N/m, k2=3000 N/m, u1=0 (fixed), u3=0.02 m (prescribed)
  u = [0, 0.015, 0.02] m
"""

import numpy as np
import pytest

Q2_EXPECTED = np.array([0.0, 0.015, 0.02])
RTOL = 0.0001


def test_q2_displacements(q2_u):
    """Node displacements must match the prescribed-displacement solution."""
    assert q2_u is not None, "Q2 displacement array not provided"
    assert np.allclose(q2_u, Q2_EXPECTED, rtol=RTOL), (
        f"u={q2_u.tolist()}, expected≈{Q2_EXPECTED.tolist()}"
    )
