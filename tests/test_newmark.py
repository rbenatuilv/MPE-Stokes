# tests/test_newmark.py
import numpy as np

from spx.newmark import Newmark


def test_newmark_exact_for_constant_acceleration():
    """d(t) = t^2 has a = 2 and z = 2t; average-acceleration Newmark is exact for it."""
    nm, dt = Newmark(0.25, 0.5, 0.1), 0.1
    d_old, z_old, a_old = np.array([0.0]), np.array([0.0]), np.array([2.0])
    d_new = np.array([dt**2])
    assert np.allclose(nm.acceleration(d_new, d_old, z_old, a_old), 2.0)
    assert np.allclose(nm.velocity(d_new, d_old, z_old, a_old), 2 * dt)
