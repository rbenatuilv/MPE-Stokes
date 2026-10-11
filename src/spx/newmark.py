"""Newmark-beta for d, and the state carried from one time step to the next."""
from dataclasses import dataclass

import dolfinx
from petsc4py import PETSc


@dataclass(frozen=True)
class Newmark:
    """Newmark-beta formulas: new acceleration and velocity of d from the new d and the old state."""
    beta: float
    gamma: float
    dt: float

    def acceleration(self, d, d_old, z_old, a_old):          # theory.md (T1)
        """a^{n+1} from d^{n+1} and (d, z, a)^n. Works on UFL expressions and on NumPy arrays."""
        b, dt = self.beta, self.dt
        return (d - d_old) / (b * dt**2) - z_old / (b * dt) - (1 - 2 * b) / (2 * b) * a_old

    def velocity(self, d, d_old, z_old, a_old):              # (T2) with (T1) substituted
        """z^{n+1} (= d_t d) from d^{n+1} and (d, z, a)^n. Works on UFL expressions and arrays."""
        b, g, dt = self.beta, self.gamma, self.dt
        return (g / (b * dt)) * (d - d_old) - (g / b - 1) * z_old - dt * (g / (2 * b) - 1) * a_old


class TimeState:
    """Old-step Functions + dt; physics modules read it, the time loop advances it."""

    def __init__(self, V, mesh, time_cfg):
        """Create the Newmark formulas, the dt Constant and the 'old' Functions (all zero)."""
        self.nm = Newmark(time_cfg.newmark_beta, time_cfg.newmark_gamma, time_cfg.dt)
        self.dt = dolfinx.fem.Constant(mesh, PETSc.ScalarType(time_cfg.dt))
        self.uS_old = dolfinx.fem.Function(V["uS"])
        self.pP_old = dolfinx.fem.Function(V["pP"])
        self.d_old = dolfinx.fem.Function(V["d"])
        self.z_old = dolfinx.fem.Function(V["d"])
        self.a_old = dolfinx.fem.Function(V["d"])

    # UFL expressions of the NEW velocity/acceleration in terms of the new d
    def z_new(self, d):
        """UFL expression of the new velocity in terms of the new d (used inside the forms)."""
        return self.nm.velocity(d, self.d_old, self.z_old, self.a_old)

    def a_new(self, d):
        """UFL expression of the new acceleration in terms of the new d (used inside the forms)."""
        return self.nm.acceleration(d, self.d_old, self.z_old, self.a_old)

    def advance(self, sol):
        """After a solve: (z, a) <- new values, then old <- new. ORDER MATTERS."""
        d = sol["d"].x.array
        a_new = self.nm.acceleration(d, self.d_old.x.array, self.z_old.x.array, self.a_old.x.array)
        z_new = self.nm.velocity(d, self.d_old.x.array, self.z_old.x.array, self.a_old.x.array)
        self.a_old.x.array[:] = a_new
        self.z_old.x.array[:] = z_new
        self.d_old.x.array[:] = d
        self.uS_old.x.array[:] = sol["uS"].x.array
        self.pP_old.x.array[:] = sol["pP"].x.array
        for f in (self.a_old, self.z_old, self.d_old, self.uS_old, self.pP_old):
            f.x.scatter_forward()
