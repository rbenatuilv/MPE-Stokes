"""Two-field Biot model: solid displacement d and pore pressure p.

Equations (non-dimensional; c = model.c holds the coefficients)
    rho d_tt - div( sigma_e(d) - alpha p I ) = f                       solid momentum
    c0 p_t + alpha div d_t - div( kappa grad p ) + beta p = g           fluid mass
with sigma_e(d) = 2 mu eps(d) + lambda div(d) I and kappa = K / mu_F. The Darcy flux is not
an unknown: it is recovered from p,
    q = -kappa grad p_h            (cellwise, exactly, in DG_{deg_p - 1})
so it is discontinuous across elements and not mass-conserving (see post.normal_jump).

Discretisation: DG P_deg for d, DG P_deg_p for p (default deg_p = deg), SIPG for both
operators, Nitsche for the Dirichlet data. Time: Newmark-beta for d (d_t, d_tt from the new
d), theta-method for the mass equation (theta = 1 backward Euler, 1/2 Crank-Nicolson); with
startup_steps > 0 the first steps use theta = 1 (Rannacher start, damps sudden loads).

Boundary conditions (per named boundary, from the YAML):
    displacement   d = g_d          Nitsche (elasticity) + its velocity in the mass equation
    traction       (sigma_e - alpha p I) n = t      natural
    pressure       p = g_p          Nitsche (Darcy)
    flux           q.n = g_q        natural (outward Darcy flux)

Sources (`sources` in the YAML, optional):  body_force f [N/m^3],  fluid_source g [1/s].
"""
import dolfinx
import ufl

from biotx.common import BiotModel, b_P, elasticity


class TwoField(BiotModel):
    name = "biot2"
    fields = {"d": "displacement", "p": "pressure"}
    newmark = ("d",)
    sources = {"body_force": "body_force", "fluid_source": "rate"}
    material_keys = BiotModel.material_keys + ("beta",)

    def _elements(self):
        d = self.exp.discretization
        self.deg_p = d.deg_p if d.deg_p is not None else d.deg
        return {"d": self.element(d.deg, vector=True), "p": self.element(self.deg_p)}

    def _init_state(self):
        """Old-step state besides Newmark (d, d_t, d_tt): p; theta as a Constant."""
        tc = self.exp.time
        self.p_old = dolfinx.fem.Function(self.V["p"])
        self.theta_target = tc.theta
        self.theta = dolfinx.fem.Constant(self.mesh, 1.0 if tc.startup_steps > 0 else tc.theta)

        # flux q = -kappa grad p, interpolated into DG_{deg_p - 1} after every step
        Vq = dolfinx.fem.functionspace(self.mesh, ("DG", max(self.deg_p - 1, 0), (self.mesh.geometry.dim,)))
        self._q = dolfinx.fem.Function(Vq, name="flux")
        self._q_expr = dolfinx.fem.Expression(-self.c.kappa * ufl.grad(self.u["p"]),
                                              Vq.element.interpolation_points)

    @property
    def pen_darcy(self):
        """SIPG penalty of the Darcy operator, zeta kpen k^2 (kpen = max(kappa, 1))."""
        c = self.c
        return c.zeta * c.kpen * self.k**2

    def darcy(self, p, q):
        """SIPG form of -div(kappa grad p) + beta p, with Nitsche p = . on the pressure
        boundaries (only the part that depends on p; the data is in mass_data)."""
        c, m, n = self.c, self.m, self.m.n
        pen = self.pen_darcy
        F = (c.kappa * ufl.inner(ufl.grad(p), ufl.grad(q)) * m.dx
             - c.kappa * ufl.inner(ufl.avg(ufl.grad(p)), ufl.jump(q, n)) * m.dS
             - c.kappa * ufl.inner(ufl.avg(ufl.grad(q)), ufl.jump(p, n)) * m.dS
             + pen / m.hF * ufl.jump(p) * ufl.jump(q) * m.dS
             + c.beta * p * q * m.dx)
        ds = self.bc.measure("pressure")
        if ds is not None:
            F += (- c.kappa * ufl.inner(ufl.grad(p), n) * q * ds
                  - c.kappa * ufl.inner(ufl.grad(q), n) * p * ds
                  + pen / m.h * p * q * ds)
        return F

    def mass_data(self, q):
        """Right-hand side of the mass equation at the current time t: source, pressure
        (Nitsche) and flux data, and the boundary velocity of a prescribed displacement."""
        c, m, n = self.c, self.m, self.m.n
        pen = self.pen_darcy
        G = self.source("fluid_source") * q * m.dx
        for ds, gp in self.bc.pressure:
            G += - c.kappa * ufl.inner(ufl.grad(q), n) * gp * ds + pen / m.h * gp * q * ds
        for ds, gq in self.bc.flux:
            G += - gq * q * ds                       # kappa grad p . n = -q.n
        for ds, _, gd_t in self.bc.displacement:     # data of the Gamma_d term inside b_P
            G += - c.alpha * q * ufl.inner(gd_t, n) * ds
        return G

    def _residual(self, u, v):
        c, m, th, dt = self.c, self.m, self.theta, self.dt
        d, p, vd, q = u["d"], u["p"], v["d"], v["p"]
        sd = self.sd

        # solid momentum at t^{n+1}; d_tt written through the new d (Newmark)
        F = (c.rho * ufl.inner(sd.a_new(d), vd) * m.dx
             + elasticity(d, vd, self)
             + c.alpha * b_P(p, vd, self)
             - ufl.inner(self.source("body_force", (self.mesh.geometry.dim,)), vd) * m.dx)
        for ds, tr in self.bc.traction:
            F -= ufl.inner(tr, vd) * ds

        # fluid mass with the theta-method:
        #   c0 (p - p_old)/dt - alpha b_P(q, z_theta) + theta A(p) + (1 - theta) A(p_old)
        #     = theta G(t^{n+1}) + (1 - theta) G(t^n),    z_theta = theta z^{n+1} + (1 - theta) z^n
        z = th * sd.z_new(d) + (1 - th) * sd.z_old
        G = self.mass_data(q)
        F += (c.c0 * (p - self.p_old) / dt * q * m.dx
              - c.alpha * b_P(q, z, self)
              + th * self.darcy(p, q) + (1 - th) * self.darcy(self.p_old, q)
              - th * G - (1 - th) * self.at_old_time(G))
        return F

    def _advance(self):
        self.sd.advance(self.u["d"])
        self.p_old.x.array[:] = self.u["p"].x.array
        self.p_old.x.scatter_forward()
        self._q.interpolate(self._q_expr)
        if self.step + 1 == self.exp.time.startup_steps and self.theta_target != 1.0:
            self.theta.value = self.theta_target       # end of the Rannacher start
            return True                                # the matrix depends on theta
        return False

    def _set_initial(self, v):
        super()._set_initial(v)
        self.p_old.x.array[:] = self.u["p"].x.array
        self._q.interpolate(self._q_expr)

    def strong_form(self, F, C):
        """Sources of (B1)-(B2) for which the fields F (SI UFL) are an exact solution; C
        holds the SI calculus (mms.Manufactured)."""
        m = self.exp.material
        d, p = F["d"], F["p"]
        f = m.rho * C.dt(d, 2) - C.div(C.sigma_e(d) - m.alpha * p * ufl.Identity(len(d)))
        g = (m.c0 * C.dt(p) + m.alpha * C.div(C.dt(d)) - C.div(m.K / m.muF * C.grad(p))
             + m.beta * p)
        return {"body_force": f, "fluid_source": g}

    def exact_flux(self, F, C):
        """q = -kappa grad p of the fields F (SI)."""
        return -self.exp.material.K / self.exp.material.muF * C.grad(F["p"])

    def flux(self):
        """Darcy flux q = -kappa grad p_h (non-dimensional), DG_{deg_p - 1}."""
        return self._q

    # -- the mass balance over one step (flux.py: the formulas are in its docstring)
    def trace_degree(self):
        return self.deg_p

    def numerical_flux(self, old):
        """theta Fhat(p^{n+1}) + (1 - theta) Fhat(p^n), Fhat(p) out of K: the SIPG flux of darcy()."""
        c, m, n = self.c, self.m, self.m.n
        th, pen = old.th, self.pen_darcy

        def at_level(p, at):
            interior = -c.kappa * ufl.inner(ufl.avg(ufl.grad(p)), n("+")) + pen / m.hF * ufl.jump(p)
            bnd = [(ds, -c.kappa * ufl.inner(ufl.grad(p), n) + pen / m.h * (p - at(gp))) for ds, gp in self.bc.pressure]
            return interior, bnd + [(ds, at(gq)) for ds, gq in self.bc.flux]

        (i1, b1), (i0, b0) = at_level(self.u["p"], lambda e: e), at_level(old.u["p"], old.at_old_time)
        return th * i1 + (1 - th) * i0, [(ds, th * a + (1 - th) * b) for (ds, a), (_, b) in zip(b1, b0)]

    def step_velocities(self, old):
        q1, q0 = -self.c.kappa * ufl.grad(self.u["p"]), -self.c.kappa * ufl.grad(old.u["p"])
        return {"end": q1, "avg": old.th * q1 + (1 - old.th) * q0}

    def fluid_gained(self, old, q):
        c, m, n = self.c, self.m, self.m.n
        th, p, p0 = old.th, self.u["p"], old.u["p"]
        z_th = th * self.sd.z_old + (1 - th) * old.z["d"]
        g = self.source("fluid_source")
        G = (c.c0 / self.dt * (p - p0) * q * m.dx + c.beta * (th * p + (1 - th) * p0) * q * m.dx
             - (th * g + (1 - th) * old.at_old_time(g)) * q * m.dx - c.alpha * b_P(q, z_th, self))
        for ds, _, gd_t in self.bc.displacement:
            G += c.alpha * q * ufl.inner(th * gd_t + (1 - th) * old.at_old_time(gd_t), n) * ds
        return G

    def flux_lifting(self, old, psi, dS):
        """Lifting of the symmetric SIPG term: sum_F int_F omega kappa (psi.n) [[p_th]]."""
        c, n, th = self.c, self.m.n, old.th
        p_th = th * self.u["p"] + (1 - th) * old.u["p"]
        L = c.kappa * ufl.jump(p_th) * ufl.inner(ufl.avg(psi), n("+")) * dS
        for ds, gp in self.bc.pressure:
            g_th = th * gp + (1 - th) * old.at_old_time(gp)
            L += c.kappa * (p_th - g_th) * ufl.inner(psi, n) * ds
        return L
