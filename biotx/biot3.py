"""Three-field Biot model: solid displacement d, filtration displacement w,
pore pressure p. w is the relative fluid displacement phi (u_fluid - u_solid), so its time
derivative IS the Darcy flux:
    q = w_t                       (an unknown of the system, H(div)-like: [[w.n]] penalised)

Equations (non-dimensional; c = model.c)
    rho d_tt + rhoF w_tt - div( sigma_e(d) - alpha p I ) = f_d          solid momentum
    rhoF d_tt + rhoW w_tt + (1/kappa) w_t + grad p        = f_w          fluid momentum (Darcy + inertia)
    c0 p + alpha div d + div w                            = f_p          mass (no time derivative)
with rhoF = fluid density, rhoW = rhoF * tortuosity / phi, kappa = K / mu_F. With rho_f = 0
the fluid momentum is Darcy's law and the model reduces to the two-field one.

Discretisation: DG P_deg for d and w, DG P_deg_p for p (default deg - 1: P2/P2/P1 is the stable
choice; equal orders lose a rate in p). SIPG elasticity for d; for w only the normal jump is
penalised (eta (lambda + 2 mu) k^2 / hF). Time: Newmark-beta for both d and w.

Boundary conditions (per named boundary):
    displacement   d = g_d          Nitsche
    traction       (sigma_e - alpha p I) n = t      natural
    pressure       p = g_p          natural for w (enters through -(p, div wt))
    flux           q.n = 0          w.n = 0, penalised (only zero flux is supported here)

Sources (`sources`, optional): body_force f_d [N/m^3], fluid_force f_w [N/m^3],
mass_source f_p [-].
"""
import dolfinx
import ufl

from biotx.common import BiotModel, NewmarkField, b_P, elasticity


def _is_zero(value):
    return isinstance(value, (int, float)) and value == 0


class ThreeField(BiotModel):
    fields = ("d", "w", "p")

    def __init__(self, exp, domain=None):
        bad = [n for n, bc in exp.boundaries.items()
               if "flux" in bc and not (_is_zero(bc["flux"]) or bc["flux"] == "exact")]
        if bad:
            raise NotImplementedError(f"biot3: only flux: 0 is supported (boundaries {bad})")
        super().__init__(exp, domain)
        ds = self.bc.measure("flux")
        if self.mms is not None and ds is not None:           # flux: exact must mean w.n = 0
            self.mms.require_zero(ufl.inner(self.mms.exact("w"), self.m.n), ds,
                                  "w.n on the flux boundaries (biot3 supports only flux: 0)")

    def _elements(self):
        d = self.exp.discretization
        self.deg_p = d.deg_p if d.deg_p is not None else max(d.deg - 1, 0)
        return {"d": self.element(d.deg, vector=True), "w": self.element(d.deg, vector=True),
                "p": self.element(self.deg_p)}

    def _init_state(self):
        self.sd = NewmarkField(self.V["d"], self.nm)
        self.sw = NewmarkField(self.V["w"], self.nm)

    def b_W(self, p, w):
        """DG form of (grad p, w) = -(p, div w) + <{p}, [[w.n]]> + <p, w.n>_{flux boundary}.
        On pressure boundaries the term <p, w.n> carries the given pressure (right-hand side)."""
        m, n = self.m, self.m.n
        F = -p * ufl.div(w) * m.dx + ufl.avg(p) * ufl.jump(w, n) * m.dS
        ds = self.bc.measure("flux")
        if ds is not None:
            F += p * ufl.inner(w, n) * ds
        return F

    def _residual(self, u, v):
        c, m, n = self.c, self.m, self.m.n
        d, w, p, vd, vw, q = u["d"], u["w"], u["p"], v["d"], v["w"], v["p"]
        sd, sw = self.sd, self.sw
        dim = self.mesh.geometry.dim
        pen_w = c.eta * (c.lam + 2 * c.mu) * self.k**2

        # solid momentum
        F = (c.rho * ufl.inner(sd.a_new(d), vd) * m.dx
             + c.rhoF * ufl.inner(sw.a_new(w), vd) * m.dx
             + elasticity(d, vd, self)
             + c.alpha * b_P(p, vd, self)
             - ufl.inner(self.source("body_force", "body_force", (dim,)), vd) * m.dx)
        for ds, tr in self.bc.traction:
            F -= ufl.inner(tr, vd) * ds

        # fluid momentum: inertia + Darcy drag (1/kappa) w_t + grad p
        F += (c.rhoF * ufl.inner(sd.a_new(d), vw) * m.dx
              + c.rhoW * ufl.inner(sw.a_new(w), vw) * m.dx
              + 1 / c.kappa * ufl.inner(sw.z_new(w), vw) * m.dx
              + self.b_W(p, vw)
              + pen_w / m.hF * ufl.jump(w, n) * ufl.jump(vw, n) * m.dS
              - ufl.inner(self.source("fluid_force", "body_force", (dim,)), vw) * m.dx)
        ds = self.bc.measure("flux")
        if ds is not None:                                    # w.n = 0, penalised
            F += pen_w / m.h * ufl.inner(w, n) * ufl.inner(vw, n) * ds
        for ds, gp in self.bc.pressure:                       # p given: natural
            F += gp * ufl.inner(vw, n) * ds

        # mass: c0 p + alpha div d + div w = f_p
        F += (c.c0 * p * q * m.dx
              - c.alpha * b_P(q, d, self)
              - self.b_W(q, w)
              - self.source("mass_source", "none") * q * m.dx)
        for ds, gd, _ in self.bc.displacement:                # data of the Gamma_d term in b_P
            F += c.alpha * q * ufl.inner(gd, n) * ds
        return F

    def _advance(self):
        self.sd.advance(self.u["d"])
        self.sw.advance(self.u["w"])
        return False

    def _set_initial(self, v):
        f = self.exp.scales.factor
        for s, name in ((self.sd, "d"), (self.sw, "w")):
            for func, key, kind in ((s.u_old, name, "displacement"), (s.z_old, name + "_t", "velocity"),
                                    (s.a_old, name + "_tt", "acceleration")):
                if key in v:
                    self.interpolate(func, f[kind] * v[key])
            self.u[name].x.array[:] = s.u_old.x.array

    def strong_form(self, F, C):
        """Sources of (B3)-(B5) for which the fields F (SI UFL) are an exact solution."""
        m = self.exp.material
        d, w, p = F["d"], F["w"], F["p"]
        rhoW = m.rho_f * m.tortuosity / m.phi
        f_d = (m.rho * C.dt(d, 2) + m.rho_f * C.dt(w, 2)
               - C.div(C.sigma_e(d) - m.alpha * p * ufl.Identity(len(d))))
        f_w = m.rho_f * C.dt(d, 2) + rhoW * C.dt(w, 2) + m.muF / m.K * C.dt(w) + C.grad(p)
        f_p = m.c0 * p + m.alpha * C.div(d) + C.div(w)
        return {"body_force": f_d, "fluid_force": f_w, "mass_source": f_p}

    def exact_flux(self, F, C):
        """q = w_t of the fields F (SI)."""
        return C.dt(F["w"])

    def flux(self):
        """Darcy flux q = w_t (non-dimensional): the Newmark velocity of w."""
        return self.sw.z_old
