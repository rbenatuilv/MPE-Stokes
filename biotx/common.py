"""What the Biot models share: DG building blocks, Newmark, boundary data, the linear
solver and the `BiotModel` base class (mesh, constants, mixed space, time loop).

A model (biot2.py, biot3.py) is a subclass that declares
    name             the `model:` value in the YAML; defining the class registers it
    fields           the unknowns and the kind of each (its SI conversion), e.g.
                     {"d": "displacement", "p": "pressure"}
    newmark          the fields integrated with Newmark (the base class keeps their old
                     value, velocity and acceleration in self.nmk[f])
    sources          the `sources` it reads and the kind of each, e.g. {"fluid_source": "rate"}
    material_keys    the `material` parameters it reads (others given are reported as unused)
and implements
    _elements()      the DG element of each field
    _init_state()    any old-step state beyond self.nmk (biot2: p_old, theta)
    _residual()      the weak form of one time step, F(trial, test) = 0
    _advance()       how the old-step state is updated after a solve (True: the matrix changed)
    flux()           the Darcy flux q = phi (v_fluid - v_solid)
    strong_form(), exact_flux()      sources and flux of an exact solution (mms.py)
and, for the conservative velocity (flux.py), its step-consistent mass balance
    trace_degree(), numerical_flux(old), step_velocities(old), fluid_gained(old, q), flux_lifting(old, psi, dS)
Everything else (assembly, LU factorisation, time loop, initial state, outputs) is done here.

Notation, as in the models:  [[.]] jump, {.} average on interior facets; n outward normal;
h cell diameter, hF harmonic mean of the two diameters on an interior facet; k = deg.
"""
from types import SimpleNamespace

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from petsc4py import PETSc

from biotx import geometry
from biotx.expressions import Expressions
from biotx.mms import Manufactured

MUMPS = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
         "mat_mumps_icntl_14": 80}             # extra MUMPS workspace


# ---- DG building blocks ----------------------------------------------------------------
def eps(u):
    """Small-strain tensor."""
    return ufl.sym(ufl.grad(u))


def sigma_e(u, c):
    """Effective (elastic) stress 2 mu eps(u) + lambda div(u) I."""
    return 2 * c.mu * eps(u) + c.lam * ufl.div(u) * ufl.Identity(len(u))


def sym_jump(u, n):
    """Symmetrised jump [[u (x) n]] of a vector field on an interior facet."""
    return ufl.sym(ufl.outer(u("+"), n("+")) + ufl.outer(u("-"), n("-")))


def sym_bnd(u, n):
    """Symmetrised u (x) n on a boundary facet."""
    return ufl.sym(ufl.outer(u, n))


def elasticity(d, v, M):
    """SIPG residual of -div sigma_e(d) with Nitsche d = g_d on the displacement boundaries:
        (sigma_e(d), eps(v)) - <{sigma_e(d)}, [[v n]]> - <{sigma_e(v)}, [[d n]]>
          + pen/hF <[[d n]], [[v n]]>                                     (interior facets)
          - <sigma_e(d), (v n)> - <sigma_e(v), ((d - g_d) n)> + pen/h <(d - g_d) n, v n>   (Gamma_d)
    pen = eta (lambda + 2 mu) k^2 (M.pen_elastic)."""
    c, m, n = M.c, M.m, M.m.n
    pen = M.pen_elastic
    F = (ufl.inner(sigma_e(d, c), eps(v)) * m.dx
         - ufl.inner(ufl.avg(sigma_e(d, c)), sym_jump(v, n)) * m.dS
         - ufl.inner(ufl.avg(sigma_e(v, c)), sym_jump(d, n)) * m.dS
         + pen / m.hF * ufl.inner(sym_jump(d, n), sym_jump(v, n)) * m.dS)
    for ds, g, _ in M.bc.displacement:
        F += (- ufl.inner(sigma_e(d, c), sym_bnd(v, n)) * ds
              - ufl.inner(sigma_e(v, c), sym_bnd(d - g, n)) * ds
              + pen / m.h * ufl.inner(sym_bnd(d - g, n), sym_bnd(v, n)) * ds)
    return F


def b_P(p, v, M):
    """DG form of (grad p, v) = -(p, div v) + <{p}, [[v.n]]> + <p, v.n>_{Gamma_d}.
    Used as +alpha b_P(p, v) in the solid momentum equation and -alpha b_P(q, d_t) in the
    mass equation, so the discrete coupling is exactly antisymmetric (energy-stable)."""
    m, n = M.m, M.m.n
    F = -p * ufl.div(v) * m.dx + ufl.avg(p) * ufl.jump(v, n) * m.dS
    ds = M.bc.measure("displacement")
    if ds is not None:
        F += p * ufl.inner(v, n) * ds
    return F


# ---- time scheme -------------------------------------------------------------------------
class Newmark:
    """Newmark-beta: the new acceleration and velocity in terms of the new displacement u
    and the old (u, z, a). Works on UFL expressions (inside forms) and on NumPy arrays."""

    def __init__(self, beta, gamma, dt):
        self.beta, self.gamma, self.dt = beta, gamma, dt

    def acceleration(self, u, u_old, z_old, a_old):
        b, dt = self.beta, self.dt
        return (u - u_old) / (b * dt**2) - z_old / (b * dt) - (1 - 2 * b) / (2 * b) * a_old

    def velocity(self, u, u_old, z_old, a_old):
        b, g, dt = self.beta, self.gamma, self.dt
        return (g / (b * dt)) * (u - u_old) - (g / b - 1) * z_old - dt * (g / (2 * b) - 1) * a_old


class NewmarkField:
    """Old displacement, velocity and acceleration of one Newmark-integrated field."""

    def __init__(self, V, nm):
        self.nm = nm
        self.u_old, self.z_old, self.a_old = (dolfinx.fem.Function(V) for _ in range(3))

    def z_new(self, u):
        """New velocity as a UFL expression of the new displacement u."""
        return self.nm.velocity(u, self.u_old, self.z_old, self.a_old)

    def a_new(self, u):
        """New acceleration as a UFL expression of the new displacement u."""
        return self.nm.acceleration(u, self.u_old, self.z_old, self.a_old)

    def advance(self, u):
        """After a solve: new a and z (from the old u), then u_old <- u."""
        x, old = u.x.array, (self.u_old.x.array, self.z_old.x.array, self.a_old.x.array)
        a, z = self.nm.acceleration(x, *old), self.nm.velocity(x, *old)
        self.a_old.x.array[:], self.z_old.x.array[:], self.u_old.x.array[:] = a, z, x
        for f in (self.a_old, self.z_old, self.u_old):
            f.x.scatter_forward()


# ---- boundary data -----------------------------------------------------------------------
class BoundaryConditions:
    """The `boundaries` section, grouped by kind, non-dimensional:
        displacement: [(ds, d*, d*_t)]     traction: [(ds, total traction*)]
        pressure:     [(ds, p*)]           flux:     [(ds, outward Darcy flux q.n *)]
    normal_traction s is the traction s n (a fluid pressure P on a wall: s = -P)."""

    def __init__(self, exp, domain, ex, m, exact=None):
        """exact: the model's Manufactured solution, for conditions given as `exact`."""
        f = exp.scales.factor
        known, given = set(domain.names), set(exp.boundaries)
        if given - known:
            raise ValueError(f"unknown boundaries {sorted(given - known)}; the mesh has {sorted(known)}")
        if known - given:
            raise ValueError(f"no conditions for boundaries {sorted(known - given)}")
        self.tags = {k: [] for k in ("displacement", "traction", "pressure", "flux")}
        self.displacement, self.traction, self.pressure, self.flux = [], [], [], []
        dim = domain.mesh.geometry.dim

        def value(bc, key):
            v = bc[key]
            return exact.boundary(key, m.n) if isinstance(v, str) and v == "exact" else ex(v)

        for name, bc in exp.boundaries.items():
            for key in ("displacement", "traction"):
                v = bc.get(key)
                if isinstance(v, (list, tuple)) and len(v) != dim:
                    raise ValueError(f"boundary '{name}': {key} needs {dim} components, got {len(v)}")
            tag = domain.names[name]
            ds = m.ds(tag)
            if "displacement" in bc:
                g = value(bc, "displacement")
                self.displacement.append((ds, f["displacement"] * g, f["velocity"] * ex.ddt(g)))
                self.tags["displacement"].append(tag)
            else:
                tr = value(bc, "traction") if "traction" in bc else ex(bc["normal_traction"]) * m.n
                self.traction.append((ds, f["pressure"] * tr))
                self.tags["traction"].append(tag)
            if "pressure" in bc:
                self.pressure.append((ds, f["pressure"] * value(bc, "pressure")))
                self.tags["pressure"].append(tag)
            else:
                self.flux.append((ds, f["velocity"] * value(bc, "flux")))
                self.tags["flux"].append(tag)
        self._ds = m.ds

    def measure(self, kind):
        """ds over all boundaries of one kind, or None."""
        tags = self.tags[kind]
        return self._ds(tuple(tags)) if tags else None


# ---- linear solver -----------------------------------------------------------------------
_ksp_count = 0


def make_ksp(A, options=None):
    """KSP for the matrix A: MUMPS LU plus the extra PETSc options, under its own options
    prefix (so that several solvers in one script do not interfere)."""
    global _ksp_count
    _ksp_count += 1
    prefix = f"biotx{_ksp_count}_"
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    values = {**MUMPS, **(options or {})}
    for k, v in values.items():
        opts[prefix + k] = v
    ksp.setFromOptions()
    for k in values:                       # read: drop them from the global options database
        opts.delValue(prefix + k)
    return ksp


class LinearSolver:
    """A assembled and LU-factorised once; the right-hand side reassembled every step."""

    def __init__(self, a, L, W, options=None):
        self.a, self.L = dolfinx.fem.form(a), dolfinx.fem.form(L)
        self.A = dolfinx.fem.petsc.assemble_matrix(self.a)
        self.A.assemble()
        self.b = dolfinx.fem.petsc.create_vector(W)
        self.ksp = make_ksp(self.A, options)

    def update_matrix(self):
        """Reassemble A (after a Constant in the bilinear form changed); refactorises."""
        self.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.A, self.a)
        self.A.assemble()
        self.ksp.setOperators(self.A)

    def solve(self, u):
        """Solve into the Function u (of the mixed space)."""
        with self.b.localForm() as bl:
            bl.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.ksp.solve(self.b, u.x.petsc_vec)
        if self.ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"linear solve failed, reason {self.ksp.getConvergedReason()}")
        u.x.scatter_forward()


# ---- the model base class ----------------------------------------------------------------
REGISTRY = {}                          # name -> model class (biotx.models.make_model)


class BiotModel:
    """Mesh, constants, boundary data, mixed DG space, solver and time loop of one experiment.

        model = TwoField(exp)          # or ThreeField(exp); exp from biotx.load(...)
        for t in model.steps():        # t in seconds; the state is already updated
            ...                        # model.u["d"], model.u["p"], model.flux(), ...
    """

    name: str = ""
    fields: dict = {}
    newmark: tuple = ()
    sources: dict = {}
    material_keys: tuple = ("lam", "mu", "alpha", "c0", "K", "muF", "rho")

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if "name" in vars(cls):
            REGISTRY[cls.name] = cls

    def __init__(self, exp, domain=None):
        """domain: an already built geometry.Domain (to reuse one mesh in several runs)."""
        self.exp, s, tc, disc = exp, exp.scales, exp.time, exp.discretization
        self.domain = domain or geometry.build(exp.geometry, s.L0, exp.root)
        mesh = self.mesh = self.domain.mesh
        const = lambda v: dolfinx.fem.Constant(mesh, PETSc.ScalarType(v))

        # non-dimensional time: t is t^{n+1} inside the forms, t_old is t^n
        self.dt = tc.dt / s.T0
        self.t, self.t_old = const(tc.t0 / s.T0), const(tc.t0 / s.T0)
        self.time, self.step = tc.t0, 0

        # coefficients as Constants (they can be changed later: then call solver.update_matrix)
        c = vars(exp.material.scaled(s))
        c.update(eta=disc.eta, zeta=disc.zeta, kpen=max(c["kappa"], 1.0))   # kpen: Darcy penalty scale
        self.c = SimpleNamespace(**{k: const(v) for k, v in c.items()})
        self.k = disc.deg

        # measures (quadrature 2k+1), normal and cell sizes
        h = ufl.CellDiameter(mesh)
        md = {"quadrature_degree": 2 * disc.deg + 1}
        self.m = SimpleNamespace(
            dx=ufl.Measure("dx", domain=mesh, metadata=md),
            ds=ufl.Measure("ds", domain=mesh, subdomain_data=self.domain.facet_tags, metadata=md),
            dS=ufl.Measure("dS", domain=mesh, metadata=md),
            n=ufl.FacetNormal(mesh), h=h, hF=2 * h("+") * h("-") / (h("+") + h("-")))

        # data: loads, boundary conditions and sources, all non-dimensional UFL
        self.ex = Expressions(mesh, s, self.t, exp.params, exp.loads)
        self.mms = Manufactured(self) if exp.exact else None        # sources, BC data, initial state
        self.bc = BoundaryConditions(exp, self.domain, self.ex, self.m, self.mms)
        self.nm = Newmark(tc.newmark_beta, tc.newmark_gamma, self.dt)
        # weight of t^{n+1} in the time scheme: 1 unless the model has a theta-method (biot2)
        self.theta_target, self.theta = 1.0, const(1.0)

        # mixed DG space, one collapsed space and Function per field
        el = self._elements()
        self.W = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([el[f] for f in self.fields]))
        self.V, self._dofs, self.u = {}, {}, {}
        for i, f in enumerate(self.fields):
            V, dofs = self.W.sub(i).collapse()
            self.V[f], self._dofs[f] = V, np.asarray(dofs).reshape(-1)
            self.u[f] = dolfinx.fem.Function(V, name=f)
        self.sol = dolfinx.fem.Function(self.W)
        self.reconstruction = None                     # biotx.flux.Reconstruction, made on request (cflux)

        self.nmk = {f: NewmarkField(self.V[f], self.nm) for f in self.newmark}
        self._init_state()
        F = self._residual(dict(zip(self.fields, ufl.TrialFunctions(self.W))),
                           dict(zip(self.fields, ufl.TestFunctions(self.W))))
        self.solver = LinearSolver(ufl.lhs(F), ufl.rhs(F), self.W, exp.solver)
        if self.mms is not None:
            self.set_initial(**self.mms.initial())

    # -- to be provided by the models
    def _elements(self): raise NotImplementedError
    def _init_state(self): pass
    def _residual(self, u, v): raise NotImplementedError
    def _advance(self) -> bool: raise NotImplementedError
    def flux(self): raise NotImplementedError
    def strong_form(self, fields, C): raise NotImplementedError    # sources for given fields (mms.py)
    def exact_flux(self, fields, C): raise NotImplementedError
    # the mass balance over one step (flux.py); old: a flux.StepState (the state at t^n)
    def trace_degree(self) -> int: raise NotImplementedError        # normal traces of the RT velocity
    def numerical_flux(self, old): raise NotImplementedError        # (interior, [(ds, boundary)])
    def step_velocities(self, old): raise NotImplementedError       # {"end": ..., "avg": ...}
    def fluid_gained(self, old, q): raise NotImplementedError       # mass equation without its flux part
    def flux_lifting(self, old, psi, dS): return None               # cell term of the RT velocity, if any

    @property
    def units(self):
        """Kind of each output quantity (its SI conversion, Scales.factor)."""
        return {**self.fields, "flux": "velocity", "velocity": "velocity", "cflux": "velocity"}

    @property
    def sd(self):
        """The Newmark state of the solid displacement d."""
        return self.nmk["d"]

    @property
    def pen_elastic(self):
        """SIPG penalty of elasticity, eta (lambda + 2 mu) k^2."""
        c = self.c
        return c.eta * (c.lam + 2 * c.mu) * self.k**2

    # -- helpers for the models
    def element(self, deg, vector=False):
        """DG element of degree deg (vector-valued if vector)."""
        shape = (self.mesh.geometry.dim,) if vector else ()
        return basix.ufl.element("DG", self.mesh.basix_cell(), deg, shape=shape)

    def source(self, name, shape=()):
        """A `sources` entry, non-dimensional (zero if absent)."""
        kind = self.sources[name]
        if self.mms is not None:                                   # derived from `exact`
            return self.exp.scales.factor[kind] * self.mms.sources[name]
        value = self.exp.sources.get(name, np.zeros(shape).tolist() if shape else 0.0)
        return self.exp.scales.factor[kind] * self.ex(value)

    def at_old_time(self, form):
        """The same expression or form with t^{n+1} replaced by t^n."""
        return ufl.replace(form, {self.t: self.t_old})

    def interpolate(self, f, expr):
        f.interpolate(dolfinx.fem.Expression(expr, f.function_space.element.interpolation_points))

    def _set_initial(self, v):
        """v: non-dimensional UFL values. A Newmark field f takes f, f_t, f_tt (its old state and
        the current value); any other field takes f (its current value)."""
        f = self.exp.scales.factor
        for name, kind in self.fields.items():
            if name in self.nmk:
                s = self.nmk[name]
                for func, key, k in ((s.u_old, name, kind), (s.z_old, name + "_t", "velocity"),
                                     (s.a_old, name + "_tt", "acceleration")):
                    if key in v:
                        self.interpolate(func, f[k] * v[key])
                self.u[name].x.array[:] = s.u_old.x.array
            elif name in v:
                self.interpolate(self.u[name], f[kind] * v[name])

    # -- public API
    def set_initial(self, **values):
        """Initial state instead of rest, at t0. Values as in the YAML (SI) or callables
        f(x, t): each field f, and f_t, f_tt for the Newmark ones (d, d_t, d_tt, p; w, w_t, w_tt)."""
        self._set_initial({k: self.ex(v) for k, v in values.items()})

    def steps(self):
        """Advance from t0 to T. Yields the time [s] after each step."""
        tc, T0 = self.exp.time, self.exp.scales.T0
        for step in range(1, tc.nsteps + 1):
            self.t.value = (tc.t0 + step * tc.dt) / T0
            self.solver.solve(self.sol)
            for f in self.fields:
                self.u[f].x.array[:] = self.sol.x.array[self._dofs[f]]
                self.u[f].x.scatter_forward()
            if self._advance():
                self.solver.update_matrix()
            if self.reconstruction is not None:         # the conservative velocity of this step
                self.reconstruction.step()
            self.t_old.value = self.t.value
            self.time, self.step = tc.t0 + step * tc.dt, step
            yield self.time

    def run(self, recorder=None, progress=None):
        """Run to T, calling recorder(model) at t0 and after every step, then progress(model)
        after every step. Returns the recorder."""
        if recorder is not None:
            recorder(self)
        for _ in self.steps():
            if recorder is not None:
                recorder(self)
            if progress is not None:
                progress(self)
        return recorder

    def conservative_flux(self):
        """The H(div) velocity of the last step (biotx.flux), non-dimensional. The reconstruction is
        made on the first call, which must come before the first step; from then on every step
        updates it. Before any step it is zero."""
        if self.reconstruction is None:
            if self.step != 0:
                raise RuntimeError("cflux: request it before the first step (it needs every step)")
            from biotx.flux import Reconstruction      # (flux imports this module)
            self.reconstruction = Reconstruction(self)
        return self.reconstruction.u

    def quantity(self, name):
        """(Function, factor to SI) of an output quantity: d, p, flux, velocity, cflux (w in biot3)."""
        f = {"flux": self.flux, "velocity": lambda: self.sd.z_old, "cflux": self.conservative_flux}.get(name)
        func = f() if f else self.u[name]
        return func, 1.0 / self.exp.scales.factor[self.units[name]]
