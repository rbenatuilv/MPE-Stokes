"""Conservative velocity from a Biot step (H(div) reconstruction). Post-processing: the solver is
not touched; after each step, the solution at both ends of the step is turned into a velocity whose
flow through every facet is the solver's own.

    rec = biotx.flux.Reconstruction(model)
    for t in model.steps():
        u = rec.step()          # Raviart-Thomas Function (non-dimensional), for the step just taken

CellBalance(model) (end of file) gives the fluid books of every cell with each velocity: how much fluid
a raw velocity invents, and that the reconstructed one does not (verification/balance).

Why. The solver conserves fluid cell by cell with its numerical flux Fhat, a single value on each
facet. The raw velocity (biot2: -kappa grad p_h; biot3: w_t) is a field on the cells: on a facet the two
neighbouring cells see different normal flows, and a transport equation fed with it sees fluid created
or destroyed (verification/balance). H(div) is the space of fields whose divergence is square
integrable; for piecewise polynomials it means that the normal component is continuous across facets.
(It does not mean divergence-free: div u is the rate at which the cells release fluid.) Its standard
finite element is Raviart-Thomas, whose unknowns are the normal flows on the facets and moments inside
the cells.

The reconstruction. u in RT with normal traces of degree k (biot2: k = deg_p; biot3: k = deg) such that

    every facet F    int_F (u.n) mu = int_F Fhat mu                  for mu in P_k(F)
    every cell K     int_K u . psi = int_K q_avg . psi + L(psi)       for psi in [P_{k-1}(K)]^d

with the step-consistent quantities (th: theta of the step)

    biot2   Fhat = th Fhat(p^{n+1}) + (1 - th) Fhat(p^n),   Fhat(p) out of K:
                     interior    -kappa {grad p}.n + pen / hF (p_K - p_N)
                     pressure    -kappa grad p.n + pen / h (p - g_p)
                     flux        g_q
            q_avg = -kappa grad p_th,    L(psi) = sum_F int_F omega kappa (psi.n) [[p_th]]
                    (omega = 1/2 inside, 1 on pressure boundaries, [[p]] = p_K - p_N or p - g_p)
    biot3   Fhat = {Dw / dt}.n inside, (Dw / dt).n on pressure boundaries, 0 on flux boundaries;
            q_avg = Dw / dt,  L = 0                                      (Dw = w^{n+1} - w^n)

The facet equations make every cell's balance exact. The cell equations, with the lifting L of the
symmetric SIPG term, make it exact against every pressure test function, not only on averages:

    (div u, q)_K = -gained(q)      for every q in P_{deg_p}(K)   (biot2: deg_p = k; biot3: deg_p = k - 1)

where gained is the solver's mass equation without its flux part (fluid_gained). This is the
form of the construction in Ern, Nicaise and Vohralik (2007) and Bastian and Riviere (2003).

Each model states its own quantities, next to its residual: trace_degree() (k), numerical_flux(old)
(Fhat), step_velocities(old) (q_avg and the end-of-step velocity), fluid_gained(old, q) and
flux_lifting(old, psi, dS) (L). This module only solves for u, for any model that provides them.

Solve. Facet and interior unknowns decouple: the facet block is the normal-trace mass matrix (block
diagonal by facet), the interior block the moments of the RT bubbles (block diagonal by cell). Both
are constant, assembled and factorised once; each step assembles two right-hand sides. Serial only.
"""
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from petsc4py import PETSc

from biotx.common import make_ksp


class StepState:
    """Copies of the state at t^n (the model overwrites its own old state when it advances), so that
    quantities over the step t^n -> t^{n+1} can be formed after each solve: u[f] for every field,
    z[f] the velocity of every Newmark field, t and th (the theta of the step). Call advance() after use."""

    def __init__(self, model):
        self.m = m = model
        const = lambda x: dolfinx.fem.Constant(m.mesh, float(x))
        self.t, self.th = const(m.t.value), const(m.theta.value)
        self.u = {f: dolfinx.fem.Function(m.V[f]) for f in m.fields}
        self.z = {f: dolfinx.fem.Function(m.V[f]) for f in m.newmark}
        self.advance()

    def at_old_time(self, e):
        """A data expression of t^{n+1} evaluated at t^n."""
        return ufl.replace(e, {self.m.t: self.t})

    def advance(self):
        """The state just computed becomes the old state of the next step."""
        m = self.m
        pairs = [(self.u[f], m.u[f]) for f in self.u] + [(self.z[f], m.nmk[f].z_old) for f in self.z]
        for dst, src in pairs:
            dst.x.array[:] = src.x.array
        self.t.value = m.t.value
        self.th.value = m.theta.value                         # the theta of the next solve


# The model's own step-consistent mass balance (the methods of the same name on each model).
def numerical_flux(model, old):
    """The solver's wall flow over the step: (interior, [(ds, boundary)]), with `interior` the flow
    out of the '+' cell of each interior facet and `boundary` the outward flow on each boundary."""
    return model.numerical_flux(old)


def raw_velocities(model, old):
    """The velocity fields a transport equation could be given: end of step and step average."""
    return model.step_velocities(old)


def fluid_gained(model, old, q):
    """The solver's mass equation over the step without its flux part, tested with q: the fluid
    gained (storage, solid deformation, minus sources) per unit time. gained(q) + (flux part)(q) = 0."""
    return model.fluid_gained(old, q)


class Reconstruction:
    """Raviart-Thomas velocity of every step (see the module docstring). Call step() after each solve."""

    def __init__(self, model):
        if model.mesh.comm.size > 1:
            raise NotImplementedError("flux reconstruction: serial runs only")
        self.m = m = model
        mesh, n = m.mesh, m.m.n
        self.k = k = m.trace_degree()                         # degree of the normal traces
        if k < 1:
            raise ValueError("flux reconstruction needs normal traces of degree >= 1")
        self.V = dolfinx.fem.functionspace(mesh, basix.ufl.element("RT", mesh.basix_cell(), k + 1))
        self.u = dolfinx.fem.Function(self.V, name="flux")
        self.old = old = StepState(m)

        md = {"quadrature_degree": 2 * k + 2}
        dS = ufl.Measure("dS", domain=mesh, metadata=md)
        ds_all = ufl.Measure("ds", domain=mesh, metadata=md)
        dx = ufl.Measure("dx", domain=mesh, metadata=md)
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        W = dolfinx.fem.functionspace(mesh, ("DG", k - 1, (mesh.geometry.dim,)))
        psi = ufl.TestFunction(W)

        # facet equations: normal-trace mass (bubbles have no normal trace: their rows are zero)
        a_facet = (ufl.inner(u("+"), n("+")) * ufl.inner(v("+"), n("+")) * dS
                   + ufl.inner(u, n) * ufl.inner(v, n) * ds_all)
        interior, bnd = m.numerical_flux(old)
        L_facet = interior * ufl.inner(v("+"), n("+")) * dS + sum(b * ufl.inner(v, n) * ds for ds, b in bnd)
        # cell equations: moments against [P_{k-1}]^d
        a_cell = ufl.inner(u, psi) * dx
        L_cell = ufl.inner(m.step_velocities(old)["avg"], psi) * dx
        lifting = m.flux_lifting(old, psi, dS)                 # biot2: of the symmetric SIPG term
        if lifting is not None:
            L_cell += lifting

        # facet and interior unknowns of RT
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        local_int = np.asarray(self.V.dofmap.dof_layout.entity_dofs(tdim, 0), dtype=np.int32)
        cell_dofs = self.V.dofmap.list[:ncells]
        interior_dofs = np.unique(cell_dofs[:, local_int].ravel()).astype(np.int32)
        ndofs = self.V.dofmap.index_map.size_local
        facet_dofs = np.setdiff1d(np.arange(ndofs, dtype=np.int32), interior_dofs)
        self.is_f = PETSc.IS().createGeneral(facet_dofs)
        self.is_b = PETSc.IS().createGeneral(interior_dofs)
        # rows of the cell equations: all [P_{k-1}]^d moments
        self.is_rows = PETSc.IS().createGeneral(np.arange(W.dofmap.index_map.size_local * W.dofmap.index_map_bs,
                                                          dtype=np.int32))

        A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_facet))
        A.assemble()
        B = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_cell))
        B.assemble()
        self.A_ff = A.createSubMatrix(self.is_f, self.is_f)
        self.B_rb = B.createSubMatrix(self.is_rows, self.is_b)
        self.B_rf = B.createSubMatrix(self.is_rows, self.is_f)
        self.ksp_f = make_ksp(self.A_ff)
        self.ksp_b = make_ksp(self.B_rb)
        self.L_facet, self.L_cell = dolfinx.fem.form(L_facet), dolfinx.fem.form(L_cell)
        self.uf, self.ub, self.bf = self.A_ff.createVecRight(), self.B_rb.createVecRight(), self.A_ff.createVecLeft()
        self.rhs_b = self.B_rb.createVecLeft()

    def step(self):
        """The conservative velocity of the step just taken; then the step state advances."""
        u = self.compute()
        self.old.advance()
        return u

    def compute(self):
        """The conservative velocity of the step just taken, without advancing (self.old still holds
        t^n: quantities over the step can be formed before calling self.old.advance())."""
        bf = dolfinx.fem.petsc.assemble_vector(self.L_facet)
        bf.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bc = dolfinx.fem.petsc.assemble_vector(self.L_cell)
        bc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        f_idx, b_idx = self.is_f.getIndices(), self.is_b.getIndices()
        self.bf.setArray(bf.getArray(readonly=True)[f_idx])
        self.ksp_f.solve(self.bf, self.uf)
        self.B_rf.mult(self.uf, self.rhs_b)                    # move the facet part to the right-hand side
        self.rhs_b.aypx(-1.0, bc)
        self.ksp_b.solve(self.rhs_b, self.ub)
        x = self.u.x.array
        x[f_idx] = self.uf.getArray(readonly=True)
        x[b_idx] = self.ub.getArray(readonly=True)
        self.u.x.scatter_forward()
        return self.u


class CellBalance:
    """The fluid books of every cell over each step, as DG0 vectors (one entry per owned cell,
    non-dimensional). Over a step, per unit time, cell K gains what comes in through its walls:

        gained_K + wall_solver_K = 0           exactly, with the solver's wall flow (numerical_flux)

    A velocity field q has wall flows wall_q_K = int_{dK} q.n of its own; fake_K = wall_q_K -
    wall_solver_K is fluid q invents (> 0: it takes out fluid that is not there). terms() gives
        gained, wall_solver, wall_end, wall_avg, wall_rec, throughput
    for the raw velocities of the step (model.step_velocities: end of step and step average) and
    the reconstructed one (rec); throughput_K = int_{dK} |Fhat|. jump(k) is ||[[q.n]]|| / ||{q}.n||
    over the interior facets. All share the reconstruction's copy of the state at t^n:

        books = CellBalance(model)
        for t in model.steps():
            b = books.terms()
            books.advance()
    """

    def __init__(self, model):
        self.m = m = model
        self.rec = Reconstruction(m)
        self.old = old = self.rec.old
        v = ufl.TestFunction(dolfinx.fem.functionspace(m.mesh, ("DG", 0)))
        self.nloc = m.mesh.topology.index_map(m.mesh.topology.dim).size_local
        wall_in, wall_bnd = m.numerical_flux(old)
        self.velocities = {**m.step_velocities(old), "rec": self.rec.u}
        dS, n = m.m.dS, m.m.n
        out = lambda q: (ufl.inner(q("+"), n("+")) * v("+") + ufl.inner(q("-"), n("-")) * v("-")) * dS \
            + ufl.inner(q, n) * v * m.m.ds                   # q.n out of each cell, from inside it
        forms = {"gained": m.fluid_gained(old, v),
                 "wall_solver": wall_in * ufl.jump(v) * dS + sum(b * v * ds for ds, b in wall_bnd),
                 "throughput": abs(wall_in) * (v("+") + v("-")) * dS + sum(abs(b) * v * ds for ds, b in wall_bnd),
                 **{f"wall_{k}": out(q) for k, q in self.velocities.items()}}
        self.forms = {k: dolfinx.fem.form(f) for k, f in forms.items()}
        self.volume = self._assemble(dolfinx.fem.form(v * m.m.dx))
        self.jumps = {k: (dolfinx.fem.form(ufl.jump(q, n) ** 2 * dS),
                          dolfinx.fem.form(ufl.inner(ufl.avg(q), n("+")) ** 2 * dS))
                      for k, q in self.velocities.items()}

    def _assemble(self, form):
        vec = dolfinx.fem.assemble_vector(form)
        vec.scatter_reverse(dolfinx.la.InsertMode.add)
        return vec.array[:self.nloc].copy()

    def terms(self) -> dict:
        """The per-cell terms of the step just taken (computes the reconstruction, does not advance)."""
        self.rec.compute()
        return {k: self._assemble(f) for k, f in self.forms.items()}

    def jump(self, k) -> float:
        num, den = (dolfinx.fem.assemble_scalar(f) for f in self.jumps[k])
        return float(np.sqrt(num / den)) if den > 0 else 0.0

    def advance(self):
        """The state just computed becomes the old state of the next step."""
        self.old.advance()
