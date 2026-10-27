"""Conservative velocity from a Biot step (H(div) reconstruction). Post-processing: the solver is
not touched; after each step, the solution at both ends of the step is turned into a velocity whose
flow through every facet is the solver's own.

    rec = biotx.flux.Reconstruction(model)
    for t in model.steps():
        u = rec.step()          # Raviart-Thomas Function (non-dimensional; rec.to_si for SI), for the step just taken

Why. The solver conserves fluid cell by cell with its numerical flux Fhat, a single value on each
facet. The raw velocity (biot2: -kappa grad p_h; biot3: w_t) is a field on the cells: on a facet the two
neighbouring cells see different normal flows, and a transport equation fed with it sees fluid created
or destroyed (experiments/balance). H(div) is the space of fields whose divergence is square
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

where gained is the solver's mass equation without its flux part (fluid_gained below). This is the
form of the construction in Ern, Nicaise and Vohralik (2007) and Bastian and Riviere (2003).

Solve. Facet and interior unknowns decouple: the facet block is the normal-trace mass matrix (block
diagonal by facet), the interior block the moments of the RT bubbles (block diagonal by cell). Both
are constant, assembled and factorised once; each step assembles two right-hand sides. Serial only.
"""
from types import SimpleNamespace

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from petsc4py import PETSc

from biotx.common import b_P, make_ksp


class StepState:
    """Copies of the state at t^n (the model overwrites its own old state when it advances), so that
    quantities over the step t^n -> t^{n+1} can be formed after each solve. Call advance() after use."""

    def __init__(self, model):
        self.m = m = model
        const = lambda x: dolfinx.fem.Constant(m.mesh, float(x))
        self.t = const(m.t.value)
        self.th = const(m.theta.value if hasattr(m, "theta") else 1.0)
        self.three = m.exp.model == "biot3"
        self.p = dolfinx.fem.Function(m.V["p"])
        self.d = dolfinx.fem.Function(m.V["d"])
        self.z = dolfinx.fem.Function(m.V["d"])               # Newmark velocity of d
        if self.three:
            self.w = dolfinx.fem.Function(m.V["w"])
        self.advance()

    def at_old_time(self, e):
        """A data expression of t^{n+1} evaluated at t^n."""
        return ufl.replace(e, {self.m.t: self.t})

    def advance(self):
        """The state just computed becomes the old state of the next step."""
        m = self.m
        pairs = [(self.p, m.u["p"]), (self.d, m.u["d"]), (self.z, m.sd.z_old)]
        if self.three:
            pairs.append((self.w, m.u["w"]))
        for dst, src in pairs:
            dst.x.array[:] = src.x.array
        self.t.value = m.t.value
        if hasattr(m, "theta"):
            self.th.value = m.theta.value                     # the theta of the next solve


def numerical_flux(model, old):
    """The solver's wall flow over the step: (interior, [(ds, boundary)]), with `interior` the flow
    out of the '+' cell of each interior facet and `boundary` the outward flow on each boundary."""
    m, c, mm, n = model, model.c, model.m, model.m.n
    if m.exp.model == "biot3":
        Dw = (m.u["w"] - old.w) / m.dt
        return ufl.inner(ufl.avg(Dw), n("+")), [(ds, ufl.inner(Dw, n)) for ds, _ in m.bc.pressure]
    th, pen = old.th, c.zeta * c.kpen * m.k**2

    def at_level(p, at):
        interior = -c.kappa * ufl.inner(ufl.avg(ufl.grad(p)), n("+")) + pen / mm.hF * ufl.jump(p)
        bnd = [(ds, -c.kappa * ufl.inner(ufl.grad(p), n) + pen / mm.h * (p - at(gp))) for ds, gp in m.bc.pressure]
        return interior, bnd + [(ds, at(gq)) for ds, gq in m.bc.flux]

    (i1, b1), (i0, b0) = at_level(m.u["p"], lambda e: e), at_level(old.p, old.at_old_time)
    return th * i1 + (1 - th) * i0, [(ds, th * a + (1 - th) * b) for (ds, a), (_, b) in zip(b1, b0)]


def raw_velocities(model, old):
    """The velocity fields a transport equation could be given: end of step and step average."""
    m = model
    if m.exp.model == "biot3":
        return {"end": m.sw.z_old, "avg": (m.u["w"] - old.w) / m.dt}
    q1, q0 = -m.c.kappa * ufl.grad(m.u["p"]), -m.c.kappa * ufl.grad(old.p)
    return {"end": q1, "avg": old.th * q1 + (1 - old.th) * q0}


def fluid_gained(model, old, q):
    """The solver's mass equation over the step without its flux part, tested with q: the fluid
    gained (storage, solid deformation, minus sources) per unit time. gained(q) + (flux part)(q) = 0."""
    m, c, mm, n = model, model.c, model.m, model.m.n
    if m.exp.model == "biot3":
        Dd = m.u["d"] - old.d
        fp = m.source("mass_source", "none")
        G = (c.c0 / m.dt * (m.u["p"] - old.p) * q * mm.dx - (fp - old.at_old_time(fp)) / m.dt * q * mm.dx
             - c.alpha / m.dt * b_P(q, Dd, m))
        for ds, gd, _ in m.bc.displacement:
            G += c.alpha / m.dt * q * ufl.inner(gd - old.at_old_time(gd), n) * ds
        return G
    th, p = old.th, m.u["p"]
    z_th = th * m.sd.z_old + (1 - th) * old.z
    g = m.source("fluid_source", "rate")
    G = (c.c0 / m.dt * (p - old.p) * q * mm.dx + c.beta * (th * p + (1 - th) * old.p) * q * mm.dx
         - (th * g + (1 - th) * old.at_old_time(g)) * q * mm.dx - c.alpha * b_P(q, z_th, m))
    for ds, _, gd_t in m.bc.displacement:
        G += c.alpha * q * ufl.inner(th * gd_t + (1 - th) * old.at_old_time(gd_t), n) * ds
    return G


class Reconstruction:
    """Raviart-Thomas velocity of every step (see the module docstring). Call step() after each solve."""

    def __init__(self, model):
        if model.mesh.comm.size > 1:
            raise NotImplementedError("flux reconstruction: serial runs only")
        self.m = m = model
        mesh, n = m.mesh, m.m.n
        three = m.exp.model == "biot3"
        self.k = k = m.k if three else m.deg_p                 # degree of the normal traces
        if k < 1:
            raise ValueError("flux reconstruction needs normal traces of degree >= 1")
        self.V = dolfinx.fem.functionspace(mesh, basix.ufl.element("RT", mesh.basix_cell(), k + 1))
        self.u = dolfinx.fem.Function(self.V, name="flux")
        self.old = old = StepState(m)
        self.to_si = m.exp.scales.L0 / m.exp.scales.T0

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
        interior, bnd = numerical_flux(m, old)
        L_facet = interior * ufl.inner(v("+"), n("+")) * dS + sum(b * ufl.inner(v, n) * ds for ds, b in bnd)
        # cell equations: moments against [P_{k-1}]^d
        a_cell = ufl.inner(u, psi) * dx
        q_avg = raw_velocities(m, old)["avg"]
        L_cell = ufl.inner(q_avg, psi) * dx
        if not three:                                          # lifting of the symmetric SIPG term
            c, th = m.c, old.th
            p_th = th * m.u["p"] + (1 - th) * old.p
            L_cell += c.kappa * ufl.jump(p_th) * ufl.inner(ufl.avg(psi), n("+")) * dS
            for ds, gp in m.bc.pressure:
                g_th = th * gp + (1 - th) * old.at_old_time(gp)
                L_cell += c.kappa * (p_th - g_th) * ufl.inner(psi, n) * ds

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
