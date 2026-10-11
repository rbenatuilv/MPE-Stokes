"""Assemble the block matrix once; solve once per time step."""
import dolfinx
import multiphenicsx.fem.petsc as mpx
from petsc4py import PETSc

MUMPS = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
         "mat_mumps_icntl_14": 80}          # extra MUMPS workspace (avoids error -9 in 3D)


class BlockSolver:
    """Restricted block system whose matrix is assembled once and solved many times."""

    def __init__(self, a, L, restrictions, options=MUMPS, prefix="spx_"):
        """Compile the forms, assemble the matrix A, create the vectors and configure the KSP."""
        self.a, self.L, self.r = dolfinx.fem.form(a), dolfinx.fem.form(L), restrictions
        self.A = mpx.assemble_matrix(self.a, bcs=[], kind="mpi", restriction=(self.r, self.r))
        self.A.assemble()
        spaces = dolfinx.fem.extract_function_spaces(self.L)
        self.b = mpx.create_vector(spaces, kind="mpi", restriction=self.r)
        self.x = mpx.create_vector(spaces, kind="mpi", restriction=self.r)
        self.ksp = PETSc.KSP().create(self.A.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setOptionsPrefix(prefix)
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in options.items():
            opts[key] = value
        opts.prefixPop()
        self.ksp.setFromOptions()

    def solve_into(self, funcs):
        """Reassemble only the right-hand side, solve, and copy the solution into `funcs`
        ([uS, pS, d, pP] Functions). Returns the number of iterations."""
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        mpx.assemble_vector(self.b, self.L, restriction=self.r)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.ksp.solve(self.b, self.x)
        if self.ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"linear solve failed, reason {self.ksp.getConvergedReason()}")
        self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        mpx.assign(self.x, funcs, self.r)            # vector -> [uS, pS, d, pP]
        for f in funcs:
            f.x.scatter_forward()
        return self.ksp.getIterationNumber()
