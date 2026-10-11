"""Collects (row, col) bilinear and (row) linear block forms by field name."""
import dolfinx
import numpy as np
from petsc4py import PETSc
import ufl


class BlockSystem:
    """An n x n table of bilinear forms and n linear forms, filled by the physics modules."""

    def __init__(self, mesh, fields, test_fns, zero_measures):
        """Start with an empty table for the given fields (in this order)."""
        self.fields = list(fields)                     # e.g. ["uS", "pS"] or all four
        self.idx = {f: i for i, f in enumerate(self.fields)}
        n = len(self.fields)
        self.a = [[None] * n for _ in range(n)]
        self.L = [None] * n
        self._mesh, self._test, self._zero_dx = mesh, test_fns, zero_measures

    def add_a(self, row, col, form):
        """Add a bilinear form to block (row, col), e.g. ("uS", "pS"); sums if already set."""
        i, j = self.idx[row], self.idx[col]
        self.a[i][j] = form if self.a[i][j] is None else self.a[i][j] + form

    def add_L(self, row, form):
        """Add a linear form (right-hand side) to a row; sums if already set."""
        i = self.idx[row]
        self.L[i] = form if self.L[i] is None else self.L[i] + form

    def finalize(self):
        """Return (a, L). multiphenicsx needs a linear form in EVERY row."""
        for f, i in self.idx.items():
            if self.L[i] is None:
                v = self._test[f]
                # a dolfinx Constant, NOT a literal 0: UFL simplifies 0*v to an empty form
                zero = dolfinx.fem.Constant(self._mesh, np.zeros(v.ufl_shape, dtype=PETSc.ScalarType))
                self.L[i] = ufl.inner(zero, v) * self._zero_dx[f]
        return self.a, self.L
