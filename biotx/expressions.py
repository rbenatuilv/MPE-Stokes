"""YAML values -> UFL expressions, in SI, of the position x and the time t.

A value can be
    a number                 2.5                      (a Constant: never simplified away)
    a list (vector)          [0, "-rho * 9.81"]
    a string                 "A * (1 - cos(2*pi*t)) / 2"
    a Python callable        lambda x, t: ...         (Python API only; x, t are UFL)

Names available in strings:
    x, y, z    position [m]            t      time [s]          r = |x|
    pi, e, sin, cos, tan, exp, log, sqrt, abs, tanh, sinh, cosh, atan, min, max
    where(cond, a, b)                  e.g. where(t < 0.1, t / 0.1, 1)
    every key of `params`, and every earlier key of `loads` (loads may build on each other)

t is a ufl.variable, so d/dt of any data is exact (ufl.diff): this is how the boundary
velocity of a prescribed displacement is obtained.
"""
import math

import dolfinx
import numpy as np
import ufl
from petsc4py import PETSc
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives

def _expand(expr):
    """Evaluate every d/dt (and d/dx) now: replacing t by t_old later then needs no Diff nodes."""
    return apply_derivatives(apply_algebra_lowering(expr))


VARIABLES = ("x", "y", "z", "t", "r")
FUNCTIONS = {
    "pi": math.pi, "e": math.e,
    "sin": ufl.sin, "cos": ufl.cos, "tan": ufl.tan, "exp": ufl.exp, "log": ufl.ln,
    "sqrt": ufl.sqrt, "abs": abs, "tanh": ufl.tanh, "sinh": ufl.sinh, "cosh": ufl.cosh,
    "atan": ufl.atan, "min": ufl.min_value, "max": ufl.max_value,
    "where": ufl.conditional,
}


class Expressions:
    """Evaluator bound to one mesh, one time Constant and one set of scales."""

    def __init__(self, mesh, scales, t_const, params=None, loads=None):
        """t_const: the non-dimensional time Constant of the forms (t* = t / T0)."""
        self.mesh = mesh
        self.x = scales.L0 * ufl.SpatialCoordinate(mesh)        # SI position
        self.t = ufl.variable(scales.T0 * t_const)              # SI time
        dim = mesh.geometry.dim
        ns = dict(FUNCTIONS)
        ns.update({"x": self.x[0], "y": self.x[1], "t": self.t,
                   "r": ufl.sqrt(ufl.inner(self.x, self.x))})
        if dim == 3:
            ns["z"] = self.x[2]
        ns.update({k: float(v) for k, v in (params or {}).items()})
        self.ns = ns
        for name, value in (loads or {}).items():               # in order: may chain
            self.ns[name] = self(value)

    def _const(self, value):
        return dolfinx.fem.Constant(self.mesh, PETSc.ScalarType(value))

    def __call__(self, value):
        """UFL expression (SI) of a YAML value or callable."""
        if callable(value) and not isinstance(value, ufl.core.expr.Expr):
            return _expand(ufl.as_ufl(value(self.x, self.t)))
        if isinstance(value, (int, float)):
            return self._const(float(value))
        if isinstance(value, (list, tuple)):
            if all(isinstance(v, (int, float)) for v in value):
                return self._const(np.asarray(value, dtype=float))
            return ufl.as_vector([self(v) for v in value])
        if isinstance(value, str):
            out = eval(value, {"__builtins__": {}}, self.ns)
            return self(out) if isinstance(out, (int, float)) else ufl.as_ufl(out)
        if isinstance(value, ufl.core.expr.Expr):
            return value
        raise TypeError(f"cannot turn {value!r} into an expression")

    def ddt(self, expr):
        """Exact time derivative d/dt (SI) of an expression; a zero Constant if it is constant."""
        out = _expand(ufl.diff(expr, self.t))
        if isinstance(out, ufl.constantvalue.Zero):
            return self._const(np.zeros(expr.ufl_shape))
        return out
