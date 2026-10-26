"""Manufactured solutions: the exact fields solve the discrete problem up to the
discretisation error, so the errors must fall at the theoretical rates. Exercises every
boundary condition kind, sources, nonzero Dirichlet data (and its velocity) and
model.set_initial; the Python API takes callables f(x, t) wherever the YAML takes strings."""
import dolfinx
import numpy as np
import pytest
import ufl

from biotx import ThreeField, TwoField
from biotx.config import from_dict

pi = np.pi
NORMALS = {"left": (-1, 0), "right": (1, 0), "bottom": (0, -1), "top": (0, 1)}


def tf(t):
    return 1 + ufl.sin(t)


def d_ex(x, t):
    return tf(t) * ufl.as_vector([ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]),
                                  ufl.sin(pi * x[1]) * ufl.cos(pi * x[0])])


def w_ex(x, t):
    return tf(t) * ufl.as_vector([ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) + 0.2,
                                  ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]) + 0.2])


def p_ex(x, t):
    return tf(t) * (ufl.cos(pi * x[0]) * ufl.cos(pi * x[1]) + 0.5)


def sig_tot(x, t):          # all coefficients 1
    d = d_ex(x, t)
    return 2 * ufl.sym(ufl.grad(d)) + ufl.div(d) * ufl.Identity(2) - p_ex(x, t) * ufl.Identity(2)


def ddt(f, n=1):
    def g(x, t):
        e = f(x, t)
        for _ in range(n):
            e = ufl.diff(e, t)
        return e
    return g


def traction(side):
    return lambda x, t: sig_tot(x, t) * ufl.as_vector(NORMALS[side])


def outflow(side):          # q.n = -kappa grad p . n
    return lambda x, t: -ufl.inner(ufl.grad(p_ex(x, t)), ufl.as_vector(NORMALS[side]))


MATERIAL = {"lambda": 1.0, "mu": 1.0, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0,
            "rho": 1.0, "rho_f": 1.0, "phi": 0.5, "tortuosity": 1.0}      # rhoW = 2


def experiment(model, N, deg, boundaries, sources, T=0.1, dt=0.003125):
    return from_dict({"model": model, "material": MATERIAL,
                      "geometry": {"shape": "rectangle", "corners": [[0, 0], [1, 1]], "n": [N, N]},
                      "boundaries": boundaries, "sources": sources,
                      "discretization": {"deg": deg},
                      "time": {"T": T, "dt": dt, "theta": 0.5}}, f"mms_{model}")


def errors(model, exact):
    dx = ufl.Measure("dx", domain=model.mesh, metadata={"quadrature_degree": model.k + 4})
    out = {}
    for name, f in exact.items():
        e = model.u[name] - f(model.ex.x, model.ex.t)
        local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(e, e) * dx))
        out[name] = np.sqrt(model.mesh.comm.allreduce(local))
    return out


def rates(errs):
    e = np.array([list(r.values()) for r in errs])
    return dict(zip(errs[0], np.log2(e[-2] / e[-1])))


def run2(N, deg=1):
    f = lambda x, t: ddt(d_ex, 2)(x, t) - ufl.div(sig_tot(x, t))
    g = lambda x, t: (ddt(p_ex)(x, t) + ufl.div(ddt(d_ex)(x, t))
                      - ufl.div(ufl.grad(p_ex(x, t))))
    bnd = {"left":   {"displacement": d_ex, "pressure": p_ex},
           "bottom": {"displacement": d_ex, "flux": outflow("bottom")},
           "right":  {"traction": traction("right"), "flux": outflow("right")},
           "top":    {"traction": traction("top"), "pressure": p_ex}}
    m = TwoField(experiment("biot2", N, deg, bnd, {"body_force": f, "fluid_source": g}))
    m.set_initial(d=d_ex, d_t=ddt(d_ex), d_tt=ddt(d_ex, 2), p=p_ex)
    m.run()
    return errors(m, {"d": d_ex, "p": p_ex})


def run3(N, deg=2):
    f_d = lambda x, t: ddt(d_ex, 2)(x, t) + ddt(w_ex, 2)(x, t) - ufl.div(sig_tot(x, t))
    f_w = lambda x, t: (ddt(d_ex, 2)(x, t) + 2 * ddt(w_ex, 2)(x, t) + ddt(w_ex)(x, t)
                        + ufl.grad(p_ex(x, t)))
    f_p = lambda x, t: p_ex(x, t) + ufl.div(d_ex(x, t)) + ufl.div(w_ex(x, t))
    bnd = {"left":   {"displacement": d_ex, "pressure": p_ex},
           "bottom": {"displacement": d_ex, "pressure": p_ex},
           "right":  {"traction": traction("right"), "pressure": p_ex},
           "top":    {"traction": traction("top"), "pressure": p_ex}}
    m = ThreeField(experiment("biot3", N, deg, bnd,
                              {"body_force": f_d, "fluid_force": f_w, "mass_source": f_p}))
    m.set_initial(d=d_ex, d_t=ddt(d_ex), d_tt=ddt(d_ex, 2), w=w_ex, w_t=ddt(w_ex), w_tt=ddt(w_ex, 2))
    m.run()
    return errors(m, {"d": d_ex, "w": w_ex, "p": p_ex})


def test_biot2_p1_space_rates():
    """P1/P1, all four boundary condition kinds: second order in L2 for d and p."""
    r = rates([run2(N) for N in (8, 16, 32)])   # N = 16 is still pre-asymptotic for p
    assert r["d"] > 1.8 and r["p"] > 1.8, r


@pytest.mark.slow
def test_biot3_p2p2p1_space_rates():
    """P2/P2/P1: d and w third order in L2, p second order."""
    r = rates([run3(N) for N in (4, 8, 16)])
    assert r["d"] > 2.7 and r["w"] > 2.7 and r["p"] > 1.6, r
