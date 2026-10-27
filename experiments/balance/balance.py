"""Fake fluid of a velocity field: cell-by-cell fluid bookkeeping (item 1, Step 1).

    python balance.py ../annulus_biot2.yaml [--set key=value] [--name NAME]   # -> results/<name>.npz

The idea. In every cell K, over every time step, the fluid the cell gained must equal what came in
through its walls minus what went out. Per unit time:

    gained_K + wall_out_K = 0

The solver satisfies this exactly, with its own wall flows (the numerical flux). The transport
equation would instead get a velocity FIELD q, and with the wall flows of q the books do not quite
balance. What is left over is fluid that q invents or destroys in K:

    fake_K(q) = gained_K + wall_out_K(q) = wall_out_K(q) - wall_out_K(solver)

    fake_K > 0: q takes out of K fluid that is not there;  fake_K < 0: q brings in fluid from nowhere.

In the transport equation dc/dt + div(q c) - mu Lap c = f, fake_K acts as a fake solute source.

Velocities tested (what a transport equation would receive):
    end   end-of-step velocity    biot2: -kappa grad p^{n+1};   biot3: w_t^{n+1} (Morasca's choice)
    avg   step-average velocity   biot2: th q^{n+1} + (1 - th) q^n;   biot3: (w^{n+1} - w^n) / dt
    rec   reconstructed velocity  biot2, biot3: the H(div) field of biotx.flux, whose wall flows are the
                                  solver's (expected: fake fluid at round-off)
The fluid moved during a step is the step average times dt, so `avg` is right in time; `end` adds an
error in time. Both are the raw field in space: on a wall, the value from each side differs.

Where the terms come from: biotx.flux (numerical_flux, fluid_gained, raw_velocities; the formulas are
in its docstring). Each model's fluid mass equation holds for every pressure test function; the
pressures are discontinuous, so "1 on cell K, 0 elsewhere" is one of them, and testing with it gives
the cell's books exactly. The script checks gained + wall flow (solver) = 0 at every step.

Saved, in SI (m^2/s per unit depth for the flows, 1/s for rates):
    per step     for each velocity: total = sum|fake| / sum throughput, worst = max_K |fake_K| / throughput_K,
                 drift80 = 80 s * max_K |fake_K| / |K| (how much a unit concentration would drift in the
                 worst cell over 80 s if nothing else moved it); appendix: global = sum fake, jump =
                 ||[[q.n]]|| / ||{q}.n|| on interior walls; solver_check = max |gained + wall (solver)| / max throughput
    maps         gained, wall_solver, wall_end, wall_avg, wall_rec, throughput per cell at the snapshot times
    period       the same six, averaged over the last period (the averaged velocity Morasca used)
throughput_K = int_{dK} |Fhat|, the fluid flowing through K.
"""
import argparse
from pathlib import Path

import dolfinx
import numpy as np
import ufl
import yaml

import biotx
from biotx import flux

VELOCITIES = ("end", "avg", "rec")
TERMS = ("gained", "wall_solver", "wall_end", "wall_avg", "wall_rec", "throughput")


def _assemble(form, nloc):
    vec = dolfinx.fem.assemble_vector(form)
    vec.scatter_reverse(dolfinx.la.InsertMode.add)
    return vec.array[:nloc].copy()


class CellBooks:
    """The bookkeeping terms of every cell, for one model, as DG0 vectors (one entry per cell). The
    solver's wall flow, the fluid gained and the raw velocities come from biotx.flux, which also
    builds the reconstructed velocity; all share one copy of the state at t^n."""

    def __init__(self, model):
        self.m = m = model
        self.rec = flux.Reconstruction(m)
        self.old_ = old = self.rec.old
        self.v = v = ufl.TestFunction(dolfinx.fem.functionspace(m.mesh, ("DG", 0)))
        self.nloc = m.mesh.topology.index_map(m.mesh.topology.dim).size_local
        wall_in, wall_bnd = flux.numerical_flux(m, old)
        velocities = {**flux.raw_velocities(m, old), "rec": self.rec.u}
        dS, n = m.m.dS, m.m.n
        out = lambda q: (ufl.inner(q("+"), n("+")) * v("+") + ufl.inner(q("-"), n("-")) * v("-")) * dS \
            + ufl.inner(q, n) * v * m.m.ds                   # q.n out of each cell, from inside it
        forms = {"gained": flux.fluid_gained(m, old, v),
                 "wall_solver": wall_in * ufl.jump(v) * dS + sum(b * v * ds for ds, b in wall_bnd),
                 "throughput": abs(wall_in) * (v("+") + v("-")) * dS + sum(abs(b) * v * ds for ds, b in wall_bnd),
                 **{f"wall_{k}": out(q) for k, q in velocities.items()}}
        self.forms = {k: dolfinx.fem.form(f) for k, f in forms.items()}
        self.volume = _assemble(dolfinx.fem.form(v * m.m.dx), self.nloc)
        self.jumps = {k: (dolfinx.fem.form(ufl.jump(q, n) ** 2 * dS),
                          dolfinx.fem.form(ufl.inner(ufl.avg(q), n("+")) ** 2 * dS)) for k, q in velocities.items()}

    def terms(self):
        """The per-cell terms of the step just taken (non-dimensional)."""
        self.rec.compute()
        return {k: _assemble(f, self.nloc) for k, f in self.forms.items()}

    def jump(self, k):
        num, den = (dolfinx.fem.assemble_scalar(f) for f in self.jumps[k])
        return float(np.sqrt(num / den)) if den > 0 else 0.0

    def next_step(self):
        """The state just computed becomes the old state of the next step."""
        self.old_.advance()


def run(path, overrides=None, name=None, period=1.0):
    exp = biotx.load(path, overrides or {})                  # no Recorder: output.* only gives the map times
    model = biotx.make_model(exp)
    books = CellBooks(model)
    s, dim = exp.scales, model.mesh.geometry.dim
    flow_si = s.L0**dim / s.T0                               # a flow through a cell: volume per time
    volume = books.volume * s.L0**dim
    snaps, T, dt = exp.output.times(exp.time), exp.time.T, exp.time.dt

    per_step = {"t": [], "solver_check": []}
    for k in VELOCITIES:
        for key in ("total", "worst", "drift80", "global", "jump"):
            per_step[f"{k}_{key}"] = []
    maps, period_sum, nperiod = {"t": []}, {k: 0.0 for k in TERMS}, 0

    for t in model.steps():
        b = {k: v * flow_si for k, v in books.terms().items()}
        tp = b["throughput"]
        busy = tp > 1e-12 * tp.max()                         # cells with fluid flowing through them
        per_step["t"].append(t)
        per_step["solver_check"].append(np.abs(b["gained"] + b["wall_solver"]).max() / tp.max())
        for k in VELOCITIES:
            fake = b[f"wall_{k}"] - b["wall_solver"]
            per_step[f"{k}_total"].append(np.abs(fake).sum() / tp.sum())
            per_step[f"{k}_worst"].append((np.abs(fake[busy]) / tp[busy]).max())
            per_step[f"{k}_drift80"].append(80.0 * (np.abs(fake) / volume).max())
            per_step[f"{k}_global"].append(fake.sum())
            per_step[f"{k}_jump"].append(books.jump(k))
        if any(abs(t - ts) < dt / 2 for ts in snaps):
            maps["t"].append(t)
            for k in TERMS:
                maps.setdefault(k, []).append(b[k])
        if t > T - period + dt / 2:
            nperiod += 1
            for k in TERMS:
                period_sum[k] = period_sum[k] + b[k]
        books.next_step()

    mesh = model.mesh
    out = {**{k: np.array(v) for k, v in per_step.items()},
           **{f"map_{k}": np.array(v) for k, v in maps.items()},
           **{f"period_{k}": v / max(nperiod, 1) for k, v in period_sum.items()},
           "x": mesh.geometry.x[:, :dim] * s.L0, "cells": mesh.geometry.dofmaps[0][:books.nloc],
           "volume": volume, "model": exp.model, "dt": dt, "period": period,
           "experiment": yaml.safe_dump(exp.raw, sort_keys=False)}
    path = Path(__file__).parent / "results" / f"{name or exp.name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--name")
    a = ap.parse_args(argv)
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    print("saved", run(a.yaml, overrides, a.name))


if __name__ == "__main__":
    main()
