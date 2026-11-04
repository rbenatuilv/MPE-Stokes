"""biotx: two- and three-field Biot experiments, described by YAML files.

    import biotx
    exp = biotx.load("applications/brain_annulus/annulus_biot2.yaml", {"time.dt": 0.005})
    path = biotx.simulate(exp)             # build, run with the outputs of exp.output, save

    model = biotx.make_model(exp)          # or step by step: TwoField or ThreeField, from exp.model
    rec = model.run(biotx.Recorder(model))
    rec.save()
"""
import importlib

from biotx.compare import compare
from biotx.config import Experiment, load
from biotx.fields import Fields, load_fields
from biotx.geometry import build as build_geometry
from biotx.models import REGISTRY as MODELS
from biotx.models import ThreeField, TwoField, make_model
from biotx.post import Points, Recorder, load_results
from biotx.util import ndofs


def simulate(exp, verbose=True):
    """Run an experiment with the outputs of its `output` section and save them; returns the
    results path. What `python -m biotx.run` does, for scripts:

        exp = biotx.load("applications/brain_annulus/annulus_biot2.yaml", {"output.fields": ["p", "cflux"]})
        path = biotx.simulate(exp)
    """
    import time
    from mpi4py import MPI
    talk = verbose and MPI.COMM_WORLD.rank == 0
    t0 = time.perf_counter()
    model = make_model(exp)
    if talk:
        ncells = model.mesh.topology.index_map(model.mesh.topology.dim).size_global
        print(f"{exp.name}: {exp.model}, {ncells} cells, {ndofs(model)} dofs, {exp.time.nsteps} steps "
              f"(setup {time.perf_counter() - t0:.1f} s)", flush=True)
    every = max(1, exp.time.nsteps // 10)

    def progress(m):
        if m.step % every == 0:
            print(f"  t = {m.time:.4g} s  ({time.perf_counter() - t0:.1f} s)", flush=True)

    path = model.run(Recorder(model), progress if talk else None).save()
    if talk:
        print(f"saved {path}")
    return path


def __getattr__(name):
    """biotx.conditioning and biotx.convergence, imported on first use (SLEPc, study code)."""
    if name in ("conditioning", "convergence"):
        return importlib.import_module(f"biotx.{name}")
    raise AttributeError(f"module 'biotx' has no attribute {name!r}")

__all__ = ["conditioning", "convergence", "Experiment", "load", "make_model", "simulate", "TwoField", "ThreeField", "Recorder",
           "Points", "load_results", "build_geometry", "Fields", "load_fields", "compare"]
