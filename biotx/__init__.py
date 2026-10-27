"""biotx: two- and three-field Biot experiments, described by YAML files.

    import biotx
    exp = biotx.load("experiments/annulus_biot2.yaml", {"time.dt": 0.005})
    path = biotx.simulate(exp)             # build, run with the outputs of exp.output, save

    model = biotx.make_model(exp)          # or step by step: TwoField or ThreeField, from exp.model
    rec = model.run(biotx.Recorder(model))
    rec.save()
"""
from biotx.biot2 import TwoField
from biotx.biot3 import ThreeField
from biotx.compare import compare
from biotx.config import Experiment, load
from biotx.fields import Fields, load_fields
from biotx.geometry import build as build_geometry
from biotx.post import Points, Recorder, load_results

MODELS = {"biot2": TwoField, "biot3": ThreeField}


def make_model(exp, domain=None):
    """The model class named by exp.model, built for this experiment."""
    return MODELS[exp.model](exp, domain)


def simulate(exp, verbose=True):
    """Run an experiment with the outputs of its `output` section and save them; returns the
    results path. What `python -m biotx.run` does, for scripts:

        exp = biotx.load("experiments/annulus_biot2.yaml", {"output.fields": ["p", "cflux"]})
        path = biotx.simulate(exp)
    """
    import time
    from mpi4py import MPI
    talk = verbose and MPI.COMM_WORLD.rank == 0
    t0 = time.perf_counter()
    model = make_model(exp)
    if talk:
        ncells = model.mesh.topology.index_map(model.mesh.topology.dim).size_global
        ndofs = model.W.dofmap.index_map.size_global * model.W.dofmap.index_map_bs
        print(f"{exp.name}: {exp.model}, {ncells} cells, {ndofs} dofs, {exp.time.nsteps} steps "
              f"(setup {time.perf_counter() - t0:.1f} s)", flush=True)
    rec = Recorder(model)
    rec(model)
    every = max(1, exp.time.nsteps // 10)
    for t in model.steps():
        rec(model)
        if talk and model.step % every == 0:
            print(f"  t = {t:.4g} s  ({time.perf_counter() - t0:.1f} s)", flush=True)
    path = rec.save()
    if talk:
        print(f"saved {path}")
    return path


from biotx import conditioning, convergence  # noqa: E402  (after make_model, which they use)

__all__ = ["conditioning", "convergence", "Experiment", "load", "make_model", "simulate", "TwoField", "ThreeField", "Recorder",
           "Points", "load_results", "build_geometry", "Fields", "load_fields", "compare"]
