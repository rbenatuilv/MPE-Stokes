"""Run one experiment from the command line.

    python -m biotx.run experiments/pulsatile.yaml
    python -m biotx.run experiments/pulsatile.yaml --set model=biot3 --set discretization.deg=2
    python -m biotx.run experiments/pulsatile.yaml --set geometry.h=0.00025 --name pulsatile_ref

--set takes dotted keys of the YAML tree; values are parsed as YAML (numbers, lists, strings).
"""
import argparse
import time

import yaml
from mpi4py import MPI

import biotx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--name", help="output file stem (default: output.name or the YAML stem)")
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    if args.name:
        overrides["output.name"] = args.name
    exp = biotx.load(args.yaml, overrides)

    rank0 = MPI.COMM_WORLD.rank == 0
    t0 = time.perf_counter()
    model = biotx.make_model(exp)
    if rank0:
        ncells = model.mesh.topology.index_map(model.mesh.topology.dim).size_global
        print(f"{exp.name}: {exp.model}, {ncells} cells, {model.W.dofmap.index_map.size_global * model.W.dofmap.index_map_bs} dofs, "
              f"{exp.time.nsteps} steps (setup {time.perf_counter() - t0:.1f} s)")
    rec = biotx.Recorder(model)
    rec(model)
    every = max(1, exp.time.nsteps // 10)
    for t in model.steps():
        rec(model)
        if rank0 and model.step % every == 0:
            print(f"  t = {t:.4g} s  ({time.perf_counter() - t0:.1f} s)", flush=True)
    path = rec.save()
    if rank0:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
