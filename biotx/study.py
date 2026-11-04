"""What the study tools (convergence.py, conditioning.py) share: a study YAML is an experiment
tree (usually extending a `base:` experiment) plus one section that describes the study.

    class Study(biotx.study.Study):
        section = "convergence"            # the YAML section
        check = staticmethod(_check)       # raw tree -> validated section (with defaults)

The tool then provides run(study) -> a result with table() and save(), and its CLI is
cli(description, load, run).
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mpi4py import MPI

from biotx.config import _override, from_dict, merge, read_study
from biotx.util import parse_set


@dataclass
class Study:
    """An experiment tree plus its validated study section (spec)."""
    raw: dict
    name: str
    root: Path = Path(".")
    spec: dict = field(default_factory=dict)

    section = ""                        # class attributes, set by each tool
    check = None

    def experiment(self, overrides=None):
        return from_dict(_override(self.raw, overrides or {}), self.name, self.root)

    @classmethod
    def load(cls, path, overrides=None):
        """Read a study YAML (with an optional `base:` experiment it extends)."""
        path = Path(path)
        raw = _override(read_study(path), overrides or {})
        return cls(raw, raw.get("output", {}).get("name") or path.stem, path.parent, cls.check(raw))

    @classmethod
    def from_experiment(cls, exp, **spec):
        """A study from an Experiment and the keys of its section."""
        raw = merge(exp.raw, {cls.section: spec})
        return cls(raw, exp.name, exp.root, cls.check(raw))


def save_result(raw, root, name, data, path=None):
    """np.savez(data) to `path` or <root>/<output.dir>/<name>.npz, on rank 0; returns the path."""
    if MPI.COMM_WORLD.rank != 0:
        return None
    out = raw.get("output", {}).get("dir", "results")
    path = Path(path) if path else Path(root) / out / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return path


def cli(description, load, run, argv=None, **run_options):
    """python -m biotx.<tool> <yaml> [--set KEY=VALUE ...]: run, save, print the table."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    res = run(load(args.yaml, parse_set(args.set)), **run_options)
    path = res.save()
    if MPI.COMM_WORLD.rank == 0:
        print(res.table())
        print(f"saved {path}")
    return res
