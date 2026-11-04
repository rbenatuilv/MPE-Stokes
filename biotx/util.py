"""Small helpers shared by the models, the outputs and the analysis tools."""
import re

import dolfinx
import numpy as np
import yaml
from mpi4py import MPI


def integrate(form, comm) -> float:
    """The value of a compiled scalar form, summed over the ranks."""
    return comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


def ndofs(model) -> int:
    """Global number of unknowns of the model's mixed space."""
    return model.W.dofmap.index_map.size_global * model.W.dofmap.index_map_bs


class _Loader(yaml.SafeLoader):
    """PyYAML follows YAML 1.1, where a float needs a dot: 1e-3 would be the string '1e-3'.
    This loader also reads exponent-only numbers (1e-3, 2E5, as in YAML 1.2) as floats."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."))


def yaml_load(text):
    """yaml.safe_load, with 1e-3 read as a number (see _Loader)."""
    return yaml.load(text, Loader=_Loader)


def parse_set(items) -> dict:
    """--set KEY=VALUE arguments as dotted-key overrides; values are parsed as YAML."""
    return {k: yaml_load(v) for k, v in (s.split("=", 1) for s in items)}


def format_table(rows, title=None, align="right") -> str:
    """Rows of strings (the first is the header) in padded columns; under a title the rows are
    indented by two spaces."""
    w = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    pad = str.rjust if align == "right" else str.ljust
    lines = ["  ".join(pad(c, wj) for c, wj in zip(r, w)) for r in rows]
    if title is None:
        return "\n".join(lines)
    return "\n".join([title] + ["  " + line for line in lines])


def load_npz(path) -> dict:
    """The arrays of a saved .npz, with 'experiment' (the YAML tree) parsed back into a dict."""
    with np.load(path, allow_pickle=True) as f:
        out = {k: f[k] for k in f.files}
    out["experiment"] = yaml_load(str(out["experiment"]))
    return out
