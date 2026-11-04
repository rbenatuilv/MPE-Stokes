"""docs/YAML_REFERENCE.md is the list of YAML keys: every key the code accepts must appear in it."""
import dataclasses
import re
from pathlib import Path

import pytest

from biotx import conditioning, config, convergence, geometry

REFERENCE = (Path(__file__).parents[1] / "docs" / "YAML_REFERENCE.md").read_text()


def _keys():
    keys = set()
    for cls in (config.Scales, config.Material, config.Discretization, config.Time, config.Output):
        keys |= {"lambda" if f.name == "lam" else f.name for f in dataclasses.fields(cls)}
    keys |= {"shape", "names"} | set(geometry.SHAPE_KEYS) | set().union(*geometry.SHAPE_KEYS.values())
    keys |= convergence.COMMON | set().union(*convergence.KEYS.values()) | conditioning.KEYS
    return sorted(keys)


@pytest.mark.parametrize("key", _keys())
def test_key_is_documented(key):
    assert re.search(rf"`{re.escape(key)}[`:]", REFERENCE), f"`{key}` is not in docs/YAML_REFERENCE.md"
