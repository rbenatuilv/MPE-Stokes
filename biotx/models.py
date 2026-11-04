"""The models, by the name used in the YAML (`model:`). Defining a BiotModel subclass with a
`name` registers it (common.REGISTRY); importing this module registers the built-in ones."""
from biotx.biot2 import TwoField
from biotx.biot3 import ThreeField
from biotx.common import REGISTRY

__all__ = ["REGISTRY", "TwoField", "ThreeField", "make_model"]


def make_model(exp, domain=None):
    """The model class named by exp.model, built for this experiment."""
    return REGISTRY[exp.model](exp, domain)
