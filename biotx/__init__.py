"""biotx: two- and three-field Biot experiments, described by YAML files.

    import biotx
    exp = biotx.load("experiments/pulsatile.yaml", {"time.dt": 0.005})
    model = biotx.make_model(exp)          # TwoField or ThreeField, from exp.model
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


__all__ = ["Experiment", "load", "make_model", "TwoField", "ThreeField", "Recorder",
           "Points", "load_results", "build_geometry", "Fields", "load_fields", "compare"]
