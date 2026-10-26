"""Experiment description: one YAML file -> one frozen `Experiment`.

Everything in the YAML is in SI units (m, s, Pa, kg/m^3, m^2, Pa s). The `scales` section
(L0 [m], T0 [s], P0 [Pa]) defines the non-dimensional problem that is actually solved:

    x* = x / L0,   t* = t / T0,   p* = p / P0,   d* = d / L0,   q* = q T0 / L0.

The conversion is done once, in `Scales` and `Material.scaled`; the models never see SI
values. Scales of 1 (the default) mean "solve with the numbers as given".

Overrides (parameter sweeps) use dotted keys on the raw YAML tree:
    load("annulus_biot2.yaml", {"time.dt": 0.005, "params.A": 266.6, "geometry.h": 2.5e-4})
"""
import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import yaml

from biotx import geometry
from biotx.expressions import FUNCTIONS, VARIABLES

MODELS = ("biot2", "biot3")
SOURCES = {"biot2": ("body_force", "fluid_source"),                    # what each model reads
           "biot3": ("body_force", "fluid_force", "mass_source")}
FIELDS = {"biot2": ("d", "p"), "biot3": ("d", "w", "p")}               # unknowns (for `exact`)
MECHANICAL = ("displacement", "traction", "normal_traction")   # one per boundary
HYDRAULIC = ("pressure", "flux")                               # one per boundary


@dataclass(frozen=True)
class Scales:
    """Reference length [m], time [s] and stress [Pa]."""
    L0: float = 1.0
    T0: float = 1.0
    P0: float = 1.0

    # factors that turn an SI value of each kind into its non-dimensional value
    @property
    def factor(self) -> dict:
        L0, T0, P0 = self.L0, self.T0, self.P0
        return {"length": 1 / L0, "time": 1 / T0, "pressure": 1 / P0, "displacement": 1 / L0,
                "velocity": T0 / L0, "acceleration": T0**2 / L0, "body_force": L0 / P0, "rate": T0, "none": 1.0}


@dataclass(frozen=True)
class Material:
    """Physical parameters in SI. Give Lame (lambda, mu) or (E, nu) in the YAML."""
    lam: float               # Lame lambda [Pa]
    mu: float                # shear modulus [Pa]
    alpha: float             # Biot-Willis coefficient [-]
    c0: float                # constrained specific storage [1/Pa]
    K: float                 # intrinsic permeability [m^2]
    muF: float               # pore-fluid viscosity [Pa s]
    rho: float = 0.0         # mixture density [kg/m^3] (solid inertia)
    rho_f: float = 0.0       # pore-fluid density [kg/m^3]               (three-field only)
    phi: float = 1.0         # porosity [-]                               (three-field only)
    tortuosity: float = 1.0  # [-]                                        (three-field only)
    beta: float = 0.0        # leakage coefficient [1/(Pa s)]             (two-field only)

    @classmethod
    def from_dict(cls, raw: dict) -> "Material":
        raw = dict(raw)
        if "lambda" in raw:
            raw["lam"] = raw.pop("lambda")
        if "E" in raw:                                   # (E, nu) -> Lame
            E, nu = raw.pop("E"), raw.pop("nu")
            raw["lam"], raw["mu"] = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        m = cls(**raw)
        for name in ("mu", "K", "muF"):
            if getattr(m, name) <= 0:
                raise ValueError(f"material.{name} must be positive")
        return m

    def scaled(self, s: Scales) -> SimpleNamespace:
        """Non-dimensional coefficients of the equations (plain floats).
        Only kappa = K / muF enters the Darcy law, so muF disappears."""
        L0, T0, P0 = s.L0, s.T0, s.P0
        inertia = L0**2 / (P0 * T0**2)                  # rho -> rho*
        return SimpleNamespace(
            lam=self.lam / P0, mu=self.mu / P0, alpha=self.alpha, c0=self.c0 * P0,
            kappa=self.K / self.muF * P0 * T0 / L0**2,
            rho=self.rho * inertia, rhoF=self.rho_f * inertia,
            rhoW=self.rho_f * self.tortuosity / self.phi * inertia,
            beta=self.beta * P0 * T0)


@dataclass(frozen=True)
class Discretization:
    """DG degrees and SIPG penalties. deg_p: pressure degree (None: deg for biot2,
    deg - 1 for biot3, the stable P2/P2/P1 choice)."""
    deg: int = 2
    deg_p: int | None = None
    eta: float = 10.0        # elasticity penalty (also the normal-jump penalty of w in biot3)
    zeta: float = 10.0       # Darcy penalty (biot2)


@dataclass(frozen=True)
class Time:
    """Time interval [s], step [s] and schemes. The run starts from rest at t0 unless the
    model is given an initial state (Python API: model.set_initial)."""
    T: float
    dt: float
    t0: float = 0.0
    theta: float = 1.0           # pressure equation of biot2: 1 backward Euler, 0.5 Crank-Nicolson
    startup_steps: int = 0       # first steps with theta = 1 (Rannacher start, for sudden loads)
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5

    @property
    def nsteps(self) -> int:
        n = round((self.T - self.t0) / self.dt)
        if abs(n * self.dt - (self.T - self.t0)) > 1e-9 * max(1.0, abs(self.T)):
            raise ValueError("T - t0 must be a multiple of dt")
        return n


@dataclass(frozen=True)
class Output:
    """What the Recorder stores (all positions and times in SI)."""
    dir: str = "results"
    name: str | None = None                       # file stem; default: the YAML stem
    series: list = field(default_factory=lambda: ["flux", "p"])   # sampled at probes every step
    probes: dict = field(default_factory=dict)    # name -> [x, y(, z)]
    lines: dict = field(default_factory=dict)     # name -> {from: [x, y], to: [x, y], n: int}
    grid: int | None = None                       # 2D: n x n grid over the bounding box
    slices: dict = field(default_factory=dict)    # 3D: name -> {normal: x|y|z, at: value, n: int}
    snapshots: list = field(default_factory=list) # times of grid / slice / line / jump / fields
    fields: bool | list = False                   # full DG fields at the snapshots (true: d, p, flux)
    boundary_flux: bool = True                    # integral of q.n on every named boundary
    paraview: int = 0                             # write .bp files (SI) every n steps (0: never)


@dataclass(frozen=True)
class Experiment:
    """The whole experiment. geometry, params, loads, boundaries, sources stay plain
    dicts: they are interpreted by geometry.py and expressions.py."""
    name: str
    model: str
    scales: Scales
    material: Material
    geometry: dict
    boundaries: dict
    time: Time
    discretization: Discretization = Discretization()
    params: dict = field(default_factory=dict)
    loads: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    output: Output = Output()
    solver: dict = field(default_factory=dict)    # extra PETSc options (default: MUMPS LU)
    exact: dict = field(default_factory=dict)     # manufactured solution (see mms.py)
    convergence: dict = field(default_factory=dict)   # study description (see convergence.py)
    raw: dict = field(default_factory=dict, repr=False, compare=False)
    root: Path = Path(".")                         # directory of the YAML file

    def with_(self, overrides: dict) -> "Experiment":
        """A copy with dotted-key overrides, e.g. {"time.dt": 0.005, "discretization.deg": 1}."""
        return from_dict(_override(self.raw, overrides), self.name, self.root)


def _override(raw: dict, overrides: dict) -> dict:
    raw = copy.deepcopy(raw)
    for key, value in (overrides or {}).items():
        parts = key.split(".")
        d = raw
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return raw


def _check_boundaries(boundaries: dict):
    """Every boundary: exactly one mechanical and one hydraulic condition."""
    for name, bc in boundaries.items():
        unknown = set(bc) - set(MECHANICAL) - set(HYDRAULIC)
        if unknown:
            raise ValueError(f"boundary '{name}': unknown condition(s) {sorted(unknown)}")
        mech = [k for k in MECHANICAL if k in bc]
        hyd = [k for k in HYDRAULIC if k in bc]
        if len(mech) != 1 or len(hyd) != 1:
            raise ValueError(f"boundary '{name}' needs exactly one of {MECHANICAL} and one of "
                             f"{HYDRAULIC}; got {sorted(bc)}")


def _check_names(model, params, loads, sources):
    """Sources the model reads, and params / loads that do not hide a built-in formula name."""
    unknown = set(sources) - set(SOURCES[model])
    if unknown:
        raise ValueError(f"sources: {sorted(unknown)} not used by {model}; it reads {list(SOURCES[model])}")
    reserved = set(FUNCTIONS) | set(VARIABLES)
    for section, names in (("params", params), ("loads", loads)):
        bad = reserved & set(names)
        if bad:
            raise ValueError(f"{section}: {sorted(bad)} would hide built-in formula names; rename them")
    both = set(params) & set(loads)
    if both:
        raise ValueError(f"{sorted(both)} defined in both params and loads")


def _check_exact(model, exact, sources, boundaries):
    """A manufactured solution gives every unknown and replaces the sources; `exact` as a
    boundary value needs one, and cannot be a normal traction (the exact traction has a
    tangential part in general)."""
    if exact:
        missing, extra = set(FIELDS[model]) - set(exact), set(exact) - set(FIELDS[model])
        if missing or extra:
            raise ValueError(f"exact: {model} needs exactly {list(FIELDS[model])}; "
                             f"missing {sorted(missing)}, unknown {sorted(extra)}")
        if sources:
            raise ValueError("sources: not allowed with `exact` (they are derived from it)")
    for name, bc in boundaries.items():
        for key, value in bc.items():
            if isinstance(value, str) and value == "exact":
                if not exact:
                    raise ValueError(f"boundary '{name}': {key}: exact needs an `exact` section")
                if key == "normal_traction":
                    raise ValueError(f"boundary '{name}': normal_traction cannot be exact; use traction: exact")


def from_dict(raw: dict, name: str = "experiment", root=".") -> Experiment:
    """Build an Experiment from a raw YAML tree (unknown keys raise TypeError)."""
    model = raw.get("model", "biot2")
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got '{model}'")
    boundaries = raw["boundaries"]
    _check_boundaries(boundaries)
    geometry.check(raw["geometry"])
    _check_names(model, raw.get("params", {}), raw.get("loads", {}), raw.get("sources", {}))
    _check_exact(model, raw.get("exact", {}), raw.get("sources", {}), boundaries)
    out = raw.get("output", {})
    return Experiment(
        name=out.get("name") or name, model=model,
        scales=Scales(**raw.get("scales", {})),
        material=Material.from_dict(raw["material"]),
        geometry=raw["geometry"], boundaries=boundaries,
        time=Time(**raw["time"]),
        discretization=Discretization(**raw.get("discretization", {})),
        params=raw.get("params", {}), loads=raw.get("loads", {}),
        sources=raw.get("sources", {}), output=Output(**out),
        solver=raw.get("solver", {}), exact=raw.get("exact", {}),
        convergence=raw.get("convergence", {}), raw=copy.deepcopy(raw), root=Path(root))


def merge(base: dict, top: dict) -> dict:
    """base updated with top, recursively for nested sections."""
    out = copy.deepcopy(base)
    for k, v in top.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_study(path) -> dict:
    """The YAML tree of a study file. A top-level `base:` names another YAML (relative to the
    study) that the study extends with its own keys; the base's `output` is dropped."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if "base" in raw:
        base = yaml.safe_load((path.parent / raw.pop("base")).read_text())
        base.pop("output", None)                         # probes etc. are not used by studies
        raw = merge(base, raw)
    return raw


def load(path, overrides: dict | None = None) -> Experiment:
    """Read a YAML experiment, applying dotted-key overrides."""
    path = Path(path)
    raw = _override(yaml.safe_load(path.read_text()), overrides or {})
    return from_dict(raw, path.stem, path.parent)
