"""Typed configuration: one YAML file -> frozen dataclasses."""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MeshConfig:
    kind: str
    dim: int
    refinements: list[int]
    cells: dict[str, int]
    facets: dict[str, list[int]]
    file: str | None = None


@dataclass(frozen=True)
class Material:
    rhoS: float      # fluid density
    muS: float       # fluid viscosity
    rhoP: float      # poroelastic density
    lambdaP: float   # Lame lambda
    muP: float       # Lame mu
    alpha: float     # Biot-Willis coefficient
    c0: float        # constrained specific storage coefficient
    muF: float       # pore-fluid viscosity
    K: float         # permeability
    beta: float      # external coupling
    gammaBJS: float  # BJS slip coefficient

    def __post_init__(self):
        for name in ("muS", "muP", "muF", "K"):
            if getattr(self, name) <= 0:
                raise ValueError(f"material.{name} must be positive")


@dataclass(frozen=True)
class Discretization:
    deg: int        # polynomial degree of uS, d, pP
    degP: int       # polynomial degree of pS 
    gammaV: float   # Stokes velocity penalty
    gammaP: float   # Stokes pressure-jump stabilization parameter
    eta: float      # elasticity penalty
    zeta: float     # Darcy penalty


@dataclass(frozen=True)
class TimeConfig:
    t0: float
    T: float
    dt: float
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5


@dataclass(frozen=True)
class SolverConfig:
    kind: str = "direct"    # "direct" (MUMPS) or "iterative" (GMRES)
    rtol: float = 1e-10
    maxit: int = 500


@dataclass(frozen=True)
class OutputConfig:
    dir: str
    every: int = 1


@dataclass(frozen=True)
class Config:
    mesh: MeshConfig
    material: Material
    discretization: Discretization
    time: TimeConfig | None
    solver: SolverConfig
    output: OutputConfig


def load(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        mesh=MeshConfig(**raw["mesh"]),
        material=Material(**raw["material"]),
        discretization=Discretization(**raw["discretization"]),
        time=TimeConfig(**raw["time"]) if "time" in raw else None,
        solver=SolverConfig(**raw.get("solver", {})),
        output=OutputConfig(**raw["output"]),
    )
