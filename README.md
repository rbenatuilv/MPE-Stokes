# biotx

Two- and three-field Biot models in DG (FEniCSx 0.11), driven by YAML experiments. See
`experiments/README.md` for how to write and run an experiment.

| file | what is in it |
|---|---|
| `biotx/config.py` | YAML to `Experiment`; SI to non-dimensional (`Scales`, `Material.scaled`); overrides |
| `biotx/geometry.py` | rectangle / disc / annulus, box / ball / shell, gmsh meshes with **named** boundaries |
| `biotx/expressions.py` | strings or callables to UFL expressions of x, t (SI); exact d/dt |
| `biotx/common.py` | shared DG pieces (SIPG elasticity, b_P), Newmark, BCs, MUMPS solver, `BiotModel` time loop |
| `biotx/biot2.py` | `TwoField` (d, p): equations, scheme, BCs, flux = −κ∇p, explained at the top |
| `biotx/biot3.py` | `ThreeField` (d, w, p): equations, scheme, BCs, flux = w_t, explained at the top |
| `biotx/post.py` | `Recorder` (probes, lines, grid / slices, boundary fluxes, normal jump, ParaView in SI), `load_results` |
| `biotx/fields.py` | full DG fields at the snapshots: `FieldWriter` (`output.fields`), `load_fields` -> Functions |
| `biotx/compare.py` | relative L2 errors of runs w.r.t. a reference (any meshes), boundary-flux errors; `python -m biotx.compare` |
| `biotx/run.py` | `python -m biotx.run file.yaml --set key=value` |

To add a new model: subclass `BiotModel` and define `fields`, `_elements`, `_init_state`, `_residual`, `_advance`, `_set_initial` and `flux`. The base class does the rest.

Tests (`python -m pytest`, from the repo root):

- `test_regression_spx.py`: identical to the old spx `pulsatile.py`, to round-off, for 2F P1, 2F P2 and 3F. The spx results are frozen in `tests/data/spx_pulsatile.npz`, because spx was removed.
- `test_mms.py`: manufactured-solution space rates. They cover every BC kind, the sources and `set_initial`.
- `test_config.py`: parsing, scaling, overrides and BC validation.
- `test_fields.py`: exact fields round trip, comparisons (self = 0), 3D shapes, a 3D run with slices, and a check that SI ParaView output leaves the run unchanged.
