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
| `biotx/flux.py` | conservative velocity after each step (post-processing): Raviart–Thomas field whose facet flows are the solver's numerical flux. Output quantity `cflux` (and `cbflux_*`), or `Reconstruction(model).step()` |
| `biotx/post.py` | `Recorder` (probes, lines, grid / slices, boundary fluxes, normal jump, ParaView in SI), `load_results` |
| `biotx/fields.py` | full DG fields at the snapshots: `FieldWriter` (`output.fields`), `load_fields` -> Functions, time averages (`Fields.average`) |
| `biotx/mms.py` | manufactured solutions: an `exact:` section -> sources, `exact` boundary data, initial state (from each model's `strong_form`) |
| `biotx/convergence.py` | convergence studies in space, in time and by self-refinement (nested meshes); L2 / H1-seminorm errors and orders; `python -m biotx.convergence` |
| `biotx/conditioning.py` | 2-norm condition number of the step matrix (SLEPc: σ_max, and σ_min through A⁻¹A⁻ᵀ), raw, Jacobi- or Schur-scaled, which fields carry σ_min; sweeps of one YAML key; `python -m biotx.conditioning` |
| `biotx/compare.py` | relative L2 errors of runs w.r.t. a reference (any meshes), boundary-flux errors; `python -m biotx.compare` |
| `biotx/run.py` | `python -m biotx.run file.yaml --set key=value` |

To add a new model: subclass `BiotModel` and define `fields`, `_elements`, `_init_state`, `_residual`, `_advance`, `_set_initial` and `flux`. The base class does the rest.

Tests (`python -m pytest`, from the repo root):

- `test_regression_spx.py`: identical to the old spx `pulsatile.py`, to round-off, for 2F P1, 2F P2 and 3F. The spx results are frozen in `tests/data/spx_pulsatile.npz`, because spx was removed.
- `test_convergence.py`: manufactured solutions with every BC kind derived from `exact`; orders in space (P1, P2/P2/P1), in time (Crank–Nicolson 2, backward Euler 1, Newmark 2) and by self-refinement; the checks on `exact` and `convergence`.
- `test_conditioning.py`: SLEPc condition numbers against the dense SVD (both models, raw, Jacobi and Schur, 2D and 3D), the Schur diagonal by hand, the matrix of the requested θ, the study driver and its checks.
- `test_flux.py`: the reconstructed velocity's facet flows equal the solver's, its normal jumps vanish, its divergence is the solver's fluid balance (both models, every BC kind), and it converges at the order of the raw flux; `cflux` in probes, fields (exact), ParaView and `cbflux`; snapshot windows; `Fields.average`.
- `test_config.py`: parsing, scaling, overrides and BC validation.
- `test_fields.py`: exact fields round trip, comparisons (self = 0), 3D shapes, a 3D run with slices, and a check that SI ParaView output leaves the run unchanged.
