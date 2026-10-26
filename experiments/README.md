# Experiments with biotx

An experiment is **one YAML file** describing the physics. Every key is documented in
[`YAML_REFERENCE.md`](YAML_REFERENCE.md). You run it with

```bash
conda activate biotx                     # dolfinx 0.11 + gmsh; biotx installed with pip install -e . (repo root)
python -m biotx.run experiments/annulus_biot2.yaml
python -m biotx.run experiments/annulus_biot3.yaml --set time.dt=0.005 --name annulus_biot3_fine
mpirun -n 4 python -m biotx.run experiments/annulus_biot2.yaml      # parallel works too
```

The results go to `<yaml dir>/<output.dir>/<name>.npz`. Everything in that file is in SI units. Open it with `biotx.load_results(path)`.

## Designing an experiment: the YAML, section by section

All values are in **SI**. `scales` only chooses the non-dimensional problem that is solved, so you never convert units by hand.

```yaml
model: biot2              # biot2: (d, p)    biot3: (d, w, p), fluid inertia, q = w_t

scales: {L0: 1.0e-2, T0: 1.0, P0: 1500.0}     # m, s, Pa (default 1, 1, 1)

material:                 # lambda+mu or E+nu; rho_f, phi, tortuosity only for biot3; beta only biot2
  {mu: 1500, lambda: 11567, alpha: 0.49, c0: 1.0e-6, K: 1.57e-11, muF: 3.5e-3,
   rho: 1000, rho_f: 1000, phi: 0.2, tortuosity: 1.6}

geometry:                 # 2D: rectangle | disc | annulus   3D: box | ball | shell   any: gmsh
  shape: annulus
  radius: 1.0e-2
  inner: {center: [3.0e-3, 0], radius: 2.5e-3}
  h: 5.0e-4
  names: {inner: ventricle, outer: pia}   # rename boundaries to what they mean

params: {A: 133.3}        # numbers you want to vary (sweeps: --set params.A=266.6)
loads:                    # named expressions of x, y, t, r and params; may build on each other
  pV: "A * (1 - cos(2*pi*t)) / 2"
  ramp: "where(t < 0.1, t / 0.1, 1)"

boundaries:               # EVERY boundary: one mechanical + one hydraulic condition
  ventricle: {pressure: pV, normal_traction: -pV}
  pia:       {pressure: 0,  displacement: [0, 0]}

sources: {}               # biot2: body_force, fluid_source | biot3: body_force, fluid_force, mass_source

discretization: {deg: 2, eta: 10, zeta: 10}   # deg_p defaults: deg (biot2), deg-1 (biot3)
time: {T: 3.0, dt: 0.01, theta: 0.5}          # theta: biot2 pressure equation; startup_steps: Rannacher

output:
  series: [flux, p]       # quantities sampled at the probes every step (d, p, flux, velocity, w)
  probes: {thick: [-5.0e-3, 0]}
  lines:  {thin: {from: [5.51e-3, 0], to: [9.99e-3, 0], n: 201}}
  grid: 81                # 2D only: n x n grid over the bounding box, at the snapshots
  snapshots: [2.25, 2.5, 2.75, 3.0]
  fields: true            # full fields at the snapshots -> <name>_fields.npz (to compare runs)
  boundary_flux: true     # outflow through every named boundary, every step
  paraview: 0             # every n steps (.bp files, SI: mesh in m, fields in SI units)
```

### Boundary conditions

| mechanical | meaning | hydraulic | meaning |
|---|---|---|---|
| `displacement: [gx, gy]` | d = g (Nitsche) | `pressure: g` | p = g |
| `traction: [tx, ty]` | total traction (σ_e − αpI)n = t | `flux: g` | outward Darcy flux q·n = g (biot3: only 0) |
| `normal_traction: s` | traction s·n (a fluid pressure P on a wall: s = −P) | | |

A boundary of the mesh with no conditions is an error, and so is a misspelled condition. Nothing is ever silently left traction-free.

### What is in the .npz

| key | shape | when |
|---|---|---|
| `t` | (nt,) | every step (and t0) |
| `probe_<q>` | (nt, nprobes[, 2]) | every step, for each quantity in `series` |
| `bflux_<boundary>` | (nt,) | every step: ∫ q·n ds (m²/s per unit depth in 2D) |
| `snap_t`, `grid_<q>`, `slice_<name>_<q>`, `line_<name>_<q>`, `jump` | (nsnap, ...) | at the snapshots |
| `probes`, `probe_names`, `grid`, `slice_<name>`, `line_<name>` | positions [m] | |
| `ncells`, `experiment` | | the mesh size and the full YAML (exactly reproducible) |

## 3D experiments

Everything above works in 3D. These are the differences:

- Geometry: `box` (boundaries left/right, front/back, bottom/top for x, y, z), `ball` (outer),
  `shell`, the 3D annulus (outer and inner, and the inner sphere may be off-centre), or a 3D `gmsh`
  file with `dim: 3`. Examples are at the top of `biotx/geometry.py`.
- Vectors have three components (`displacement: [0, 0, 0]`). A wrong length is an error.
  Expressions can use `z`.
- Probes and lines take 3D points.
- `grid` is replaced by named **slices**: each one is an n x n grid on a plane, at the snapshots.

```yaml
geometry: {shape: shell, radius: 1.0e-2, inner: {center: [3.0e-3, 0, 0], radius: 2.5e-3},
           h: 1.0e-3, names: {inner: ventricle, outer: pia}}
output:
  slices: {mid: {normal: z, at: 0.0, n: 81}, side: {normal: x, at: -5.0e-3}}
```

`slice_<name>_<q>` has shape (nsnap, n*n[, 3]) and `slice_<name>` holds the points (n*n, 3).
`reshape(n, n)` gives `[i_b, i_a]`, where (a, b) are the two in-plane axes in xyz order.

## Comparing runs

Run with `output.fields: true`. Next to `<name>.npz` you then also get `<name>_fields.npz`,
which holds the exact DG fields at the snapshots, in SI. Then:

```bash
python -m biotx.compare results/ref.npz results/run_a.npz results/run_b.npz --t-min 2
```

```text
run    d          p          flux       Q_pia      Q_ventricle
run_a  1.291e-01  6.611e-02  1.619e-01  7.159e-02  2.448e-01
```

- **Field columns.** `max_t ||q_run - q_ref||_L2 / ||q_ref||_L2` over the snapshot times both runs share. The run is
  interpolated onto the reference mesh, so the meshes, degrees and models may all differ. This
  works in 2D and 3D, serial or under `mpirun`.
- **`Q_` columns.** `max_t |Q_run - Q_ref| / max_t |Q_ref|` of the boundary fluxes.
- **From Python.** `biotx.compare(ref, runs, t_min=...)` returns the same numbers, with the error at every
  snapshot.
- **Single fields.** `biotx.load_fields(path).function("p", i)` gives back one field as a dolfinx Function. You can use it for your own norms,
  or to look at it again later.

The grid, slices and lines are for **figures**. The fields are for **numbers**. A DG field jumps
across facets, so point samples near a facet hit one side arbitrarily. The L2 comparison
integrates over everything.

## Convergence studies

A convergence study is a YAML file with a `convergence:` section, run with `python -m biotx.convergence`. Every
key is in [`YAML_REFERENCE.md`](YAML_REFERENCE.md) §14–15. `experiments/convergence/` has five studies:

| file | study | what it shows |
|---|---|---|
| `mms_biot2_space.yaml` | space | P1/P1 (or `--set discretization.deg=2`): orders in h, every BC kind |
| `mms_biot3_space.yaml` | space | P2/P2/P1: orders in h |
| `mms_biot2_time.yaml` | time | Crank–Nicolson order 2 (`--set time.theta=1`: backward Euler, order 1) |
| `mms_biot3_time.yaml` | time | Newmark order 2 |
| `annulus_biot2_self.yaml` | self | the annulus experiment, refined uniformly, with no exact solution |

```bash
python -m biotx.convergence experiments/convergence/mms_biot2_space.yaml
```

The manufactured studies give an `exact:` solution. The sources, every boundary condition written as
`exact`, and the initial state are derived from it symbolically, so nothing is derived by hand.

## Condition numbers

A conditioning study measures cond₂ = σ_max/σ_min of the step matrix for each value of one swept
key (raw, Jacobi- and Schur-scaled), run with `python -m biotx.conditioning`. Every key is in
[`YAML_REFERENCE.md`](YAML_REFERENCE.md) §16. `experiments/conditioning/` has six studies, on the
annulus with the brain parameters:

| file | sweep |
|---|---|
| `annulus_biot2_h.yaml`, `annulus_biot3_h.yaml` | mesh size h = 4, 2, 1, 0.5 mm, at dt = 10 ms |
| `annulus_biot2_h_dt2.5ms.yaml`, `annulus_biot3_h_dt2.5ms.yaml` | the same, at dt = 2.5 ms |
| `annulus_biot2_dt.yaml`, `annulus_biot3_dt.yaml` | time step dt = 40 ms to 2.5 ms, at h = 0.5 mm |

```bash
python -m biotx.conditioning experiments/conditioning/annulus_biot2_h.yaml
```

`conditioning.ipynb` shows them all, in tables and figures, with a reading of the results.

## When the YAML is not enough: the Python API

```python
import biotx
exp = biotx.load("experiments/annulus_biot2.yaml", {"params.A": 266.6})
model = biotx.make_model(exp)                  # or biotx.TwoField(exp) / biotx.ThreeField(exp)
model.set_initial(d=lambda x, t: ...)          # optional: start from a state, not from rest
rec = biotx.Recorder(model)
rec(model)
for t in model.steps():                        # one solved step per iteration
    q, to_si = model.quantity("flux")          # any Function; multiply values by to_si for SI
    ...                                        # your own measurements
    rec(model)
rec.save()
```

- Anywhere the YAML takes a string, Python can take a callable `f(x, t)` returning UFL (x and t in SI). The `exact:` section accepts callables too (see `tests/test_convergence.py`).
- `biotx.Points(model.mesh, pts, L0)` evaluates any Function at points, and works in parallel.
- To reuse one mesh in several runs, pass `domain=model.domain` to the next model.
- To change a coefficient mid-run, set `model.c.<name>.value = ...` and then call `model.solver.update_matrix()`. The matrix is factorised only once.

Examples in this folder:

- `annulus_biot2.yaml`: the two-field model on a concentric annulus, with a pulsating cavity
  pressure and a fixed, drained outer wall.
- `annulus_biot3.yaml`: the same problem with the three-field model.

Both save fields, so they can be compared with `python -m biotx.compare`.
