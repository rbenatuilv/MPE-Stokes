# biotx YAML reference

This file lists **every key** an experiment YAML can contain, with its meaning, units, default value and
the model it applies to. For a gentler introduction see `README.md`.

Contents:

1. General rules
2. `model`
3. `scales`
4. `material`
5. `geometry`
6. `params`
7. `loads` and the expression language
8. `boundaries`
9. `sources`
10. `discretization`
11. `time`
12. `output`
13. `solver`
14. `exact`: manufactured solutions
15. `convergence`: convergence studies
16. Command line and Python overrides
17. Errors and pitfalls
18. Complete examples (2D, 3D)

---

## 1. General rules

- **Everything is in SI units**: m, s, Pa, kg/m³, m², Pa·s. Results are saved in SI as well. The
  `scales` section only changes which non-dimensional problem is solved internally. You never
  convert units by hand.
- **Skeleton.** Sections marked *required* must be present. All others have defaults.

  ```yaml
  model: biot2            # optional (default biot2)
  scales: {...}           # optional
  material: {...}         # REQUIRED
  geometry: {...}         # REQUIRED
  params: {...}           # optional
  loads: {...}            # optional
  boundaries: {...}       # REQUIRED
  sources: {...}          # optional
  discretization: {...}   # optional
  time: {...}             # REQUIRED
  output: {...}           # optional
  solver: {...}           # optional
  exact: {...}            # optional: manufactured solution (§14)
  convergence: {...}      # optional: convergence study (§15)
  ```
- **Where formulas are allowed.** Formulas such as `"A*sin(2*pi*t)"` are accepted only in
  `loads`, `boundaries` and `sources`. `material`, `geometry`, `params`, `scales`, `time`,
  `discretization` and `output` take plain numbers, lists and strings.
- **Unknown keys are errors** in every section except the top level, where an extra key (e.g. a
  `description:` note) is ignored. This covers misspelled keys, sources the model does not read, and params or loads that
  would hide a built-in name (see §17).
- **The whole YAML is stored** in every result file (`experiment` key), so a run can always be
  reproduced.

---

## 2. `model`

| value | unknowns | flux q | notes |
|---|---|---|---|
| `biot2` (default) | d (solid displacement), p (pressure) | q = −(K/μF)∇p, computed afterwards from p | quasi-static or with solid inertia |
| `biot3` | d, w (relative fluid displacement), p | q = ∂w/∂t, an unknown of the system | fluid inertia included; only `flux: 0` boundaries |

```yaml
model: biot3
```

---

## 3. `scales`

These are reference values for the internal non-dimensional problem: x* = x/L0, t* = t/T0, p* = p/P0,
d* = d/L0, q* = q·T0/L0.

| key | unit | default | meaning |
|---|---|---|---|
| `L0` | m | 1 | typical length (domain size) |
| `T0` | s | 1 | typical time (period, duration) |
| `P0` | Pa | 1 | typical stress (shear modulus or load amplitude) |

The scales **do not change the physics**, only the conditioning of the numbers. A good choice
makes the non-dimensional coefficients O(1): L0 equal to the domain size, T0 equal to the forcing period,
and P0 equal to μ. With the defaults (1, 1, 1), the problem is solved with the SI numbers as given.

```yaml
scales: {L0: 1.0e-2, T0: 1.0, P0: 1500.0}
```

---

## 4. `material`

*Required.* All values in SI.

### Elasticity: give **either** `lambda` + `mu` **or** `E` + `nu`

| key | unit | required | meaning |
|---|---|---|---|
| `lambda` | Pa | yes (or E, nu) | Lamé's first parameter |
| `mu` | Pa | yes (or E, nu) | shear modulus, must be > 0 |
| `E` | Pa | instead of lambda, mu | Young's modulus |
| `nu` | – | with `E` | Poisson's ratio (< 0.5) |

### Poroelasticity

| key | unit | required | default | model | meaning |
|---|---|---|---|---|---|
| `alpha` | – | yes | | both | Biot–Willis coefficient (0 = no coupling) |
| `c0` | 1/Pa | yes | | both | storage coefficient (0 = incompressible constituents) |
| `K` | m² | yes | | both | intrinsic permeability, > 0 |
| `muF` | Pa·s | yes | | both | fluid viscosity, > 0 (only K/muF matters) |
| `rho` | kg/m³ | no | 0 | both | mixture density: solid inertia (0 = quasi-static) |
| `beta` | 1/(Pa·s) | no | 0 | biot2 | leakage: adds a β·p sink to the mass equation |
| `rho_f` | kg/m³ | no | 0 | biot3 | fluid density: fluid inertia (0 = Darcy without inertia) |
| `phi` | – | no | 1 | biot3 | porosity |
| `tortuosity` | – | no | 1 | biot3 | tortuosity: fluid inertia ρF·τ/φ |

Keys a model does not use are accepted and ignored, so the same material block serves both models.

```yaml
material: {mu: 1500, lambda: 11567, alpha: 0.49, c0: 1.0e-6, K: 1.57e-11, muF: 3.5e-3,
           rho: 1000, rho_f: 1000, phi: 0.2, tortuosity: 1.6}
```

---

## 5. `geometry`

*Required.* All lengths are in m. Every shape produces **named boundaries**, and `boundaries` (§8) must
give conditions for **all** of them.

### Keys common to every shape

| key | meaning |
|---|---|
| `shape` | `rectangle`, `box`, `disc`, `annulus`, `ball`, `shell` or `gmsh` |
| `names` | optional renaming of the boundaries, e.g. `{inner: ventricle, outer: pia}` |

Each shape accepts only the keys listed in its table below. Any other key is an error, and so is a
`rectangle` or `box` without `n` or `h`.

### `rectangle` (2D, triangles)

| key | default | meaning |
|---|---|---|
| `corners` | required | `[[x0, y0], [x1, y1]]` |
| `n` | | `[nx, ny]` cells per side |
| `h` | | target cell size, used if `n` is not given (one of the two is required) |
| `diagonal` | `right` | triangle diagonals: `left`, `right`, `crossed`, `left_right`, `right_left` |

Boundaries: `left` (x = x0), `right` (x = x1), `bottom` (y = y0), `top` (y = y1).

### `box` (3D, tetrahedra)

| key | meaning |
|---|---|
| `corners` | `[[x0, y0, z0], [x1, y1, z1]]` |
| `n` or `h` | `[nx, ny, nz]` cells per side (6 tetrahedra per cube), or a target size |

Boundaries: `left`/`right` (x), `front`/`back` (y), `bottom`/`top` (z).

### `disc` (2D) and `ball` (3D), meshed with gmsh

| key | default | meaning |
|---|---|---|
| `center` | origin | `[x, y]` or `[x, y, z]` |
| `radius` | required | radius |
| `h` | required | maximum cell size |

Boundary: `outer`.

### `annulus` (2D) and `shell` (3D): a disc or ball with a hole, meshed with gmsh

| key | default | meaning |
|---|---|---|
| `center` | origin | centre of the outer circle or sphere |
| `radius` | required | outer radius |
| `inner` | required | `{center: [...], radius: r}`. The hole may be off-centre but must lie inside. |
| `h` | required | maximum cell size |

Boundaries: `outer`, `inner`.

### `gmsh`: any mesh file

| key | default | meaning |
|---|---|---|
| `file` | required | `.msh` path, relative to the YAML file |
| `dim` | 2 | 2 for a surface mesh, 3 for a volume mesh |
| `tags` | | names for physical groups that have none, e.g. `{ventricle: 1, skull: 2}` |

Boundaries: the **named physical groups** of dimension `dim − 1` in the file, plus `tags`. The
mesh is used as it is: `h` and refinement are set when you create the file in gmsh.

```yaml
geometry:
  shape: annulus
  radius: 1.0e-2
  inner: {center: [3.0e-3, 0], radius: 2.5e-3}
  h: 5.0e-4
  names: {inner: ventricle, outer: pia}
```

---

## 6. `params`

Named **numbers** (converted to float) that formulas can use. They are the natural knobs for
parameter sweeps (`--set params.A=266.6`).

```yaml
params: {A: 133.3, f: 1.0, t_ramp: 0.1}
```

A param must not reuse a built-in name (`x`, `y`, `z`, `t`, `r`, `pi`, `e`, `sin`, …) or the name
of a load. Either one is an error.

---

## 7. `loads` and the expression language

`loads` holds named **formulas**. They are evaluated in the order written, so a load may use params
and earlier loads. Boundaries and sources can then refer to them by name. Load names follow the same
rule as params: a built-in name is an error.

```yaml
loads:
  ramp: "where(t < t_ramp, t / t_ramp, 1)"
  pV:   "ramp * A * (1 - cos(2*pi*f*t)) / 2"
```

### Values accepted wherever a formula is allowed (loads, boundaries, sources)

| form | example | meaning |
|---|---|---|
| number | `0`, `-1500` | constant |
| string | `"A*sin(2*pi*t)"` | formula |
| list | `[0, 0]`, `[0, "-rho*9.81"]` | vector, each entry a number or formula (2 entries in 2D, 3 in 3D) |

### Names available in formulas

| name | meaning |
|---|---|
| `x`, `y` | coordinates [m] |
| `z` | coordinate [m] (**3D only**) |
| `r` | distance **from the origin** [m]: √(x²+y²) in 2D, √(x²+y²+z²) in 3D |
| `t` | time [s] |
| `pi`, `e` | constants |
| `sin cos tan exp log sqrt abs tanh sinh cosh atan` | functions (`log` is the natural log) |
| `min(a, b)`, `max(a, b)` | two arguments |
| `where(cond, a, b)` | `a` where `cond` holds, else `b`. Conditions: `<`, `>`, `<=`, `>=`. Combine conditions by nesting `where`, because `and`/`or` do not work. |
| params, earlier loads | by name |

Operators: `+ - * / **` and parentheses. Python built-ins are not available.

### How formulas are used

- They are evaluated **exactly** at quadrature points, not interpolated. Time enters through `t`.
- **Time derivatives are exact.** A prescribed displacement `g(x, t)` automatically provides
  its velocity ∂g/∂t, which the mass equation needs, by symbolic differentiation. You never give it.
- In `biot2` with `theta < 1`, data are also evaluated at the previous time step automatically.

---

## 8. `boundaries`

*Required.* For **every** boundary name of the geometry, give **exactly one mechanical** and
**exactly one hydraulic** condition. Each value is a number, a formula, a load name or a vector (§7).
With an `exact` section (§14), any condition except `normal_traction` can also be the word
`exact`: its data is then derived from the exact solution.

```yaml
boundaries:
  ventricle: {pressure: pV, normal_traction: -pV}
  pia:       {pressure: 0,  displacement: [0, 0]}
```

### Mechanical (one of)

| key | value | unit | condition | how imposed |
|---|---|---|---|---|
| `displacement` | vector | m | d = g (time-dependent allowed) | Nitsche (weak) |
| `traction` | vector | Pa | (σ_e − αpI)·n = t, the **total** traction | natural |
| `normal_traction` | scalar | Pa | (σ_e − αpI)·n = s·n | natural |

For a fluid at pressure P pushing on the wall, use `normal_traction: -P` (n points outward).
`traction: [0, 0]` or `normal_traction: 0` is a free surface.

### Hydraulic (one of)

| key | value | unit | condition | how imposed | model |
|---|---|---|---|---|---|
| `pressure` | scalar | Pa | p = g | biot2: Nitsche. biot3: natural | both |
| `flux` | scalar | m/s | outward Darcy flux q·n = g (`0` = impermeable) | biot2: natural. biot3: penalised | biot2: any g. **biot3: only `0`** |

A sign reminder: `flux` > 0 means fluid **leaves** the domain, as in `bflux_<name>` in the output.

---

## 9. `sources`

Optional volume sources. Each is a number, a formula or a vector. They default to zero.

| key | value | unit | model | meaning |
|---|---|---|---|---|
| `body_force` | vector | N/m³ | both | force per volume on the mixture (e.g. gravity `[0, "-rho*9.81"]` with rho a param) |
| `fluid_source` | scalar | 1/s | biot2 | volume of fluid injected per volume per time (g in the mass equation) |
| `fluid_force` | vector | N/m³ | biot3 | force per volume on the fluid (fluid momentum equation) |
| `mass_source` | scalar | – | biot3 | volume of fluid injected per volume, **accumulated**: the biot3 mass equation has no time derivative, so this is the integral of the injection rate over time |

```yaml
sources:
  fluid_source: "1e-3 * exp(-(x**2 + y**2) / 1e-6) * where(t < 1, 1, 0)"
```

A source the chosen model does not read, or a misspelled name, is an **error**. Switching
`model` therefore also means adapting `sources`.

---

## 10. `discretization`

| key | default | meaning |
|---|---|---|
| `deg` | 2 | polynomial degree of d (and of w in biot3). ≥ 1 |
| `deg_p` | biot2: `deg`. biot3: `deg − 1` | degree of p. biot2: keep ≥ 1, because the flux has degree `deg_p − 1`. biot3: `deg − 1` is the stable choice (P2/P2/P1) |
| `eta` | 10 | elasticity penalty (and the penalty on the normal jump of w in biot3). Already multiplied internally by (λ+2μ)·deg², so it is dimensionless |
| `zeta` | 10 | Darcy penalty (biot2), multiplied internally by max(κ*, 1)·deg² |

The penalties must be "large enough" for stability. A value of 10 works for the cases tested. Much larger
values give the same answer with a worse-conditioned matrix. Much smaller values (≲ 1) can make the scheme unstable.

---

## 11. `time`

*Required.*

| key | unit | default | model | meaning |
|---|---|---|---|---|
| `T` | s | required | both | final time |
| `dt` | s | required | both | time step. `T − t0` must be a multiple of `dt` |
| `t0` | s | 0 | both | initial time. Formulas see `t` starting at `t0` |
| `theta` | – | 1 | biot2 | θ-method for the mass equation: 1 = backward Euler (1st order, robust), 0.5 = Crank–Nicolson (2nd order) |
| `startup_steps` | – | 0 | biot2 | number of initial steps run with θ = 1 before switching to `theta` (Rannacher start: damps the oscillations Crank–Nicolson produces after a sudden load) |
| `newmark_beta` | – | 0.25 | both | Newmark β for d (and w) |
| `newmark_gamma` | – | 0.5 | both | Newmark γ. (β, γ) = (1/4, 1/2) is unconditionally stable, 2nd order and without numerical damping. γ > 1/2 adds damping |

The run starts **from rest** (d = 0, p = 0, velocities = 0) at `t0`. To start from another
state, use the Python API: `model.set_initial(d=..., p=...)`. The YAML cannot do this.

---

## 12. `output`

Everything about what is saved. All positions are in m. All results are in SI.

| key | default | meaning |
|---|---|---|
| `dir` | `results` | output folder, **relative to the YAML file** |
| `name` | YAML file stem | file stem: results go to `<dir>/<name>.npz` |
| `series` | `[flux, p]` | quantities recorded at the probes **every step** |
| `probes` | `{}` | named points: `{center: [x, y]}` (3 coordinates in 3D) |
| `lines` | `{}` | named segments: `{name: {from: [...], to: [...], n: 101}}` (`n` defaults to 101) |
| `grid` | none | **2D only**: n means an n×n grid of points over the mesh bounding box |
| `slices` | `{}` | **3D only**: named planes `{name: {normal: x\|y\|z, at: value, n: 81}}` |
| `snapshots` | `[]` | times at which grid, slices, lines, the flux-jump measure and fields are recorded |
| `fields` | `false` | `true` (= `[d, p, flux]`) or a list: full fields at the snapshots, to `<name>_fields.npz`. Needed by `biotx.compare` |
| `boundary_flux` | `true` | fluid outflow ∫q·n through every named boundary, every step |
| `paraview` | 0 | write ParaView `.bp` files every n steps (0 = never), in SI, to `<dir>/<name>/<q>.bp` |

### Quantities (for `series` and `fields`)

| name | meaning | unit | model |
|---|---|---|---|
| `d` | solid displacement | m | both |
| `p` | pore pressure | Pa | both |
| `flux` | Darcy flux q | m/s | both (biot2: −κ∇p; biot3: ∂w/∂t) |
| `velocity` | solid velocity ∂d/∂t | m/s | both |
| `w` | relative fluid displacement | m | biot3 |

Grid, slices and lines record the quantities in `series` together with `flux` and `p`. ParaView writes the same set plus `d`.

### Details

- **Snapshot times** are matched to the nearest time step within dt/2. A snapshot with no step that
  close is skipped. `snap_t` stores the times actually recorded. `t0` can be a snapshot (the initial
  state).
- Points outside the mesh, such as grid corners outside a disc, give **NaN**.
- For a slice, `reshape(n, n)` gives `[i_b, i_a]`, where (a, b) are the two in-plane axes in xyz order. For example,
  `normal: z` gives (a, b) = (x, y).

### Contents of `<name>.npz` (read with `biotx.load_results`)

| key | shape | when |
|---|---|---|
| `t` | (nt,) | every step, including t0 |
| `probe_<q>` | (nt, nprobes[, dim]) | every step |
| `bflux_<boundary>` | (nt,) | every step. m²/s per unit depth in 2D, m³/s in 3D. Positive means outflow |
| `snap_t` | (nsnap,) | snapshots |
| `grid_<q>` | (nsnap, n², [dim]) | snapshots (2D) |
| `slice_<name>_<q>` | (nsnap, n², [dim]) | snapshots (3D) |
| `line_<name>_<q>` | (nsnap, npts, [dim]) | snapshots |
| `jump` | (nsnap,) | ‖⟦q·n⟧‖ / ‖{q}·n‖ on interior facets: 0 means a perfectly conservative flux |
| `probes`, `probe_names`, `grid`, `slice_<name>`, `line_<name>` | positions | |
| `ncells`, `experiment` | | mesh size, full YAML |

---

## 13. `solver`

Extra **PETSc options**. They are added to the default, a direct LU solve with MUMPS:

```yaml
solver: {}                                   # default: ksp_type preonly, pc_type lu,
                                             # pc_factor_mat_solver_type mumps, mat_mumps_icntl_14 80
solver: {mat_mumps_icntl_14: 200}            # more MUMPS workspace ("not enough memory" errors)
```

Options are written without the leading `-`, and are prefixed internally so that two models in one script do not
interfere. Any PETSc option can be passed, including iterative solvers (`ksp_type: gmres`, …), but
only the direct solver has been tested. An option that PETSc does not use prints a warning at the end.

---

## 14. `exact`: manufactured solutions

Optional. It gives **every unknown** of the model as a formula of x, y, (z), t, in SI. This uses the same language
as `loads` (§7), so params and loads can be used:

```yaml
exact:
  d: ["(1 + sin(t)) * sin(pi*x) * cos(pi*y)", "(1 + sin(t)) * sin(pi*y) * cos(pi*x)"]
  w: [...]                                   # biot3 only
  p: "(1 + sin(t)) * (cos(pi*x) * cos(pi*y) + 0.5)"
boundaries:
  left:  {displacement: exact, pressure: exact}
  right: {traction: exact,     flux: exact}
```

Everything else is then **derived exactly** by symbolic differentiation:

| what | from the exact fields |
|---|---|
| sources | the model's equations applied to them: biot2 `body_force`, `fluid_source`; biot3 `body_force`, `fluid_force`, `mass_source` |
| `displacement: exact` | d |
| `traction: exact` | (σ_e(d) − αpI)·n |
| `pressure: exact` | p |
| `flux: exact` | q·n, with q = −(K/μF)∇p (biot2) or ∂w/∂t (biot3) |
| initial state | d, ∂d/∂t, ∂²d/∂t², p (and w, ∂w/∂t, ∂²w/∂t²) at t0, instead of rest |

Rules:

- **biot2** needs `d` and `p`; **biot3** needs `d`, `w` and `p`. A missing or extra field is an error.
- A `sources` section together with `exact` is an error, because the sources are derived.
- `normal_traction: exact` is an error: the exact traction generally has a tangential part. Use `traction: exact`.
- With biot3, `flux: exact` is accepted only if w·n = 0 on those boundaries at all times. This is checked, because biot3 supports only zero flux.
- From Python, each field can also be a callable `f(x, t)` that returns UFL (x and t in SI).

`exact` is meant for convergence studies (§15), but a plain run with it works too.

---

## 15. `convergence`: convergence studies

A study is an experiment YAML with a `convergence:` section, run with

```bash
python -m biotx.convergence study.yaml [--set KEY=VALUE]
```

It runs the experiment once per level, prints a table of errors and observed orders, and saves
`<dir>/<output.dir>/<name>.npz`, which can be read with `biotx.convergence.load_result`. There are three kinds of study:

| `study` | needs | levels given by | measures |
|---|---|---|---|
| `space` | `exact` | `n: [8, 16, 32]` (cells per side, rectangle or box) **or** `h: [...]` (any shape) | ‖u − u_h‖ at fixed dt |
| `time` | `exact` | `dt: [0.1, 0.05, 0.025]` | ‖u − u_h‖ on a fixed mesh |
| `self` | nothing | `levels: 4` (the base mesh and 3 uniform refinements) | eᵢ = ‖uᵢ − uᵢ₊₁‖ between successive levels |

Common keys:

| key | default | meaning |
|---|---|---|
| `norms` | `[L2, H1]` | the L² norm and the broken H¹ **seminorm** ‖∇ₕ·‖ |
| `fields` | the model's unknowns and `flux` | any of `d`, `p`, `w`, `flux`, `velocity`. The flux is measured in L² only |
| `at` | `final` | `space`/`time`: `final` (t = T) or `max` (the maximum over all time steps). `self`: `final` or a list of times (the maximum over them) |
| `refine_dt` | `false` | `self` only: halve dt with each refinement |

Observed order between two levels: log(eᵢ/eᵢ₊₁) / log(sᵢ/sᵢ₊₁), where s is the largest cell
diameter h (`space`, `self`) or dt (`time`). Errors are in SI, computed with quadrature of degree `deg + 4`.

### Designing a study

- **`space`:** keep dt small enough that the time error stays below the space error on the finest mesh.
  A plain Crank–Nicolson run with a short T usually does it.
- **Initial state (`space`, and in general with Newmark):** choose `exact` so that the fields and their time derivatives
  vanish at t0, for example with a factor `sin(t)**2`. The run then starts from an exactly representable
  state. A nonzero initial state is only interpolated, and the mismatch with the discrete
  equilibrium keeps oscillating, because Newmark has no damping. It is harmless in d and p, but it costs one order in the velocity
  and the biot3 flux.
- **`time`:** choose `exact` as a polynomial of degree ≤ `deg` in space (for example linear in x). The
  discrete space then contains it, the mesh adds no error, and a 4×4 mesh is enough. Otherwise
  the space error stops the time error from decreasing, and the observed order falls to 0.
- **`self`:** the levels are **nested** (each one is a uniform refinement of the previous), so the
  difference of two levels is computed exactly in L² and in H¹. Every level discretizes the
  **polygon** of the base mesh. On a curved boundary, the polygon's re-entrant vertices (for example
  on a hole) limit the orders to about 2π/ω in L² and π/ω in H¹, where ω = π + 2π/N is the interior angle
  for N edges around the hole. Use a fine enough base mesh.
- **`base:`** (top level, `self` studies): the path of another YAML, relative to the study file. The
  study takes that experiment and overrides it with its own keys. The base's `output` is dropped.

### Expected orders (P_k, smooth solution)

| field | L² | H¹ |
|---|---|---|
| d, w | k+1 | k |
| p, biot2 (P_k) | k+1 | k |
| p, biot3 (P_{k−1}) | k | k−1 |
| flux, biot2 (−κ∇p_h) | k | – |
| flux, biot3 (∂w/∂t) | k+1 | – |

In time: 2 for Crank–Nicolson (θ = 0.5) and for Newmark; 1 for backward Euler (θ = 1).

Examples are in `experiments/convergence/`.

### Python

```python
from biotx import convergence as cv
res = cv.run(cv.load("experiments/convergence/mms_biot2_space.yaml", {"discretization.deg": 2}))
print(res.table()); res.rates("p", "L2"); res.save()
res = cv.run(cv.study(biotx.load("case.yaml"), study="self", levels=3))   # from an Experiment
```

---

## 16. Command line and Python overrides

```bash
python -m biotx.run case.yaml                                  # run
python -m biotx.run case.yaml --set time.dt=0.005 --name fine  # override + output name
mpirun -n 4 python -m biotx.run case.yaml                      # parallel
python -m biotx.compare ref.npz a.npz b.npz --t-min 2          # compare (needs output.fields)
```

- `--set KEY=VALUE` uses dotted keys of the YAML tree. The value is read as YAML, so quote it
  for the shell if it contains brackets or spaces:
  ```bash
  --set 'output.snapshots=[1, 2, 3]'
  --set 'boundaries.pia={pressure: 0, displacement: [0, 0]}'
  --set model=biot3 --set discretization.deg=2 --set params.A=266.6
  ```
- An override **replaces one key** and leaves the rest of the tree as it is. A missing path is created.
- From Python the overrides work the same way:
  ```python
  exp = biotx.load("case.yaml", {"time.dt": 0.005, "geometry.h": 2.5e-4})
  exp2 = exp.with_({"model": "biot3"})
  ```

---

## 17. Errors and pitfalls

These are reported as **errors**:

- a boundary of the mesh with no conditions, or conditions for a name the mesh does not have;
- zero or two mechanical (or hydraulic) conditions on one boundary, or a misspelled condition;
- a vector with the wrong number of components (2 entries on a 3D mesh, and so on);
- unknown keys in `scales`, `material`, `time`, `discretization` or `output`;
- unknown keys in `geometry` for the chosen shape (e.g. `raduis`), or a `rectangle`/`box` without `n` or `h`;
- a source the model does not read (e.g. `fluid_source` with `biot3`) or a misspelled source;
- a param or load named like a built-in (`t`, `x`, `r`, `e`, `pi`, `sin`, …), or a name used both as a param and as a load;
- `mu`, `K` or `muF` ≤ 0, an unknown `model` or geometry `shape`;
- `T − t0` that is not a multiple of `dt`;
- `exact` missing a field of the model, `exact` together with `sources`, a boundary value `exact`
  without an `exact` section, or `normal_traction: exact`;
- a `convergence` section with an unknown study or key, or with fewer than two levels;
- `flux` ≠ 0 with `biot3`;
- `grid` on a 3D mesh, or `slices` on a 2D mesh.

These are **not** detected, so watch for them:

- unknown keys at the top level, which are ignored (`base:` is read only by `biotx.convergence`);
- `r` is measured from the **origin**, not from the centre of a shifted disc or hole. Write
  `sqrt((x - 3e-3)**2 + y**2)` instead;
- probes, lines or slices outside the mesh, which give NaN rather than an error;
- snapshot times farther than dt/2 from every step, which are skipped.

---

## 18. Complete examples

### 2D: every section used

```yaml
model: biot2
scales: {L0: 1.0e-2, T0: 1.0, P0: 1500.0}
material: {E: 4000, nu: 0.4, alpha: 1.0, c0: 1.0e-6, K: 1.0e-11, muF: 3.5e-3, rho: 1000, beta: 0}
geometry: {shape: rectangle, corners: [[0, 0], [2.0e-2, 1.0e-2]], h: 5.0e-4}
params: {A: 500, f: 1.0, t_ramp: 0.2}
loads:
  ramp: "where(t < t_ramp, t / t_ramp, 1)"
  load: "ramp * A * sin(2*pi*f*t)"
boundaries:
  top:    {normal_traction: -load, flux: 0}
  bottom: {displacement: [0, 0], pressure: 0}
  left:   {displacement: [0, 0], flux: 0}
  right:  {traction: [0, 0], flux: 0}
sources:
  body_force: [0, 0]
  fluid_source: 0
discretization: {deg: 2, eta: 10, zeta: 10}
time: {T: 3.0, dt: 0.01, theta: 0.5, startup_steps: 2}
output:
  name: slab
  series: [p, flux, d]
  probes: {mid: [1.0e-2, 5.0e-3]}
  lines: {vertical: {from: [1.0e-2, 0], to: [1.0e-2, 1.0e-2], n: 101}}
  grid: 81
  snapshots: [1.0, 2.0, 3.0]
  fields: true
  boundary_flux: true
  paraview: 10
solver: {mat_mumps_icntl_14: 100}
```

### 3D: the spherical shell with a pulsating cavity

```yaml
model: biot3
scales: {L0: 1.0e-2, T0: 1.0, P0: 1500.0}
material: {mu: 1500, lambda: 11567, alpha: 0.49, c0: 1.0e-6, K: 1.57e-11, muF: 3.5e-3,
           rho: 1000, rho_f: 1000, phi: 0.2, tortuosity: 1.6}
geometry:
  shape: shell
  radius: 1.0e-2
  inner: {center: [3.0e-3, 0, 0], radius: 2.5e-3}
  h: 1.0e-3
  names: {inner: ventricle, outer: pia}
params: {A: 133.3}
loads: {pV: "A * (1 - cos(2*pi*t)) / 2"}
boundaries:
  ventricle: {pressure: pV, normal_traction: -pV}
  pia:       {pressure: 0,  displacement: [0, 0, 0]}
time: {T: 3.0, dt: 0.01}
output:
  probes: {thick: [-5.0e-3, 0, 0]}
  lines: {thin: {from: [5.51e-3, 0, 0], to: [9.99e-3, 0, 0], n: 101}}
  slices: {equator: {normal: z, at: 0.0, n: 81}, sagittal: {normal: y, at: 0.0}}
  snapshots: [2.25, 2.5, 2.75, 3.0]
  fields: true
```
