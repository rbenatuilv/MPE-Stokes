# Applications

An experiment is a YAML file in SI units: model, material, geometry, loads, boundary conditions,
time and outputs. Every key is in [`docs/YAML_REFERENCE.md`](../docs/YAML_REFERENCE.md).

The base problems `brain_annulus/annulus_biot2.yaml` and `brain_annulus/annulus_biot3.yaml` (the same
annulus, with an off-centre hole and a pulsating cavity, for each model) define the problem only. A run
adds what it records.

## Running

```bash
python -m biotx.run applications/brain_annulus/annulus_biot2.yaml --set time.dt=0.005 --set 'output.fields=[p, cflux]'
mpirun -n 4 python -m biotx.run applications/brain_annulus/annulus_biot2.yaml          # parallel
```

```python
import biotx
exp = biotx.load("applications/brain_annulus/annulus_biot2.yaml",
                 {"params.A": 266.6, "output.fields": ["p", "cflux"]})
biotx.simulate(exp)                             # the same as python -m biotx.run

model = biotx.make_model(exp)                   # or step by step, for your own measurements
for t in model.steps():
    q, to_si = model.quantity("cflux")
```

Results go to `<yaml dir>/<output.dir>/<name>.npz` (`biotx.load_results`), full fields to
`<name>_fields.npz` (`biotx.load_fields`: `.function(name, i)`, `.average(name, t_from, t_to)`).
`python -m biotx.compare ref.npz run.npz` compares runs saved with `output.fields`.

**Which velocity:** `flux` is the raw Darcy flux; `cflux` is the mass-conservative one. Give `cflux` to
a transport equation. To replay or average one period, save every step of it:
`--set 'output.snapshots={from: 2.0, to: 3.0}'`.

## The applications

| folder | what | run (from the folder) |
|---|---|---|
| `brain_annulus/` | the base problems: brain tissue around a pulsating cavity (2D) | `python -m biotx.run annulus_biot2.yaml` |
| `brain_annulus/mesh_independence/` | self-convergence of the brain run under mesh refinement | `python -m biotx.convergence annulus_biot2_self.yaml` |
| `brain_annulus/transport/` | Biot runs that save `cflux` over a period, and its period average | `python run_biot.py` |
| `shell_3d/` | the 3D version: a spherical shell with an off-centre cavity (coarse) | `python -m biotx.run shell_biot2.yaml` |

The notebooks load saved results (`results/`, next to the notebook); they recompute what is missing.
