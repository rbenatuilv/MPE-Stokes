# Experiments

An experiment is a YAML file in SI units: model, material, geometry, loads, boundary conditions,
time and outputs. Every key is in [`YAML_REFERENCE.md`](YAML_REFERENCE.md).

The base problems `annulus_biot2.yaml` and `annulus_biot3.yaml` (the same annulus, with an off-centre
hole and a pulsating cavity, for each model) define the problem only. A run adds what it records.

## Running

```bash
python -m biotx.run experiments/annulus_biot2.yaml --set time.dt=0.005 --set 'output.fields=[p, cflux]'
mpirun -n 4 python -m biotx.run experiments/annulus_biot2.yaml          # parallel
```

```python
import biotx
exp = biotx.load("experiments/annulus_biot2.yaml", {"params.A": 266.6, "output.fields": ["p", "cflux"]})
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

## Studies

| folder | what | run |
|---|---|---|
| `convergence/` | orders in space and time (manufactured solutions), self-convergence | `python -m biotx.convergence convergence/<study>.yaml` |
| `conditioning/` | condition numbers of the 2F and 3F matrices (raw, Jacobi, Schur) | `python -m biotx.conditioning conditioning/<study>.yaml` |
| `balance/` | fake fluid made by each velocity field, cell by cell | `python balance.py ../annulus_biot2.yaml` |
| `transport/` | Biot runs that save `cflux` over a period, and its period average | `python run_biot.py` |

Each folder has a notebook with the results. The notebooks load saved results; set `RERUN = True` to
recompute them.
