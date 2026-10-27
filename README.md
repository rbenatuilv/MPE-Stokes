# biotx

Two- and three-field Biot poroelasticity in DG (FEniCSx 0.11), driven by YAML experiments, with a
mass-conservative (H(div)) velocity for transport.

```bash
conda activate biotx && pip install -e .                       # once, from the repo root
python -m biotx.run experiments/annulus_biot2.yaml --set 'output.fields=[p, cflux]'
python -m pytest                                                # tests (about 1 minute)
```

| where | what |
|---|---|
| `experiments/README.md` | how to write and run experiments; the studies |
| `experiments/YAML_REFERENCE.md` | every YAML key |
| `docs/theory.md` | the models, the discretization, verification, conditioning, the H(div) velocity |
| `biotx/biot2.py`, `biot3.py` | the two models, each explained at the top |
| `biotx/common.py` | shared DG forms, Newmark, boundary conditions, solver, time loop |
| `biotx/flux.py` | the conservative velocity `cflux`, built after each step |
| `biotx/config.py`, `geometry.py`, `expressions.py` | YAML, meshes with named boundaries, formulas of x and t |
| `biotx/post.py`, `fields.py` | outputs (probes, lines, grids, fields, ParaView), `load_results`, `load_fields` |
| `biotx/mms.py`, `convergence.py`, `conditioning.py`, `compare.py` | manufactured solutions, convergence and condition-number studies, run comparison |
