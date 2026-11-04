# Verification

Properties of the method itself, independent of any application: orders of convergence, condition
numbers, and the fluid balance of each velocity field. Every YAML key is in
[`docs/YAML_REFERENCE.md`](../docs/YAML_REFERENCE.md); how to write and run an experiment is in
[`applications/README.md`](../applications/README.md).

| folder | what | run (from the folder) |
|---|---|---|
| `convergence/` | orders in space and time, with manufactured solutions on squares | `python -m biotx.convergence <study>.yaml` |
| `conditioning/` | condition numbers of the 2F and 3F matrices (raw, Jacobi, Schur) | `python -m biotx.conditioning <study>.yaml` |
| `balance/` | fake fluid made by each velocity field, cell by cell (`biotx.flux.CellBalance`) | `python balance.py ../../applications/brain_annulus/annulus_biot2.yaml` |

`conditioning/` and `balance/` run in the application regime on purpose: their studies extend the brain
annulus (`base: ../../applications/brain_annulus/annulus_biot2.yaml`), because κ ≈ 1e-9 is what makes the
conditioning and the fake fluid matter.

Each folder has a notebook with the results. The notebooks load saved results (`results/`, next to the
notebook); set `RERUN = True` to recompute them.
