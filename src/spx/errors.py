"""Error norms (floats only) and the convergence-rate table."""
import numpy as np
import pandas as pd
import ufl

from spx.utils import integrate


def norm_L2(mesh, e, dx) -> float:
    """L2 norm of the expression e over the region of the measure dx."""
    return float(np.sqrt(integrate(mesh, ufl.inner(e, e) * dx)))


def seminorm_H1(mesh, e, dx) -> float:
    """Broken H1 seminorm: grad is taken cell by cell, which is what DG needs."""
    return float(np.sqrt(integrate(mesh, ufl.inner(ufl.grad(e), ufl.grad(e)) * dx)))


def convergence_table(rows: list[dict]) -> pd.DataFrame:
    """Table with one row per mesh; adds a rate_<name> column for every err_<name> column."""
    df = pd.DataFrame(rows)
    for col in [c for c in df.columns if c.startswith("err_")]:
        df["rate_" + col[4:]] = np.log(df[col].shift() / df[col]) / np.log(df["h"].shift() / df["h"])
    return df
