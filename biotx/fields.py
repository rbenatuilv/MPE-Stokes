"""Full fields at the snapshots, saved so that finished runs can be compared or post-processed afterwards.

    output:
      fields: true              # d, p, flux (or a list: [d, p, flux, cflux, w, velocity])
                                # -> <name>_fields.npz next to <name>.npz

The file holds the mesh (coordinates in m) and, for each quantity, the DG degrees of freedom
of every cell at every snapshot (in SI). A DG field is cell-local, so this is the exact
Function, not a sampling of it. Unlike the ParaView files, it can be read back as dolfinx
Functions on any number of ranks:

    fl = load_fields("results/annulus_biot2_fields.npz")
    p = fl.function("p", 0)     # Function on fl.mesh (SI coordinates) at fl.snap_t[0], in Pa
    q = fl.average("cflux", 2.0, 3.0)   # time average over (2, 3] s (cflux saved at every step)

cflux is a Raviart-Thomas field; it is stored in DG of its polynomial degree, which contains it, so
the stored field is exact and its normal component stays continuous across facets (to round-off).

The cells are written in their original (input) order and keep their vertex order, so the
cell dofs of the rebuilt mesh are the ones written (a DG element has no dof permutations).
"""
from pathlib import Path

import basix
import basix.ufl
import dolfinx
import numpy as np
import ufl
import yaml
from mpi4py import MPI

DEFAULT = ["d", "p", "flux"]


class FieldWriter:
    """Collects the DG dofs of some quantities at each call; save() writes the file (rank 0)."""

    def __init__(self, model, names):
        self.model, self.names = model, list(names)
        mesh, comm = model.mesh, model.mesh.comm
        gdim, nloc = mesh.geometry.dim, mesh.topology.index_map(mesh.topology.dim).size_local
        gi = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
        local = (gi, mesh.geometry.x[:, :gdim] * model.exp.scales.L0, gi[mesh.geometry.dofmaps[0][:nloc]])
        parts = comm.gather(local)
        self.mesh_data = None
        if comm.rank == 0:
            gi, x = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
            ids, first = np.unique(gi, return_index=True)            # shared nodes appear twice
            cmap = mesh.geometry.cmaps[0]
            self.mesh_data = {"x": x[first],
                              "cells": np.searchsorted(ids, np.concatenate([p[2] for p in parts])),
                              "cell": mesh.topology.cell_name(), "gdeg": cmap.degree,
                              "gvariant": int(cmap.variant)}
        self.meta, self.data, self.t, self._dg = {}, {n: [] for n in self.names}, [], {}

    def __call__(self, model=None):
        m = model or self.model
        comm = m.mesh.comm
        self.t.append(m.time)
        for name in self.names:
            f, fac = m.quantity(name)
            if not f.function_space.ufl_element().discontinuous:     # cflux (RT): its exact DG copy
                if name not in self._dg:
                    from biotx.post import dg_space
                    self._dg[name] = dolfinx.fem.Function(dg_space(f.function_space))
                from biotx.post import copy_to
                copy_to(self._dg[name], f)
                f = self._dg[name]
            V = f.function_space
            el, bs = V.ufl_element(), V.dofmap.bs
            self.meta[name] = (el.degree, V.value_shape)
            nloc = m.mesh.topology.index_map(m.mesh.topology.dim).size_local
            vals = f.x.array.reshape(-1, bs)[V.dofmap.list[:nloc]].reshape(nloc, -1) * fac
            parts = comm.gather(vals)
            if comm.rank == 0:
                self.data[name].append(np.concatenate(parts))

    def save(self, path):
        if self.model.mesh.comm.rank != 0:
            return None
        out = dict(self.mesh_data, snap_t=np.array(self.t), names=np.array(self.names),
                   experiment=yaml.safe_dump(self.model.exp.raw, sort_keys=False))
        for name in self.names:
            deg, shape = self.meta[name]
            out[name] = np.array(self.data[name])                     # (nsnap, ncells, ndofs)
            out[f"{name}_deg"], out[f"{name}_shape"] = deg, np.array(shape, dtype=int)
        np.savez(path, **out)
        return path


class Fields:
    """A saved _fields.npz read back: mesh (SI), snap_t, names, experiment, function(name, i)."""

    def __init__(self, path, comm=MPI.COMM_WORLD):
        self.path, self.comm = Path(path), comm
        self._file = np.load(self.path) if comm.rank == 0 else None
        f = self._file
        small = None
        if comm.rank == 0:
            small = {k: f[k] for k in ("cell", "gdeg", "gvariant", "snap_t", "names")}
            small["experiment"] = str(f["experiment"])
            small["nn"], small["gdim"] = f["cells"].shape[1], f["x"].shape[1]
            for n in small["names"]:
                small[f"{n}_deg"] = int(f[f"{n}_deg"])
                small[f"{n}_shape"] = tuple(int(v) for v in f[f"{n}_shape"])
        small = comm.bcast(small)
        self.snap_t, self.names = small["snap_t"], [str(n) for n in small["names"]]
        self.experiment = yaml.safe_load(small["experiment"])
        self._meta = {n: (small[f"{n}_deg"], small[f"{n}_shape"]) for n in self.names}

        cell, gdim = str(small["cell"]), int(small["gdim"])
        if comm.rank == 0:                         # rank 0 reads, the partitioner distributes
            x, cells = f["x"], f["cells"].astype(np.int64)
        else:
            x, cells = np.zeros((0, gdim)), np.zeros((0, small["nn"]), dtype=np.int64)
        e = ufl.Mesh(basix.ufl.element("Lagrange", cell, int(small["gdeg"]), shape=(gdim,),
                                       lagrange_variant=basix.LagrangeVariant(int(small["gvariant"]))))
        self.mesh = dolfinx.mesh.create_mesh(comm, cells, e, x)
        nloc = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        orig = np.asarray(self.mesh.topology.original_cell_index, dtype=np.int64)[:nloc]
        self._orig = comm.gather(orig)                            # rank 0: rows of each rank
        self._spaces, self._cache = {}, {}

    def space(self, name):
        if name not in self._spaces:
            deg, shape = self._meta[name]
            self._spaces[name] = dolfinx.fem.functionspace(self.mesh, ("DG", deg, shape))
        return self._spaces[name]

    def function(self, name, i):
        """The quantity `name` at snapshot i, as a new Function (SI)."""
        return self._function(name, lambda data: data[i])

    def average(self, name, t_from, t_to):
        """The mean of `name` over the snapshots with t_from < t <= t_to, as a new Function (SI).
        For cflux saved at every step of a window, the snapshot at t is the velocity over the step
        that ends at t, so this is exactly its time average over [t_from, t_to]."""
        tol = 1e-9 * max(1.0, abs(t_to))
        sel = (self.snap_t > t_from + tol) & (self.snap_t <= t_to + tol)
        if not sel.any():
            raise ValueError(f"average: no snapshot in ({t_from}, {t_to}]")
        return self._function(name, lambda data: data[sel].mean(axis=0))

    def _function(self, name, pick):
        V = self.space(name)
        parts = None
        if self.comm.rank == 0:
            if name not in self._cache:
                self._cache = {name: self._file[name]}            # keep one quantity in memory
            vals = pick(self._cache[name])
            parts = [vals[o] for o in self._orig]
        mine = self.comm.scatter(parts)
        u = dolfinx.fem.Function(V, name=name)
        nloc, bs = len(mine), V.dofmap.bs
        u.x.array.reshape(-1, bs)[V.dofmap.list[:nloc]] = mine.reshape(nloc, -1, bs)
        u.x.scatter_forward()
        return u


def load_fields(path, comm=MPI.COMM_WORLD) -> Fields:
    return Fields(path, comm)
