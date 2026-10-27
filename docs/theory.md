# Biot poroelasticity: theory note

This note states the continuous models, the assumptions behind them, and the discretization implemented in `biotx/`:

- the **two-field** model $(d, p)$, in `biotx/biot2.py`;
- the **three-field** model $(d, w, p)$ with fluid inertia in `biotx/biot3.py`.

The shared pieces are in `biotx/common.py`. The equation tags (B1), (D2) and so on are the ones used in the documentation of the code. This note grew out of the former `spx` package, a coupled Stokes–Biot code archived in `~/thesis/archive/spx_2026-10-26.tar.gz`. Only its Biot part is described here.

---

## 1. Domain and notation

- $\Omega \subset \mathbb{R}^d$, $d \in \{2,3\}$, is the poroelastic domain. $n$ is the outward unit normal on $\partial\Omega$.
- $\varepsilon(v) = \tfrac12(\nabla v + \nabla v^T)$, and $I$ is the identity.
- The boundary is made of **named** pieces (`boundaries:` in the YAML). Each piece carries exactly one mechanical and one hydraulic condition, which splits $\partial\Omega$ twice:

| Set | YAML key | Field | Condition |
|---|---|---|---|
| $\Gamma_d$ | `displacement` | $d$ | Dirichlet: $d = g_d$ |
| $\Gamma_t$ | `traction`, `normal_traction` | $d$ | Neumann: total traction |
| $\Gamma_p$ | `pressure` | $p$ | Dirichlet: $p = g_p$ |
| $\Gamma_q$ | `flux` | $p$ (or $w$) | Neumann: outward Darcy flux $q\cdot n = g_q$ |

$\Gamma_d \cup \Gamma_t = \Gamma_p \cup \Gamma_q = \partial\Omega$. The mechanical split and the hydraulic split are independent of each other.

---

## 2. Unknowns and parameters

| Symbol | Code | Meaning | Model |
|---|---|---|---|
| $d$ | `d` | solid displacement | both |
| $p$ | `p` | pore pressure | both |
| $w$ | `w` | relative fluid displacement $\phi(u_f - u_s)$ | three-field |
| $z = \partial_t d$, $a = \partial_{tt} d$ | `sd.z_old`, `sd.a_old` | solid velocity, acceleration (derived) | both |
| $q$ | `model.flux()` | Darcy flux (relative fluid volume flux) | both |

| Symbol | YAML key (`material`) | Meaning |
|---|---|---|
| $\lambda$, $\mu$ | `lambda`, `mu` (or `E`, `nu`) | Lamé parameters of the drained skeleton |
| $\alpha$ | `alpha` | Biot–Willis coefficient |
| $c_0$ | `c0` | constrained specific storage |
| $K$, $\mu_F$ | `K`, `muF` | intrinsic permeability (isotropic, scalar), fluid viscosity |
| $\kappa = K/\mu_F$ | — | hydraulic mobility |
| $\rho$ | `rho` | density of the saturated medium |
| $\beta$ | `beta` | linear leakage to a reservoir at zero pressure (two-field) |
| $\rho_F$, $\phi$, $\tau$ | `rho_f`, `phi`, `tortuosity` | fluid density, porosity, tortuosity (three-field) |
| $\rho_W = \rho_F\,\tau/\phi$ | — | apparent mass of the relative fluid motion (three-field) |

Stresses:

$$
\sigma_e(d) = 2\mu\,\varepsilon(d) + \lambda(\nabla\cdot d)\,I,
\qquad
\sigma^{tot} = \sigma_e(d) - \alpha\,p\,I .
$$

---

## 3. Modelling assumptions

1. **Solid:** small deformations (linear kinematics), isotropic linear elasticity, fully saturated, with a single pore-fluid network.
2. **Pore flow:**
   - **Two-field:** Darcy's law, $q = -\kappa\nabla p$, eliminated in favour of $p$.
   - **Three-field:** Darcy's law with fluid inertia. The flux $q = \partial_t w$ is kept as an unknown through $w$.
3. **Fixed geometry:** all equations are posed on the reference configuration. The deformation does not change the domain, $\phi$ or $\kappa$.
4. **Constant coefficients** in space and time. Body forces, gravity included, enter only through the sources.
5. **Exchange term (two-field):** $\beta p$ models linear leakage to a reservoir at zero pressure. Setting $\beta = 0$ removes it.

---

## 4. Strong form

### 4.1 Two-field model $(d, p)$

Find $(d, p)$ for $t \in (t_0, T]$ such that

$$
\rho\,\partial_{tt} d - \nabla\cdot\big(\sigma_e(d) - \alpha\,p I\big) = f \tag{B1}
$$

$$
c_0\,\partial_t p + \alpha\,\nabla\cdot\partial_t d - \nabla\cdot(\kappa\nabla p) + \beta\,p = g \tag{B2}
$$

with boundary conditions

$$
d = g_d \ \text{on } \Gamma_d,\qquad \big(\sigma_e(d) - \alpha p I\big) n = t_N \ \text{on } \Gamma_t,
$$

$$
p = g_p \ \text{on } \Gamma_p,\qquad q\cdot n = -\kappa\nabla p\cdot n = g_q \ \text{on } \Gamma_q .
$$

Here $g_q$ is the **outward** Darcy flux: $g_q > 0$ means fluid leaves $\Omega$. `normal_traction: s` means $t_N = s\,n$, so a fluid at pressure $P$ pushing on the wall is $s = -P$.

YAML sources: $f$ is `body_force` [N/m³] and $g$ is `fluid_source` [1/s]. After the solve, the flux is recovered as $q_h = -\kappa\nabla p_h$.

### 4.2 Three-field model $(d, w, p)$

With $w = \phi(u_f - u_s)$, the Darcy flux is $q = \partial_t w$. Find $(d, w, p)$ such that

$$
\rho\,\partial_{tt} d + \rho_F\,\partial_{tt} w - \nabla\cdot\big(\sigma_e(d) - \alpha\,p I\big) = f_d \tag{B3}
$$

$$
\rho_F\,\partial_{tt} d + \rho_W\,\partial_{tt} w + \kappa^{-1}\partial_t w + \nabla p = f_w \tag{B4}
$$

$$
c_0\,p + \alpha\,\nabla\cdot d + \nabla\cdot w = f_p \tag{B5}
$$

- (B4) is Darcy's law with inertia. With $\rho_F = 0$ it reduces to $\partial_t w = -\kappa\nabla p$, and the model reduces to the two-field one with $\beta = 0$.
- (B5) is (B2) integrated in time from a state at rest, with $\nabla\cdot q = \nabla\cdot\partial_t w$. That is why it contains no time derivative.
- Sources: `body_force` $f_d$ and `fluid_force` $f_w$ [N/m³]. `mass_source` $f_p$ [–] is the injected volume fraction **accumulated** since $t_0$.

Boundary conditions: the mechanical ones are as in 4.1.

- On $\Gamma_p$, $p = g_p$ enters naturally through the $\nabla p$ term of (B4).
- On $\Gamma_q$, only $w\cdot n = 0$ (an impermeable wall, $g_q = 0$) is implemented.

### 4.3 Initial conditions

The run starts from rest at $t_0$: $d = \partial_t d = 0$, $p = 0$, and in the three-field model also $w = \partial_t w = 0$.

- The initial acceleration is taken as zero.
- Another initial state can be given from Python (`model.set_initial`).

---

## 5. Non-dimensionalization

All input is in SI. The reference length $L_0$, time $T_0$ and stress $P_0$ (`scales:`) define the problem that is actually solved:

$$
x^* = x/L_0,\quad t^* = t/T_0,\quad p^* = p/P_0,\quad d^* = d/L_0,\quad w^* = w/L_0,\quad q^* = q\,T_0/L_0 ,
$$

$$
\lambda^* = \lambda/P_0,\ \ \mu^* = \mu/P_0,\ \ c_0^* = c_0 P_0,\ \ \kappa^* = \kappa\,\frac{P_0 T_0}{L_0^2},\ \ \rho^* = \rho\,\frac{L_0^2}{P_0 T_0^2}\ (\text{same for } \rho_F, \rho_W),\ \ \beta^* = \beta P_0 T_0 .
$$

- **The equations keep their form** in starred variables. The scales change only the conditioning of the discrete problem, not the solution.
- **Where this happens:** `config.Scales` and `Material.scaled`.
- **Results are converted back to SI** on output.
- **The rest of this note uses non-dimensional quantities without stars.**

---

## 6. Time discretization

Uniform step $\Delta t$, $t^n = t_0 + n\Delta t$. Superscripts denote time levels. Unknowns without a superscript are at $t^{n+1}$.

### 6.1 Newmark-β for $d$ (and $w$)

With parameters $(\beta_N, \gamma_N)$ and the displacement as the unknown:

$$
a^{n+1} = \frac{d - d^n}{\beta_N\Delta t^2} - \frac{z^n}{\beta_N\Delta t} - \frac{1-2\beta_N}{2\beta_N}\,a^n \tag{T1}
$$

$$
z^{n+1} = \frac{\gamma_N}{\beta_N\Delta t}(d - d^n) - \Big(\frac{\gamma_N}{\beta_N} - 1\Big) z^n - \Delta t\Big(\frac{\gamma_N}{2\beta_N} - 1\Big) a^n \tag{T2}
$$

- **How (T2) is obtained:** it is $z^{n+1} = z^n + \Delta t[(1-\gamma_N)a^n + \gamma_N a^{n+1}]$ with (T1) substituted.
- **Both are affine in the unknown $d$.** Inside the forms they are UFL expressions of the trial function (`NewmarkField.a_new`, `z_new`). After the solve they are evaluated on the dof arrays (`NewmarkField.advance`).
- **The three-field model** applies the same formulas to $w$. The flux $q = \partial_t w$ is the Newmark velocity of $w$.
- **Defaults** $\beta_N = 1/4$, $\gamma_N = 1/2$ (average acceleration): second order, unconditionally stable, no numerical damping.

### 6.2 θ-method for the two-field mass equation

(B2) is discretized with the θ-method. The momentum equation (B1) is taken at $t^{n+1}$:

$$
\frac{c_0}{\Delta t}(p - p^n) + \alpha\,\nabla\cdot z_\theta + \theta\,\mathcal{A}p + (1-\theta)\,\mathcal{A}p^n = \theta\,g^{n+1} + (1-\theta)\,g^n,
\qquad z_\theta = \theta z^{n+1} + (1-\theta) z^n , \tag{T3}
$$

where $\mathcal{A}p = -\nabla\cdot(\kappa\nabla p) + \beta p$.

- **Choice of θ:** $\theta = 1$ is backward Euler (first order). $\theta = 1/2$ is Crank–Nicolson (second order).
- **Data at $t^n$:** boundary and source data are re-evaluated at $t^n$ by substituting $t \to t^n$ in their UFL expressions (`at_old_time`).
- **Rannacher start:** with `startup_steps` $= m > 0$, the first $m$ steps use $\theta = 1$, which damps the oscillations Crank–Nicolson produces after a sudden load. The scheme then switches to the requested θ. The matrix is refactorised once at the switch.

The three-field model has no time derivative in its mass equation (B5), so it needs no θ.

### 6.3 Accuracy

- **Two-field:** second order in time with $\theta = 1/2$, first order with $\theta = 1$.
- **Three-field:** second order, from Newmark.

Both are confirmed by the convergence tests (Section 9).

---

## 7. Spatial discretization (symmetric interior penalty DG)

### 7.1 Spaces

On a conforming simplicial mesh $\mathcal{T}_h$ of $\Omega$, all fields are discontinuous ($\mathbb{P}_k^{disc}$, `DG` in the code):

| Model | $d$ | $w$ | $p$ | default |
|---|---|---|---|---|
| two-field | $[\mathbb{P}_k]^d$ | — | $\mathbb{P}_{k_p}$ | $k_p = k$ |
| three-field | $[\mathbb{P}_k]^d$ | $[\mathbb{P}_k]^d$ | $\mathbb{P}_{k_p}$ | $k_p = k-1$ (P2/P2/P1 is the stable choice) |

The unknowns form one mixed space (`basix.ufl.mixed_element`), so each step is a single linear system.

### 7.2 Facet sets and DG operators

- $\mathcal{F}_I$: interior facets.
- Facet size: $h_F = 2h^+h^-/(h^+ + h^-)$ (harmonic mean of the two cell diameters) on interior facets, and the cell diameter $h$ on boundary facets.
- Averages and jumps on an interior facet:

$$
\{w\} = \tfrac12(w^+ + w^-),\quad
[\![v]\!] = \mathrm{sym}(v^+\otimes n^+ + v^-\otimes n^-),\quad
[\![v\cdot n]\!] = v^+\cdot n^+ + v^-\cdot n^-,
$$

$$
[\![q]\!] = q^+ - q^-,\quad [\![q\,n]\!] = q^+n^+ + q^-n^-.
$$

  On a boundary facet: $\{w\} = w$, $[\![v]\!] = \mathrm{sym}(v\otimes n)$, $[\![q]\!] = q$ and $[\![q\,n]\!] = q\,n$.
- $(\cdot,\cdot)$ is the $L^2(\Omega)$ product, taken cellwise for broken gradients. $\langle\cdot,\cdot\rangle_X$ is the $L^2$ product on a set of facets $X$.
- Every integral uses quadrature of degree $2k+1$.

### 7.3 Forms

**Elasticity** (`common.elasticity`), with penalty $\eta(\lambda + 2\mu)k^2/h_F$:

$$
a_E(d,v) = (\sigma_e(d),\varepsilon(v))
- \sum_{F\in\mathcal{F}_I\cup\Gamma_d}\Big(\langle\{\sigma_e(d)\},[\![v]\!]\rangle_F + \langle\{\sigma_e(v)\},[\![d]\!]\rangle_F\Big)
+ \sum_{F\in\mathcal{F}_I\cup\Gamma_d}\Big\langle\frac{\eta(\lambda+2\mu)k^2}{h_F}[\![d]\!],[\![v]\!]\Big\rangle_F \tag{D2}
$$

**Darcy** (`TwoField.darcy`), with penalty $\zeta\tilde\kappa k^2/h_F$ and $\tilde\kappa = \max(\kappa, 1)$ in non-dimensional units:

$$
a_D(p,q) = (\kappa\nabla p,\nabla q)
- \sum_{F\in\mathcal{F}_I\cup\Gamma_p}\Big(\langle\kappa\{\nabla p\},[\![q\,n]\!]\rangle_F + \langle\kappa\{\nabla q\},[\![p\,n]\!]\rangle_F\Big)
+ \sum_{F\in\mathcal{F}_I\cup\Gamma_p}\Big\langle\frac{\zeta\tilde\kappa k^2}{h_F}[\![p]\!],[\![q]\!]\Big\rangle_F
+ (\beta p, q) \tag{D3}
$$

**Pressure–displacement coupling** (`common.b_P`), the DG form of $(\nabla p, v)$. It is used in both the momentum row and the mass row:

$$
b_P(q,v) = -(q,\nabla\cdot v) + \sum_{F\in\mathcal{F}_I}\langle\{q\},[\![v\cdot n]\!]\rangle_F + \langle q, v\cdot n\rangle_{\Gamma_d}. \tag{D4}
$$

The boundary term of $b_P$ lives on $\Gamma_d$, the displacement Dirichlet set, in both rows.

**Pressure–filtration coupling** (`ThreeField.b_W`, three-field), the DG form of $(\nabla p, w)$:

$$
b_W(q,w) = -(q,\nabla\cdot w) + \sum_{F\in\mathcal{F}_I}\langle\{q\},[\![w\cdot n]\!]\rangle_F + \langle q, w\cdot n\rangle_{\Gamma_q}. \tag{D5}
$$

**Normal-jump penalty on $w$** (three-field). It enforces, weakly, the continuity of $w\cdot n$ across facets and $w\cdot n = 0$ on $\Gamma_q$:

$$
j_W(w,v) = \sum_{F\in\mathcal{F}_I}\Big\langle\frac{\eta(\lambda+2\mu)k^2}{h_F}[\![w\cdot n]\!],[\![v\cdot n]\!]\Big\rangle_F
+ \Big\langle\frac{\eta(\lambda+2\mu)k^2}{h}\,w\cdot n,\ v\cdot n\Big\rangle_{\Gamma_q}. \tag{D6}
$$

### 7.4 Fully discrete two-field problem

At each step, find $(d, p)$ such that, for all test functions $(v, q)$, with $z^{n+1}$ and $a^{n+1}$ given by (T1)–(T2) in terms of $d$ and $z_\theta$ as in (T3):

$$
\begin{aligned}
&\rho\,(a^{n+1}, v) + a_E(d,v) + \alpha\, b_P(p,v) = \ell_M(v),\\[4pt]
&\tfrac{c_0}{\Delta t}(p - p^n, q) - \alpha\, b_P(q, z_\theta) + \theta\,a_D(p,q) + (1-\theta)\,a_D(p^n,q)
= \theta\,\ell_E^{n+1}(q) + (1-\theta)\,\ell_E^{n}(q),
\end{aligned}
$$

with the right-hand sides (Dirichlet data enter weakly, by Nitsche's method, through the same SIPG terms):

$$
\begin{aligned}
\ell_M(v) &= (f, v) + \langle t_N, v\rangle_{\Gamma_t}
- \langle \sigma_e(v), \mathrm{sym}(g_d\otimes n)\rangle_{\Gamma_d}
+ \Big\langle\frac{\eta(\lambda+2\mu)k^2}{h}\mathrm{sym}(g_d\otimes n), \mathrm{sym}(v\otimes n)\Big\rangle_{\Gamma_d},\\
\ell_E(q) &= (g, q) - \langle g_q, q\rangle_{\Gamma_q}
- \langle\kappa\nabla q\cdot n, g_p\rangle_{\Gamma_p}
+ \Big\langle\frac{\zeta\tilde\kappa k^2}{h} g_p, q\Big\rangle_{\Gamma_p}
- \alpha\,\langle q, \partial_t g_d\cdot n\rangle_{\Gamma_d}.
\end{aligned}
$$

- **The coupling is antisymmetric.** The coupling appears as $+\alpha b_P(p, v)$ in the momentum row and $-\alpha b_P(q, \partial_t d)$ in the mass row, so the discrete coupling is exactly energy-neutral.
- **The mass-row coupling is the DG form of $\alpha(\nabla\cdot\partial_t d, q)$.** It consists of $-\alpha\, b_P(q, z_\theta)$ together with $-\alpha\,\langle q,\partial_t g_d\cdot n\rangle_{\Gamma_d}$ from $\ell_E$.
- **$\partial_t g_d$ is exact.** It is computed by symbolic differentiation of the YAML data (`Expressions.ddt`).

### 7.5 Fully discrete three-field problem

At each step, find $(d, w, p)$ such that, for all test functions $(v, v_w, q)$, with $a_d^{n+1}$, $a_w^{n+1}$ and $z_w^{n+1}$ given by (T1)–(T2) for $d$ and $w$ respectively:

$$
\begin{aligned}
&\rho\,(a_d^{n+1}, v) + \rho_F\,(a_w^{n+1}, v) + a_E(d,v) + \alpha\, b_P(p,v) = \ell_M(v),\\[4pt]
&\rho_F\,(a_d^{n+1}, v_w) + \rho_W\,(a_w^{n+1}, v_w) + \kappa^{-1}(z_w^{n+1}, v_w) + b_W(p, v_w) + j_W(w, v_w)
= (f_w, v_w) - \langle g_p, v_w\cdot n\rangle_{\Gamma_p},\\[4pt]
&c_0\,(p, q) - \alpha\, b_P(q, d) - b_W(q, w) = (f_p, q) - \alpha\,\langle q, g_d\cdot n\rangle_{\Gamma_d}.
\end{aligned}
$$

- **The pressure boundary condition is natural.** $p = g_p$ enters only through $-\langle g_p, v_w\cdot n\rangle_{\Gamma_p}$, which comes from integrating $\nabla p$ by parts in (B4).
- **The flux boundary condition is penalised.** $w\cdot n = 0$ on $\Gamma_q$ is imposed by the boundary term of $j_W$, and $b_W$ drops the pressure there.
- **The couplings are again antisymmetric.** $b_P$ and $b_W$ enter the momentum and mass rows with opposite signs.
- **The scheme computes a single-valued normal flux $\{w\}\cdot n$ on each interior facet.** That flux is not the normal component of $w_h$ itself: $w_h\cdot n$ is only approximately continuous, through $j_W$.

### 7.6 Algebraic system

- **One matrix, factorised once.** $\Delta t$ and the coefficients are constant, so the matrix $A$ is assembled and LU-factorised (MUMPS) once (`LinearSolver`). Each step only reassembles the right-hand side and back-substitutes. The only exception is the switch of θ at the end of a Rannacher start.
- **Linear by construction.** The residual of each step is affine in the unknowns: (T1)–(T2) are affine and the data are known. So `ufl.lhs` and `ufl.rhs` split it exactly into $A$ and the right-hand side.
- **$A$ is non-symmetric**, because the coupling blocks are antisymmetric.
- **No strongly imposed Dirichlet conditions**, so there is no lifting step.

---

## 8. Design choices and their rationale

| Choice | Reason | Alternative |
|---|---|---|
| DG for all fields | local conservation; natural weak boundary conditions; every degree of freedom belongs to one cell, which simplifies parallel assembly and field checkpoints | continuous elements: fewer degrees of freedom, but no element-wise conservation |
| P2/P2/P1 in the three-field model | inf-sup stable pair for $(w, p)$; equal orders lose one rate in $p$ | $k_p = k$ with a pressure-jump stabilization |
| Penalty scaling $k^2/h_F$ | standard SIPG scaling; correct for $k > 1$ | — |
| Penalty constants $\eta = \zeta = 10$ | SIPG is coercive only for large enough penalties; 10 is enough for every 2D case tested. For the 3D Stokes problem, spx needed $\geq 40$ on tetrahedra, so check this in 3D. | penalty $\propto k(k+d-1)$ with an explicit trace-inequality constant |
| $\tilde\kappa = \max(\kappa,1)$ in the Darcy penalty | keeps a minimum penalty on pressure jumps when $\kappa \ll 1$, where a purely DG pressure would otherwise decouple cell by cell | $\tilde\kappa = \kappa$ (consistent scaling) |
| Non-dimensional solve (Section 5) | brings the coefficients to $O(1)$ (brain tissue: $\kappa \sim 10^{-9}$ m²/(Pa·s) in SI) so that penalties and the LU factorisation are well conditioned | solving in SI |
| Quadrature degree $2k+1$ for the system, $k+4$ for error norms | exact for the bilinear forms with affine cells; error norms must not be polluted by quadrature | — |

---

## 9. Verification

**Manufactured solutions** (`biotx/mms.py`, `tests/test_convergence.py`):

- **Exact fields:** given in the `exact:` section. Every other input is derived from them symbolically in UFL:
  - the sources, by applying the strong form (B1)–(B2) or (B3)–(B5) (`strong_form` in each model);
  - the data of every boundary condition marked `exact`;
  - the initial state.

  Time derivatives are exact through `ufl.variable(t)`.
- **Coverage:** every boundary condition kind, nonzero Dirichlet data (and its velocity), and a non-rest initial state.
- **Errors:** the $L^2$ norm and the broken $H^1$ seminorm $\|\nabla_h(u - u_h)\|$, with quadrature of degree $k+4$ (`biotx/convergence.py`).

Measured orders in space. The exact fields are smooth and non-polynomial, with a time factor $\sin^2 t$, so that the fields and $\partial_t$ vanish at $t_0$. Crank–Nicolson with a small $\Delta t$; $L^2$ orders on the finest meshes:

| Model | Elements | $d$ | $w$ | $p$ |
|---|---|---|---|---|
| two-field | P1/P1 | 1.98 | — | 1.93 |
| three-field | P2/P2/P1 | 2.96 | 3.04 | 1.92 |

These are the optimal orders $k+1$ for each field, with $k_p + 1$ for the three-field pressure. The $H^1$ orders are one less.

Why the time factor vanishes at $t_0$:

- **The initial state is only interpolated.** An exact solution that is nonzero at $t_0$ is interpolated into the discrete spaces. That differs from the discrete equilibrium by $O(h^{k+1})$.
- **Newmark keeps the mismatch.** Newmark with $(\tfrac14, \tfrac12)$ does not damp it, so it keeps oscillating at the mesh frequencies $\sim h^{-1}$.
- **Effect on $d$:** it stays $O(h^{k+1})$, so no order is lost.
- **Effect on velocities:** it becomes $O(h^k)$. The three-field flux $\partial_t w$ then converged at about 2.3 instead of 3, and it got *worse* as $\Delta t$ decreased.
- **With $\sin^2 t$:** the initial state is exactly zero, and every field reaches its optimal order. The flux reaches 2.9–3.0, as expected.

Measured orders in time. The exact fields are linear in $x$, so the discrete spaces contain them and only the time error remains. $\Delta t = 0.1 \to 0.0125$, all fields and both norms:

| Model | Scheme | order |
|---|---|---|
| two-field | Newmark + Crank–Nicolson ($\theta = 1/2$) | 2.00 |
| two-field | Newmark + backward Euler ($\theta = 1$) | 0.98–1.00 |
| three-field | Newmark | 2.00 |

**Self-convergence** (no exact solution): the base mesh is refined uniformly, and the difference $e_i = \|u_i - u_{i+1}\|$ of successive levels is measured.

- **Nested meshes.** The meshes are nested, so a coarse DG field lies exactly in the fine space, and $e_i$ is computed exactly in both norms. Its order equals the order of the true error.
- **Curved boundaries.** Every level discretizes the polygon of the base mesh. At a vertex of a polygonal hole the interior angle is $\omega = \pi + 2\pi/N$ for $N$ edges, a re-entrant corner. This limits the orders to about $2\pi/\omega$ in $L^2$ and $\pi/\omega$ in $H^1$, whatever the degree.
  - **Measured:** on the annulus with $N = 10$ cavity edges, $p$ converges at 1.68 in $L^2$ and 0.79 in $H^1$. The predictions are 1.67 and 0.83.
  - **Consequence:** a self-convergence study on a curved domain needs a fine base mesh, and it measures the convergence on the polygon.

**Regression** (`tests/test_regression_spx.py`): on the pulsatile case, biotx reproduces the results of the original spx implementation to round-off ($10^{-10}$ relative). This holds for the two-field model with P1 and P2, and for the three-field model.

---

## 10. Conditioning of the step matrix

Each step solves $A x = b$ with the same matrix $A$ (Section 7.6). Its 2-norm condition number

$$
\mathrm{cond}_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
$$

bounds how much relative perturbations of $b$ (round-off, data) can grow in $x$. It also governs the convergence of Krylov solvers, which is what matters once the direct LU no longer fits (large 3D meshes). It is computed by `biotx/conditioning.py`.

- **Singular values, not eigenvalues.** $A$ is not symmetric: the coupling blocks are antisymmetric. Its eigenvalues therefore say nothing about $\mathrm{cond}_2$, and both ends of the singular spectrum are computed with SLEPc:
  - $\sigma_{\max}$: the SLEPc SVD solver (Lanczos);
  - $\sigma_{\min} = \lambda_{\max}\big((A^TA)^{-1}\big)^{-1/2}$: the largest eigenvalue of the symmetric operator $A^{-1}A^{-T}$. Each Lanczos iteration needs one solve with $A^T$ and one with $A$, both from the same MUMPS factorisation.

  On small systems both values agree with a dense SVD to the solver tolerance (`tests/test_conditioning.py`).
- **Which matrix.** The one of a regular step. In the two-field model with a Rannacher start, that is the matrix with the requested $\theta$.
- **Raw and Jacobi-scaled.** $\mathrm{cond}_2$ depends on the units of each block: the two models have different unknowns, and the brain coefficients differ by orders of magnitude ($\kappa^* \approx 7\cdot10^{-5}$, $c_0^* = 1.5\cdot10^{-3}$). Besides the raw $A$, the study therefore reports $\tilde A = |D|^{-1/2} A\, |D|^{-1/2}$ with $D = \mathrm{diag}(A)$. That symmetric diagonal scaling removes most of the effect of units. What remains reflects the structure of the problem: mesh, time step, coupling.
- **Expected growth in $h$.** For SIPG forms, $\mathrm{cond}_2 \sim h^{-2}$. On unstructured meshes the growth is measured against the effective size $h_{\mathrm{eff}} = (|\Omega|/N_{\mathrm{cells}})^{1/d}$, which follows the refinement better than the largest cell diameter.
- **Schur-scaled.** In the three-field model the pressure diagonal is only $c_0 M_p$, and the pressure is controlled mainly through its coupling to $w$, a saddle-point structure. Eliminating $w$ gives the Schur complement $S = C + B A_{ww}^{-1} B^T \approx c_0 M_p + \kappa\Delta t\,(-\Delta_h)$ (up to the Newmark factor $\beta/\gamma$). Jacobi scaling uses $c_0$ alone, so it under-scales $p$: the scaled $w$–$p$ coupling grows like $h^{-1}\sqrt{\kappa\Delta t/c_0}$, and $\mathrm{cond}_2(\tilde A) \sim h^{-3}$. The `schur` scaling divides each pressure unknown instead by $s_{ii} = |a_{ii}| + \sum_{j\notin p} |a_{ij} a_{ji}|/|a_{jj}|$, the diagonal of $S$ with $A_{ww}$ replaced by its diagonal, which bounds the scaled coupling by 1. It is the diagonal analogue of the block preconditioner $\mathrm{diag}(A_{dd}, A_{ww}, S)$, and the fair basis for comparing the two models. In the two-field model it coincides with Jacobi to three digits, because the pressure diagonal already holds the Darcy jump penalty.
- **Measured** (annulus, brain parameters, P2, $h_{\mathrm{eff}}$ = 0.32 mm, `experiments/conditioning/conditioning.ipynb`): two-field $\sim h^{-2}$ for every scaling, nearly independent of $\Delta t$. Three-field raw 30–100 times larger; Jacobi $\sim h^{-3}$; Schur $\sim h^{-2}$ and 0.17–0.55 times the two-field value.
- **Which unknowns.** The right singular vector of $\sigma_{\min}$ shows which fields make $A$ nearly singular: the share of its squared norm on the dofs of $d$, $w$ and $p$.
- **Units and the Darcy penalty.** All the forms are invariant under the non-dimensionalization of Section 5, except the Darcy penalty of the two-field model. Its scale $\tilde\kappa = \max(\kappa, 1)$ is taken in the units of the solve, so solving in SI changes the discrete problem, not only its units. With the brain parameters, $\tilde\kappa = 1$ in both cases, which is about $1.5\cdot10^4$ times the consistent value $\kappa^*$ in the scaled problem and about $2\cdot10^8$ times $\kappa$ in SI.

---

## 11. Conservative velocity (H(div) reconstruction)

A transport equation $\partial_t c + \nabla\cdot(q\,c) - \mu\Delta c = f$ needs a velocity that conserves the fluid cell by cell; otherwise the mismatch acts as a solute source.

**What is and is not conservative.**
- **The scheme is conservative with its numerical flux $\hat F$.** Testing the mass equation with the indicator of a cell $K$ (an allowed DG test function) gives, over one step, $\text{gained}_K + \int_{\partial K}\hat F = 0$ exactly. $\hat F$ has one value per facet, which is what makes the balance close.
  - Two-field: $-\kappa\{\nabla p\}\cdot n + \tfrac{\zeta\tilde\kappa k^2}{h_F}[\![p]\!]$, θ-weighted, with its Nitsche form on $\Gamma_p$ and $g_q$ on $\Gamma_q$.
  - Three-field: $\{\Delta w\}\cdot n/\Delta t$.
- **The raw velocities are not.** $-\kappa\nabla p_h$ and $\dot w_h$ are cellwise fields whose normal components differ on the two sides of a facet.
  - On the brain annulus, $-\kappa\nabla p_h$ leaves about 0.3 % of the throughput unbalanced, and up to 20 % in the worst cells.
  - $(w^{n+1} - w^n)/\Delta t$ leaves $4\cdot10^{-6}$; the end-of-step $\dot w^{n+1}$ adds a time error about 1000 times larger.
  - See `experiments/balance/`.

**H(div)** is the space of vector fields with a square-integrable divergence. For piecewise polynomials, that means the normal component is continuous across facets. It is not a divergence-free condition: here $\nabla\cdot u$ is the rate at which the cells release fluid.

**The reconstruction** (`biotx/flux.py`, after each step, without touching the solver) is the Raviart–Thomas field $u$, with normal traces of degree $k$ ($k = k_p$ for two-field, $k$ = deg for three-field), defined by

$$\int_F (u\cdot n)\,\mu = \int_F \hat F\,\mu \quad \forall \mu \in \mathbb P_k(F), \qquad
\int_K u\cdot\psi = \int_K \bar q\cdot\psi + L_K(\psi) \quad \forall \psi \in [\mathbb P_{k-1}(K)]^d .$$

Here $\bar q$ is the step-consistent raw velocity: $\theta q^{n+1} + (1-\theta) q^n$ for two-field, and $(w^{n+1} - w^n)/\Delta t$ for three-field. $L_K$ lifts the symmetric SIPG term into the cell: in the two-field model, $L_K(\psi) = \sum_{F\subset\partial K}\int_F \omega\,\kappa\,(\psi\cdot n_K)[\![p_\theta]\!]$, with $\omega = 1/2$ on interior facets and 1 on $\Gamma_p$. In the three-field model $L_K = 0$. With this lifting,

$$(\nabla\cdot u, q)_K = -\text{gained}(q) \quad \forall q \in \mathbb P_{k_p}(K),$$

that is, the divergence is the solver's fluid balance against every pressure test function, not only on cell averages. That is what a higher-order DG transport needs to keep a uniform concentration uniform. This is the construction of Ern, Nicaise and Vohralík (2007) and of Bastian and Rivière (2003).

**Use.** The output quantity `cflux` (YAML `output.series` / `output.fields`) is this field, and `cbflux_<boundary>` its flow through each boundary. Saving every step of a period (`snapshots: {from, to}`) and `Fields.average` give the replayed or period-averaged velocity for a transport solve.

**Computation.**
- The facet and interior unknowns decouple. The facet block is block diagonal by facet and the interior block by cell.
- Both are constant, so they are factorised once, and each step costs two right-hand sides.
- Serial only for now.

**Verified** (`tests/test_flux.py`, `experiments/balance/`):
- The facet flows equal $\hat F$ and the normal jumps vanish, to round-off.
- The divergence identity holds to round-off.
- The reconstruction converges at the order of the raw flux, $k_p$ for two-field and deg + 1 for three-field, with nearly the same error.
- On the brain annulus the fake fluid drops to $10^{-13}$ (two-field) and $10^{-14}$ (three-field) of the throughput.

---

## 12. Notes from the original implementation

The original code (`sp/variational_forms.py`, before spx) used inconsistent boundary sets for the $p$–$d$ coupling. The momentum-row term was integrated on $\Gamma_d$, but the mass-row term and its data were integrated on $\Gamma_p$ (pressure Dirichlet). Both come from the weakly imposed displacement, so both belong on $\Gamma_d$ (D4). The old test case could not detect this, because it had $\Gamma_d = \Gamma_p$.
