# Biot poroelasticity: theory note

This note states the continuous models, the assumptions behind them, and the discretization implemented in `biotx/`:

- the **two-field** model $(d, p)$, in `biotx/biot2.py`;
- the **three-field** model $(d, w, p)$ with fluid inertia (Morasca 2025), in `biotx/biot3.py`.

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

Expected orders (from the schemes; the tests measure only spatial rates):

- **Two-field:** second order in time with $\theta = 1/2$, first order with $\theta = 1$.
- **Three-field:** second order, from Newmark.

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

**Manufactured solutions** (`tests/test_mms.py`):

- **Exact fields:** smooth, non-polynomial and nonzero on the boundary, with time dependence through a scalar factor.
- **Sources and boundary data:** derived symbolically in UFL from (B1)–(B5) and the boundary conditions. Time derivatives are exact through `ufl.variable(t)`.
- **Coverage:** every boundary condition kind, nonzero Dirichlet data (and its velocity), the sources, and `set_initial`.

Measured $L^2$ convergence rates:

| Model | Elements | $d$ | $w$ | $p$ |
|---|---|---|---|---|
| two-field | P1/P1 | 1.98 | — | 1.93 |
| three-field | P2/P2/P1 | 2.96 | 3.04 | 1.92 |

These are the optimal orders $k+1$ for each field.

**Regression** (`tests/test_regression_spx.py`): on the pulsatile case, biotx reproduces the results of the original spx implementation to round-off ($10^{-10}$ relative). This holds for the two-field model with P1 and P2, and for the three-field model.

---

## 10. Notes from the original implementation

The original code (`sp/variational_forms.py`, before spx) used inconsistent boundary sets for the $p$–$d$ coupling. The momentum-row term was integrated on $\Gamma_d$, but the mass-row term and its data were integrated on $\Gamma_p$ (pressure Dirichlet). Both come from the weakly imposed displacement, so both belong on $\Gamma_d$ (D4). The old test case could not detect this, because it had $\Gamma_d = \Gamma_p$.
