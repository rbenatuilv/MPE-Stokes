# Coupled Stokes–poroelasticity model: theory note

This note states the continuous model, the assumptions behind it, and the discretization implemented in `src/spx/`. Equation tags such as (S1) are cited in the code comments.

---

## 1. Domain and notation

- $\Omega = \Omega_S \cup \Omega_P \subset \mathbb{R}^d$, $d \in \{2,3\}$: $\Omega_S$ is the fluid region and $\Omega_P$ the poroelastic region.
- $\Sigma = \partial\Omega_S \cap \partial\Omega_P$ is the interface. $n_S$ is the unit normal on $\Sigma$ pointing out of $\Omega_S$, and $n_P = -n_S$.
- $n$ is the outward unit normal on $\partial\Omega$. $\tau$ is any unit tangent on $\Sigma$, and $w_\tau = w - (w\cdot n_S)\,n_S$ is the tangential part of $w$.
- $\varepsilon(w) = \tfrac12(\nabla w + \nabla w^T)$, and $I$ is the identity.
- The outer boundary is split, per field, into Dirichlet and Neumann parts:

| Set | Code name | Field | Condition |
|---|---|---|---|
| $\Gamma_S^D$ | `dirS` | $u_S$ | Dirichlet (velocity) |
| $\Gamma_S^N$ | `neuS` | $u_S$ | Neumann (traction) |
| $\Gamma_P^D$ | `dirP` | $d$ | Dirichlet (displacement) |
| $\Gamma_P^N$ | `neuP` | $d$ | Neumann (total traction) |
| $\Gamma_E^D$ | `dirE` | $p_P$ | Dirichlet (pressure) |
| $\Gamma_E^N$ | `neuE` | $p_P$ | Neumann (flux) |

$\Gamma_S^D \cup \Gamma_S^N = \partial\Omega_S\setminus\Sigma$, $\Gamma_P^D \cup \Gamma_P^N = \partial\Omega_P\setminus\Sigma$, and $\Gamma_E^D \cup \Gamma_E^N = \partial\Omega_P\setminus\Sigma$. The displacement split and the pressure split are independent of each other.

**Interface side convention (enforced in `measures.py`):** on every interface facet, the `'+'` side is the $\Omega_S$ cell and the `'-'` side is the $\Omega_P$ cell. So $n_S = n^+$ and $n_P = n^-$.

---

## 2. Unknowns and parameters

| Symbol | Code | Meaning | Defined on |
|---|---|---|---|
| $u_S$ | `uS` | fluid velocity | $\Omega_S$ |
| $p_S$ | `pS` | fluid pressure | $\Omega_S$ |
| $d$ | `d` | solid displacement | $\Omega_P$ |
| $p_P$ | `pP` | pore pressure | $\Omega_P$ |
| $z = \partial_t d$, $a = \partial_{tt} d$ | `z_old`, `a_old` | solid velocity, acceleration (derived) | $\Omega_P$ |

| Symbol | Config key | Meaning |
|---|---|---|
| $\rho_S$, $\mu_S$ | `rhoS`, `muS` | fluid density, dynamic viscosity |
| $\rho_P$ | `rhoP` | density of the saturated poroelastic medium |
| $\lambda_P$, $\mu_P$ | `lambdaP`, `muP` | Lamé parameters of the drained skeleton |
| $\alpha$ | `alpha` | Biot–Willis coefficient |
| $c_0$ | `c0` | specific storage coefficient |
| $\mu_F$ | `muF` | pore-fluid dynamic viscosity |
| $K$ | `K` | intrinsic permeability (isotropic, scalar) |
| $\kappa = K/\mu_F$ | — | hydraulic mobility |
| $\beta$ | `beta` | linear leakage coefficient (exchange with an external reservoir at zero pressure) |
| $\gamma_{BJS}$ | `gammaBJS` | Beavers–Joseph–Saffman slip coefficient |
| $B = \gamma_{BJS}\,\mu_S/\sqrt{K}$ | — | BJS friction coefficient |

Stresses:

$$
\sigma_S(u_S,p_S) = 2\mu_S\,\varepsilon(u_S) - p_S I,
\qquad
\sigma_e(d) = 2\mu_P\,\varepsilon(d) + \lambda_P(\nabla\cdot d)\,I,
\qquad
\sigma_P^{tot} = \sigma_e(d) - \alpha\,p_P I .
$$

---

## 3. Modelling assumptions

1. **Fluid:** incompressible, Newtonian, with constant density and viscosity. The flow is slow enough to drop the convective term (unsteady Stokes, not Navier–Stokes).
2. **Solid:** small deformations (linear kinematics), isotropic linear elasticity, fully saturated, with a single pore-fluid network.
3. **Pore flow:** Darcy's law with isotropic permeability, $u_D = -\kappa\nabla p_P$. The Darcy velocity is eliminated, leaving the two-field Biot formulation $(d, p_P)$.
4. **Fixed geometry:** all equations are posed on the reference configuration, and $\Sigma$ does not move. The deformation is not fed back into the fluid domain.
5. **Interface:** mass conservation, balance of normal stress and total stress, and the Beavers–Joseph–Saffman slip condition.
6. **Constant coefficients** in space and time. Body forces, gravity included, enter only through $f_S$ and $f_P$.
7. **Exchange term:** $\beta p_P$ models linear leakage to a reservoir at zero pressure. Setting $\beta = 0$ removes it.

---

## 4. Strong form

Find $(u_S, p_S, d, p_P)$ for $t \in (t_0, T]$ such that:

**Fluid region $\Omega_S$:**

$$
\rho_S\,\partial_t u_S - \nabla\cdot\sigma_S(u_S,p_S) = f_S \tag{S1}
$$

$$
\nabla\cdot u_S = 0 \tag{S2}
$$

**Poroelastic region $\Omega_P$:**

$$
\rho_P\,\partial_{tt} d - \nabla\cdot\big(\sigma_e(d) - \alpha\,p_P I\big) = f_P \tag{B1}
$$

$$
c_0\,\partial_t p_P + \alpha\,\nabla\cdot\partial_t d - \nabla\cdot(\kappa\nabla p_P) + \beta\,p_P = g_P \tag{B2}
$$

**Interface $\Sigma$:**

$$
u_S\cdot n_S + \big(\partial_t d - \kappa\nabla p_P\big)\cdot n_P = 0 \qquad\text{(mass conservation)} \tag{I1}
$$

$$
-(\sigma_S n_S)\cdot n_S = p_P \qquad\text{(normal stress)} \tag{I2}
$$

$$
-(\sigma_S n_S)_\tau = B\,(u_S - \partial_t d)_\tau \qquad\text{(BJS)} \tag{I3}
$$

$$
\sigma_S n_S + \big(\sigma_e(d) - \alpha\,p_P I\big)\,n_P = 0 \qquad\text{(total stress balance)} \tag{I4}
$$

**Boundary conditions:**

$$
u_S = g_S \ \text{on } \Gamma_S^D,\qquad \sigma_S n = h_S \ \text{on } \Gamma_S^N,
$$

$$
d = g_P \ \text{on } \Gamma_P^D,\qquad \big(\sigma_e(d) - \alpha p_P I\big) n = h_P \ \text{on } \Gamma_P^N,
$$

$$
p_P = g_E \ \text{on } \Gamma_E^D,\qquad \kappa\nabla p_P\cdot n = \phi \ \text{on } \Gamma_E^N .
$$

Here $\phi = -u_D\cdot n$ is the Darcy flux *into* $\Omega_P$.

**Initial conditions** at $t = t_0$: $u_S$, $d$, $\partial_t d$ and $p_P$ are given. $\partial_{tt} d(t_0)$ follows from (B1) at $t_0$. The pressure $p_S$ needs no initial condition.

**Remarks.**

- (I2) and (I4) together say that the normal fluid stress equals the pore pressure, and that the tangential fluid traction is carried by the skeleton.
- With $\Gamma_S^N = \emptyset$, $p_S$ is still fixed through (I2), provided $\Sigma \neq \emptyset$. The verification configs keep $\Gamma_S^N \neq \emptyset$ anyway.

---

## 5. Weak form: how the interface conditions enter

There is no Lagrange multiplier on $\Sigma$. Integrating by parts in each subdomain leaves boundary integrals on $\Sigma$, into which the interface conditions are substituted.

- **Fluid momentum (test $v_S$).** The term $-\int_\Sigma (\sigma_S n_S)\cdot v_S$ is split into its normal and tangential parts, and (I2) and (I3) are substituted:
$$
-\int_\Sigma (\sigma_S n_S)\cdot v_S = \int_\Sigma p_P\, v_S\cdot n_S + \int_\Sigma B\,(u_S - \partial_t d)_\tau\cdot v_{S,\tau}.
$$
- **Solid momentum (test $v_P$).** By (I4), $\sigma_P^{tot} n_P = -\sigma_S n_S$. Substituting (I2) and (I3) then gives
$$
-\int_\Sigma (\sigma_P^{tot} n_P)\cdot v_P = \int_\Sigma p_P\, v_P\cdot n_P - \int_\Sigma B\,(u_S - \partial_t d)_\tau\cdot v_{P,\tau}.
$$
The coefficient of $p_P$ is $1$, not $\alpha$, because the interface balances total stress.
- **Darcy (test $q_P$).** By (I1), $\kappa\nabla p_P\cdot n_P = u_S\cdot n_S + \partial_t d\cdot n_P$, so
$$
-\int_\Sigma \kappa\nabla p_P\cdot n_P\, q_P = -\int_\Sigma (u_S\cdot n_S + \partial_t d\cdot n_P)\, q_P .
$$
- The fluid mass equation (S2) and the $p_S$ terms produce no integral on $\Sigma$: substituting (I2) replaces the whole traction $\sigma_S n_S$, including its $-p_S n_S$ part.

The couplings $p_P \leftrightarrow u_S$ enter with opposite signs in the two rows, so **the resulting system is non-symmetric**.

---

## 6. Time discretization

Uniform step $\Delta t$, $t^n = t_0 + n\Delta t$. Superscripts denote time levels. Unknowns without a superscript are at $t^{n+1}$.

- $u_S$ and $p_P$: backward Euler, $\partial_t u_S \approx (u_S - u_S^n)/\Delta t$ and $\partial_t p_P \approx (p_P - p_P^n)/\Delta t$.
- $d$: Newmark-$\beta$ with parameters $(\beta_N, \gamma_N)$ and $d$ as the unknown:

$$
a^{n+1} = \frac{d - d^n}{\beta_N\Delta t^2} - \frac{z^n}{\beta_N\Delta t} - \frac{1-2\beta_N}{2\beta_N}\,a^n \tag{T1}
$$

$$
z^{n+1} = \frac{\gamma_N}{\beta_N\Delta t}(d - d^n) - \Big(\frac{\gamma_N}{\beta_N} - 1\Big) z^n - \Delta t\Big(\frac{\gamma_N}{2\beta_N} - 1\Big) a^n \tag{T2}
$$

(T2) is $z^{n+1} = z^n + \Delta t[(1-\gamma_N)a^n + \gamma_N a^{n+1}]$ with (T1) substituted. Every $\partial_t d$ in (B2), (I1) and (I3) is replaced by $z^{n+1}$, and $\partial_{tt} d$ in (B1) by $a^{n+1}$.

Properties:

- With $\beta_N = 1/4$, $\gamma_N = 1/2$ (average acceleration), Newmark is second order and unconditionally stable.
- Backward Euler is first order, so **the coupled scheme is first order in $\Delta t$**.
- $\Delta t$ is constant, so the system matrix is the same at every step.

---

## 7. Spatial discretization (symmetric interior penalty DG)

### 7.1 Spaces

On a conforming simplicial mesh $\mathcal{T}_h$ of $\Omega$ that resolves $\Sigma$:

$$
V_S = [\mathbb{P}_k^{disc}(\mathcal{T}_h^S)]^d,\quad Q_S = \mathbb{P}_{k'}^{disc}(\mathcal{T}_h^S),\quad
V_P = [\mathbb{P}_k^{disc}(\mathcal{T}_h^P)]^d,\quad Q_P = \mathbb{P}_k^{disc}(\mathcal{T}_h^P).
$$

Each space is a DG space on the whole mesh, restricted (multiphenicsx `DofMapRestriction`) to the degrees of freedom of its own subdomain. Defaults: $k = k' = 1$.

### 7.2 Facet sets and DG operators

- $\mathcal{F}_S$, $\mathcal{F}_P$: interior facets with both neighbours in $\Omega_S$ (resp. $\Omega_P$). $\mathcal{F}_\Sigma$: interface facets.
- Facet size: $h_F = 2h^+h^-/(h^+ + h^-)$ on interior facets, $h_F = h_K$ on boundary facets, and $h_{min} = \min(h^+, h^-)$.
- Averages and jumps on an interior facet:

$$
\{w\} = \tfrac12(w^+ + w^-),\quad
[\![v]\!] = \mathrm{sym}(v^+\otimes n^+ + v^-\otimes n^-),\quad
[\![v\cdot n]\!] = v^+\cdot n^+ + v^-\cdot n^-,
$$

$$
[\![q]\!] = q^+ - q^-,\quad [\![q\,n]\!] = q^+n^+ + q^-n^-.
$$

  (The code averages with $\{\cdot\}$ via `ufl.avg`.) On a boundary facet: $\{w\} = w$, $[\![v]\!] = \mathrm{sym}(v\otimes n)$, $[\![q]\!] = q$ and $[\![q\,n]\!] = q\,n$.
- $(\cdot,\cdot)_X$ is the $L^2$ product on $X$, taken cellwise for broken gradients. $\langle\cdot,\cdot\rangle_X$ is the $L^2$ product on a set of facets.

### 7.3 Bilinear forms

Stokes viscous form, with penalty $\gamma_V$:

$$
a_S(u,v) = (2\mu_S\varepsilon(u),\varepsilon(v))_{\Omega_S}
- \sum_{F\in\mathcal{F}_S\cup\Gamma_S^D}\langle 2\mu_S\{\varepsilon(u)\}, [\![v]\!]\rangle_F
- \sum_{F\in\mathcal{F}_S\cup\Gamma_S^D}\langle 2\mu_S\{\varepsilon(v)\}, [\![u]\!]\rangle_F
+ \sum_{F\in\mathcal{F}_S\cup\Gamma_S^D}\Big\langle \frac{\gamma_V\mu_S k^2}{h_F}[\![u]\!], [\![v]\!]\Big\rangle_F \tag{D1}
$$

Elasticity form: the same structure with $\sigma_e$ in place of $2\mu_S\varepsilon$, on $\mathcal{F}_P\cup\Gamma_P^D$, and penalty $\eta(\lambda_P + 2\mu_P)k^2/h_F$:

$$
a_E(d,w) = (\sigma_e(d),\varepsilon(w))_{\Omega_P}
- \sum_{F\in\mathcal{F}_P\cup\Gamma_P^D}\Big(\langle\{\sigma_e(d)\},[\![w]\!]\rangle_F + \langle\{\sigma_e(w)\},[\![d]\!]\rangle_F\Big)
+ \sum_{F\in\mathcal{F}_P\cup\Gamma_P^D}\Big\langle\frac{\eta(\lambda_P+2\mu_P)k^2}{h_F}[\![d]\!],[\![w]\!]\Big\rangle_F \tag{D2}
$$

Darcy form, with $\tilde K = \max(K, 1)$:

$$
a_D(p,q) = (\kappa\nabla p,\nabla q)_{\Omega_P}
- \sum_{F\in\mathcal{F}_P\cup\Gamma_E^D}\Big(\langle\kappa\{\nabla p\},[\![q\,n]\!]\rangle_F + \langle\kappa\{\nabla q\},[\![p\,n]\!]\rangle_F\Big)
+ \sum_{F\in\mathcal{F}_P\cup\Gamma_E^D}\Big\langle\frac{\zeta\tilde K k^2}{\mu_F h_F}[\![p]\!],[\![q]\!]\Big\rangle_F
+ (\beta p, q)_{\Omega_P} \tag{D3}
$$

Pressure–velocity coupling. One definition per subdomain is used for both the momentum row and the mass row:

$$
b_S(q,v) = -(q,\nabla\cdot v)_{\Omega_S} + \sum_{F\in\mathcal{F}_S}\langle\{q\},[\![v\cdot n]\!]\rangle_F + \langle q, v\cdot n\rangle_{\Gamma_S^D},
$$

$$
b_P(q,w) = -(q,\nabla\cdot w)_{\Omega_P} + \sum_{F\in\mathcal{F}_P}\langle\{q\},[\![w\cdot n]\!]\rangle_F + \langle q, w\cdot n\rangle_{\Gamma_P^D}. \tag{D4}
$$

The boundary term of $b_P$ lives on $\Gamma_P^D$, the displacement Dirichlet set, in both rows.

Stokes pressure-jump stabilization, needed because $k' = k$ (equal order):

$$
s_S(p,q) = \sum_{F\in\mathcal{F}_S}\Big\langle\frac{\gamma_P h_{min}}{k'}[\![p\,n]\!],[\![q\,n]\!]\Big\rangle_F. \tag{D5}
$$

### 7.4 Fully discrete problem

At each step, find $(u_S, p_S, d, p_P) \in V_S\times Q_S\times V_P\times Q_P$ such that, for all $(v_S, q_S, v_P, q_P)$, with $z^{n+1}$ and $a^{n+1}$ given by (T1) and (T2) in terms of $d$:

$$
\begin{aligned}
&\tfrac{\rho_S}{\Delta t}(u_S, v_S)_{\Omega_S} + a_S(u_S,v_S) + b_S(p_S,v_S)
+ \langle p_P^-, v_S^+\cdot n_S\rangle_\Sigma
+ \langle B(u_S^+ - z^{n+1,-})_\tau, v_{S,\tau}^+\rangle_\Sigma
= \tfrac{\rho_S}{\Delta t}(u_S^n, v_S)_{\Omega_S} + \ell_S(v_S),\\[4pt]
&-b_S(q_S,u_S) + s_S(p_S,q_S) = \ell_{q}(q_S),\\[4pt]
&\rho_P(a^{n+1}, v_P)_{\Omega_P} + a_E(d,v_P) + \alpha\, b_P(p_P,v_P)
+ \langle p_P^-, v_P^-\cdot n_P\rangle_\Sigma
- \langle B(u_S^+ - z^{n+1,-})_\tau, v_{P,\tau}^-\rangle_\Sigma
= \ell_P(v_P),\\[4pt]
&\tfrac{c_0}{\Delta t}(p_P - p_P^n, q_P)_{\Omega_P} - \alpha\, b_P(q_P, z^{n+1}) + a_D(p_P,q_P)
- \langle u_S^+\cdot n_S + z^{n+1,-}\cdot n_P,\ q_P^-\rangle_\Sigma
= \ell_E(q_P).
\end{aligned}
$$

The terms $-\alpha\, b_P(q_P, z^{n+1})$ and $-\alpha\,\langle q_P,\partial_t g_P\cdot n\rangle_{\Gamma_P^D}$ (in $\ell_E$ below) together are the DG form of $\alpha\,(\nabla\cdot\partial_t d, q_P)$.

Right-hand sides. Dirichlet data enter weakly (Nitsche) through the same SIPG terms:

$$
\begin{aligned}
\ell_S(v) &= (f_S, v)_{\Omega_S} + \langle h_S, v\rangle_{\Gamma_S^N}
- \langle 2\mu_S\varepsilon(v), \mathrm{sym}(g_S\otimes n)\rangle_{\Gamma_S^D}
+ \Big\langle\frac{\gamma_V\mu_S k^2}{h_F}\mathrm{sym}(g_S\otimes n), \mathrm{sym}(v\otimes n)\Big\rangle_{\Gamma_S^D},\\
\ell_q(q) &= -\langle q, g_S\cdot n\rangle_{\Gamma_S^D},\\
\ell_P(w) &= (f_P, w)_{\Omega_P} + \langle h_P, w\rangle_{\Gamma_P^N}
- \langle \sigma_e(w), \mathrm{sym}(g_P\otimes n)\rangle_{\Gamma_P^D}
+ \Big\langle\frac{\eta(\lambda_P+2\mu_P)k^2}{h_F}\mathrm{sym}(g_P\otimes n), \mathrm{sym}(w\otimes n)\Big\rangle_{\Gamma_P^D},\\
\ell_E(q) &= (g_P, q)_{\Omega_P} + \langle\phi, q\rangle_{\Gamma_E^N}
- \langle\kappa\nabla q\cdot n, g_E\rangle_{\Gamma_E^D}
+ \Big\langle\frac{\zeta\tilde K k^2}{\mu_F h_F} g_E, q\Big\rangle_{\Gamma_E^D}
- \alpha\,\langle q, \partial_t g_P\cdot n\rangle_{\Gamma_P^D}.
\end{aligned}
$$

In the steady problem (used as a test and for "Stationary" initial conditions), drop every term with $\Delta t$, $\rho_S$, $\rho_P$, $c_0$, $z$ or $a$.

### 7.5 Algebraic system

Unknown ordering $(u_S, p_S, d, p_P)$ gives a $4\times4$ block system $A\,x^{n+1} = F^n$:

- $A$ is assembled once, since $\Delta t$ and the coefficients are constant. Only $F^n$ changes from step to step.
- $A$ is non-symmetric (Section 5). The Stokes rows use the convention $\begin{bmatrix} A_{uu} & B^T\\ -B & S\end{bmatrix}$.
- There are no strongly imposed Dirichlet conditions, so there is no lifting step.
- Default solver: sparse LU (MUMPS). Iterative alternative: FGMRES with a field-split preconditioner.

---

## 8. Design choices and their rationale

| Choice | Reason | Alternative |
|---|---|---|
| DG for all fields | local conservation; a clean restriction to subdomains (every DOF belongs to one cell); natural weak BCs and interface coupling | continuous elements (Taylor–Hood in $\Omega_S$): about 3–4× fewer DOFs, but a different method |
| Equal order $k' = k$ + jump stabilization (D5) | reproduces the previous implementation | $k' = k-1$ with $\gamma_P = 0$, inf-sup stable without stabilization |
| Penalty scaling $k^2/h_F$ | standard SIPG scaling; correct for $k > 1$ | — |
| Penalty constants $\gamma_V = \eta = \zeta = 10$ in 2D, $\geq 40$ in 3D ($\gamma_P = 10$) | SIPG is coercive only for large enough penalties; on tetrahedra 10 is too small (verified: the 3D Stokes error grows under refinement with 10, converges with 40) | penalty $\propto k(k+d-1)$ with an explicit trace-inequality constant |
| $\tilde K = \max(K,1)$ in the Darcy penalty | keeps a minimum penalty on $p_P$ jumps when $K \ll 1$, where a pure-DG pressure would otherwise decouple cell by cell | $\tilde K = K$ (consistent scaling); compare on small-$K$ cases |
| No Lagrange multiplier on $\Sigma$ | the interface conditions are substituted into the natural boundary terms; fewer unknowns | multiplier for the normal flux/stress |
| Quadrature degree $2\max(k,k')+1$ for the system, $k+4$ for error norms | exact for the bilinear forms with affine cells; error norms must not be polluted by quadrature | — |

---

## 9. Verification with manufactured solutions

Exact fields $(u_S, p_S, d, p_P)$ are chosen smooth, non-polynomial and nonzero on the boundary. The sources $f_S$, $f_P$, $g_P$ and the boundary data come from (S1)–(B2) and the boundary conditions, derived symbolically in UFL. $u_S$ is built divergence-free (as a curl), so (S2) holds exactly. Time dependence is a scalar factor $T(t)$, with time derivatives taken exactly through `ufl.variable(t)`.

The exact fields are not required to satisfy (I1)–(I4). Define the interface residuals of the exact solution:

$$
r_n = -(\sigma_S n_S)\cdot n_S - p_P,\qquad
r_\tau = \big(-\sigma_S n_S - B(u_S - \partial_t d)\big)_\tau,
$$

$$
r_b = \sigma_S n_S + \sigma_P^{tot} n_P,\qquad
r_m = u_S\cdot n_S + (\partial_t d - \kappa\nabla p_P)\cdot n_P .
$$

Substituting "condition $= r$" in place of "condition $= 0$" in Section 5 adds the following known terms on $\Sigma$ to the right-hand sides:

$$
\text{fluid momentum: } -\langle r_n, v_S\cdot n_S\rangle_\Sigma - \langle r_\tau, v_{S,\tau}\rangle_\Sigma,
$$

$$
\text{solid momentum: } -\langle r_n, v_P\cdot n_P\rangle_\Sigma + \langle r_\tau, v_{P,\tau}\rangle_\Sigma + \langle r_b, v_P\rangle_\Sigma,
$$

$$
\text{Darcy: } -\langle r_m, q_P\rangle_\Sigma .
$$

For physical runs all residuals are zero and these terms are omitted.

Expected rates for $k = k' = 1$ (smooth solutions):

- $u_S$, $d$, $p_P$: order $k+1$ in $L^2$ and order $k$ in the broken $H^1$ seminorm.
- $p_S$: at least order $k$ in $L^2$.
- In time: order 1.

---

## 10. Differences from the previous implementation

These were found while recovering the model from the previous code (`sp/variational_forms.py`, `sp/postprocess.py`). The new implementation follows Sections 4–7.

1. **Stokes mass term.** The previous code had $\rho_S/\Delta t\,(u_S, v_S)$ on the left-hand side but $\rho_S\,(u_S^n, v_S)$ on the right-hand side. That is inconsistent with backward Euler (Section 6). Its convergence output shows $\|u_S - u_{S,h}\|_{L^2} \approx \|u_S\|_{L^2}$.
2. **Boundary set of the $p_P$–$d$ coupling.** The momentum-row term used $\Gamma_P^D$, while the mass-row term and its data used $\Gamma_E^D$. Both derive from the weakly imposed displacement, so both belong on $\Gamma_P^D$ (D4). The previous test case could not detect this, because it set $\Gamma_P^D = \Gamma_E^D$.
3. **Scaling and solver tolerance.** In test mode the $p_S$ block was scaled by $10^{-11}$, and GMRES with a relative tolerance of $10^{-6}$ appears to have stopped before $p_S$ converged ($\|p_S - p_{S,h}\|_{L^2} \approx 5\times10^8$). To be confirmed with a direct solve.
4. **Interface orientation.** The `'+'`/`'-'` side assignment on $\Sigma$ was assumed, not enforced. It is now enforced (Section 1).
