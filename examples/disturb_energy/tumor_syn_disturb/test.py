"""
Cancer Drug Resistance SDE Control Benchmark
==============================================

Continuous-time stochastic differential equation model for controlled
cancer therapy under drug resistance, adapted from:

    X. Sun, J. Bao, and Y. Shao,
    "Mathematical Modeling of Therapy-induced Cancer Drug Resistance:
     Connecting Cancer Mechanisms to Population Survival Rates,"
    Scientific Reports, 6:22498, 2016.


Overview
--------
A heterogeneous tumor comprises two interacting cell populations:
drug-sensitive cells (C_S) and drug-resistant cells (C_R). Targeted
therapy (e.g., BRAF inhibitors for melanoma) kills sensitive cells but
simultaneously triggers microenvironment adaptations that promote
resistant cell growth.  The controller must choose a drug dosage
schedule that drives the total tumor burden below a safe threshold
(goal) while preventing it from exceeding a dangerous threshold
(unsafe), despite stochastic fluctuations in cell populations and
uncertain tumor growth parameters.


System Dynamics
---------------
The state is  x = (C_S, C_R)  representing the relative numbers
(in units of 10^8 cells) of drug-sensitive and drug-resistant cancer
cells.  The control input is  u = D >= 0, the administered drug
concentration (normalized).

The continuous-time SDE is

    dx = f(x, u; lambda) dt  +  g(x) dw(t),

with  w(t) in R^2  a standard Brownian motion.


### Drift  f(x, u; lambda)

    f_1 = r_S * C_S * (1 - (C_S + C_R) / T_max)     [logistic growth]
          - u_mut * C_S                                [sensitive -> resistant transition]
          - d_S(u) * C_S                               [drug-induced death]

    f_2 = r_R * C_R * (1 - (C_S + C_R) / T_max)     [logistic growth]
          + u_mut * C_S                                [gain from transition]

where the drug-induced death rate follows Michaelis--Menten kinetics:

    d_S(u) = d_S_max * u / (K_drug + u).


### Diffusion  g(x)

    g(x) = diag( sigma_1 * C_S,  sigma_2 * C_R )

The noise is multiplicative and proportional to cell count, reflecting
that stochastic fluctuations scale with population size (demographic
noise).


Term-by-Term Description
------------------------
Eq. 1  (drug-sensitive cells, C_S):

    Term 1 — Logistic growth:
        r_S * C_S * (1 - (C_S + C_R) / T_max)
        Sensitive cells proliferate at rate r_S, limited by a shared
        carrying capacity T_max that represents nutrient/space limits.

    Term 2 — Mutation-driven transition:
        - u_mut * C_S
        A fraction u_mut of sensitive cells acquire resistance through
        genetic or epigenetic mutations and become resistant cells.

    Term 3 — Drug-induced death:
        - d_S(u) * C_S
        The drug kills sensitive cells at a rate that saturates with
        drug concentration u via Michaelis--Menten kinetics.  Higher
        dosage yields diminishing returns and increased side effects.

    Term 4 — Diffusion:
        sigma_1 * C_S * dW_1
        Stochastic fluctuation in the sensitive cell count, scaling
        with population size (demographic noise).

Eq. 2  (drug-resistant cells, C_R):

    Term 1 — Logistic growth:
        r_R * C_R * (1 - (C_S + C_R) / T_max)
        Resistant cells proliferate at rate r_R, sharing the same
        carrying capacity.  Typically r_R <= r_S (resistant cells
        grow more slowly in the absence of drug pressure).

    Term 2 — Gain from transition:
        + u_mut * C_S
        Resistant cells are gained at the same rate as they are lost
        from the sensitive population.

    Term 3 — Diffusion:
        sigma_2 * C_R * dW_2
        Stochastic fluctuation in the resistant cell count.

Note: The original model (Sun et al.) also includes Poisson-driven
metastasis and angiogenesis equations.  This benchmark focuses on the
core two-population SDE (Eqs. 1--2), which captures the fundamental
tension between drug efficacy and resistance emergence.


Uncertainty Model
-----------------
The growth rates (r_S, r_R) and/or the mutation rate (u_mut) can be
treated as set-valued unknown parameters:

    lambda = (r_S, r_R, u_mut)  in  Lambda  (a compact box).

The drift is affine in lambda for each fixed (x, u):

    f(x, u; lambda) = f_0(x, u) + F(x, u) * lambda,

so Proposition 1 of the main paper applies, yielding the robust
generator in closed form without Lambda-partitioning.

Alternatively, the carrying capacity T_max can be treated as uncertain,
modeling unknown nutrient supply or vascularization level.


Control Input
-------------
    u = D  in  [0, D_max],     scalar drug dosage.

The controller must balance:
  - High dosage:  kills sensitive cells quickly, but the Michaelis--
    Menten saturation means marginal benefit decreases, while
    transition to resistant cells continues and side-effect costs grow.
  - Low dosage:   sensitive cells survive and grow, potentially
    overwhelming the patient before resistance even matters.

This trade-off makes the problem non-trivial: aggressive treatment
clears sensitive cells but leaves a pure resistant population with
no therapeutic recourse.


Reach-Avoid Specification
-------------------------
The total tumor burden is  C_total = C_S + C_R.

    State space:
        X  = { (C_S, C_R) :  C_S >= 0,  C_R >= 0,
                              C_S + C_R <= C_max }

    Initial set (diagnosis):
        X_0 = { (C_S, C_R) :  C_S in [C_S0_lo, C_S0_hi],
                               C_R in [C_R0_lo, C_R0_hi] }
        Representing a newly diagnosed patient with a predominantly
        sensitive tumor and a small resistant subpopulation.

    Goal set (remission):
        X_g = { (C_S, C_R) :  C_S + C_R <= C_remission }
        Total tumor burden is driven below the remission threshold.

    Unsafe set (progression):
        X_u = { (C_S, C_R) :  C_S + C_R >= C_progression }
        Total tumor burden exceeds the progression threshold,
        indicating treatment failure.

The reach-avoid task: starting from diagnosis (X_0), drive the tumor
into remission (X_g) while never allowing progression (X_u).


Default Parameter Values
------------------------
Adapted from Sun et al. (2016), Table S1:

    r_S       = 0.03       growth rate of sensitive cells     [1/day]
    r_R       = 0.005      growth rate of resistant cells     [1/day]
    T_max     = 2.0        carrying capacity                  [10^8 cells]
    u_mut     = 1e-4       sensitive -> resistant mutation rate [1/day]
    d_S_max   = 0.06       maximal drug-induced death rate    [1/day]
    K_drug    = 0.5        Michaelis constant for drug effect  [normalized]
    sigma_1   = 0.02       diffusion rate, sensitive cells
    sigma_2   = 0.02       diffusion rate, resistant cells
    D_max     = 1.0        maximal drug dosage                [normalized]

    C_S0      = 0.2        initial sensitive cell count       [10^8]
    C_R0      = 0.001      initial resistant cell count       [10^8]

    C_remission   = 0.05   goal threshold                    [10^8]
    C_progression = 1.6    unsafe threshold                  [10^8]
    C_max         = 3.0    domain bound                      [10^8]


Extensions
----------
1.  **Finite-time reach-avoid (Corollary 1):**
    Require remission within a clinically relevant horizon T (e.g.,
    180 days), since prolonged treatment increases toxicity and cost.

2.  **Constrained control effort (Corollary 2):**
    Augment the state with cumulative drug exposure
        E(t) = integral_0^t u(s)^2 ds
    and treat E > E_max as unsafe.  This penalizes sustained high
    dosage, modeling cumulative toxicity / side-effect limits.

3.  **Microenvironment feedback (optional):**
    Include drug-induced resistance factor (DIRF) secretion from the
    original model.  DIRFs upregulate r_R and are proportional to
    drug concentration and sensitive cell count, creating a feedback
    loop: higher drug -> more DIRFs -> faster resistant growth.
    This makes the problem harder and more realistic.


Why This Benchmark Is Interesting
---------------------------------
- **Competing objectives:** killing sensitive cells accelerates
  resistant cell dominance.
- **Nonlinear saturation:** Michaelis--Menten drug effect means
  doubling the dose does not double the kill rate.
- **Multiplicative noise:** diffusion scales with population,
  so small populations are relatively noisier.
- **Clinical relevance:** the model is validated against melanoma
  patient survival data (Flaherty et al., NEJM 2012).
- **Fits the paper's framework:** drift is affine in uncertain
  parameters, so Proposition 1 yields closed-form robust generators.
"""