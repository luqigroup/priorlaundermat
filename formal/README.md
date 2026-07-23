# Lean 4 formalization

Machine-checked Lean 4 (`mathlib`) formalizations of the **theoretical** results of the paper
*"Prior laundering: learned priors with inherited, undetectable overconfidence."*

Every result here is **kernel-verified with no `sorry`**: `#print axioms` on each theorem shows only
Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`) and never `sorryAx`, and a
source grep for `sorry` / `admit` / `native_decide` is clean. The proofs are complete and checked, not
aspirational.

**Nothing domain-specific is axiomatized.** mathlib's disintegration and Gaussian-measure results are
used as *proved* lemmas, not assumed; where an analytic step is not yet formalized it appears as an
*explicit hypothesis* of the theorem, never as an axiom. The exact `mathlib` version is pinned in
`lake-manifest.json`, so the formalization remains buildable.

## What is formalized (precise scope)

Fully machine-checked (statement **and** proof, no `sorry`):

| Paper result | Lean file | What is proved |
|---|---|---|
| Thm `p:law` — EM-step identity + marginal-likelihood ascent | `PriorLaundering.lean` | the archive is the regularizer advanced one NPMLE/EM step; the step does not decrease the data-marginal likelihood |
| Thm `p:map` — single-best collapse | `SingleBestCollapse.lean` | deterministic collapse; Gaussian rank/kernel sharpening; zero fiber-conditional **and** marginal blind coverage |
| Prop `p:blind` — blind-fiber freeze | `BlindFreeze.lean` | the curated conditional equals the regularizer's; the Gaussian blind-block precision identity |
| Cor `c:nonident` — undetectability | `NonIdentifiability.lean` | two truths differing only on the blind subspace train the identical prior; the linear–Gaussian witness |
| the fixed-point characterization (folded into p:law's discussion) | `Fixpoint.lean` | a fixed point of the step ⟺ matching data marginals (χ²-rigidity) |
| `p:map`'s posterior-mean extension (discussion) — posterior-mean collapse (structural) | `PosteriorMean.lean` | the posterior mean is a function of its resolved component; zero blind coverage (given the exp-family regularity) |
| Prop `p:cover` — coverage shortfall | `Coverage.lean`, `GaussianCDF.lean` | closed-form coverage; overconfidence when the belief is tighter than the truth; a frozen mean error only worsens it (∂C/∂δ<0) |
| Cor `c:augment` — de-freezing | `DeFreeze.lean` | an added channel de-freezes exactly the directions it resolves |
| Prop `p:cover-na` — alignment-free coverage | `Coverage.lean` | `C` even in the mean gap; the cap `C(δ,r) ≤ C(0,r)` for **any** `δ`, equality iff `δ = 0`; below nominal at every readout when `r < z` |
| Prop `p:recover` (i) — the correction loop's freeze | `BlindFreeze.lean` | iterating fiber-constant corrections leaves the blind fiber conditional fixed |
| `p:nearnull` (i) — precision freeze bound | `NearNullFreeze.lean` | `|⟪Av, B(Av)⟫| ≤ ‖B‖‖Av‖²` (quadratic form vs operator norm); zero on the kernel, so the exact freeze is the endpoint |
| `p:mapnearnull` (i) — pinning tube | `NearNullFreeze.lean` | `μ`-strong convexity pins the archived coordinate within `|g'|/μ` of the penalty's conditional minimizer, hence within `‖Av‖‖d‖/μ` |
| `p:recover` (ii) — matrix floor reduction | `MatrixFloor.lean` | lower bounds add (Weyl) and self-adjoint congruence inherits the middle bound, both variational; with the scalar facts these give λ_{t+1} ≥ g(λ_t) with no diagonalization. including the spectral mapping λ_min(K̃) ≥ f(λ), also variational (Cauchy–Schwarz on w = (D+I)⁻¹v) rather than functional calculus |
| `p:nearnull` (i) — factored form | `NearNullFreeze.lean` | a precision difference of the form A*BA is moved by ≤ ‖B‖‖Av‖² along v, and by exactly 0 on ker A. the Woodbury rearrangement into that form is in `NearNullFactored.lean` |
| `p:mapnearnull` (i) — variational identification | `NearNullFreeze.lean` | stationarity forces the penalty's directional derivative to equal −⟪Av,d⟫, hence \|rv\| ≤ ‖Av‖‖d‖ |
| `eq:nearnull-freeze` — factored precision difference | `NearNullFactored.lean` | Woodbury (mathlib `add_mul_mul_inv_eq_sub`) plus the preamble's definitional relations give Σ_q⁻¹ − Σ_ρ⁻¹ = Aᵀ(Γ⁻¹ − Γ⁻¹Q⁻¹Γ⁻¹)A, naming B; and a PSD difference is bounded by the larger bound, giving c |
| `p:map`'s posterior-mean extension (discussion) — log-partition smoothness | `LogPartitionSmooth.lean` | `HasDerivAt Λ` by differentiation under the integral; `Λ' = (∫T·e^{τT})/M`. Injectivity in `PosteriorMeanRegularity.lean` |
| Prop `p:recover` (ii) — resolved convergence | `RecoverConvergence.lean` | fixed point, positivity, monotonicity, no-overshoot (`Ψ(d)−d = d²(d⋆−d)/(d+1)²`); MEAN error geometric via `|1−k| ≤ (1+λ)⁻¹`; COVARIANCE error geometric via `Ψ(d)−d⋆ = (d−d⋆)q(d)`, `q(d)=(2d+1)/(d+1)²<1`, with `[min d₀ d⋆, max d₀ d⋆]` invariant so the floor is permanent. Matrix→diagonal reduction (whitening) remains in the paper's supporting-results subsection |
| `c:augment`'s pooled-archive freeze — pooled archive | `BlindFreeze.lean` | a mixture of archives sharing a common blind conditional keeps it |
| Exp-family input to `p:map`'s posterior-mean extension (discussion) — `∇Λ` injective | `PosteriorMeanRegularity.lean` | on a real Hilbert space the gradient of a strictly convex function is strictly monotone, hence injective (segment restriction + 1D `strictMonoOn_deriv`); discharges the posterior-mean extension's injectivity hypothesis from strict convexity of the log-partition |
| Le Cam engine for `p:nearnull` — undetectability testing bound | `NearNullLeCam.lean` | `ε`-close data laws ⇒ every test has total error `≥ 1 − ε` (minimax form: `max` of the two errors `≥ (1−ε)/2`); the `ε = 0` endpoint recovers `c:nonident`'s exact indistinguishability |
| χ² controls TV (`p:nearnull`) | `ChiSquaredTV.lean` | Jensen/Cauchy–Schwarz: `\|P A − Q A\| ≤ √(χ²(P‖Q))`, hence the Le Cam bound with `ε = √(χ²)` — **no Pinsker** |
| Gaussian pdf-ratio integral (`p:nearnull`) | `GaussianPdfIntegral.lean` | `∫ p₁²/p₂ dvol = exp((m₁−m₂)²/v)` (the χ² numerator of two same-variance Gaussians) |
| Gaussian χ² + near-null undetectability (`p:nearnull`) | `GaussianChiSquared.lean` | `χ²(N(m₁,v)‖N(m₂,v)) = exp((m₁−m₂)²/v) − 1`, fed into the Le Cam bound: `1 − √(exp((Δm)²/v)−1) ≤ P₁.real A + P₂.real Aᶜ` for every test — the 1-D mean-misreport near-null undetectability, `→ 1` as `Δm → 0` |
| 1-D spread-misreport χ² (`p:nearnull`) | `GaussianChiSquaredVar.lean` | `χ²(N(m,v₁)‖N(m,v₂)) = v₂/√(v₁(2v₂−v₁)) − 1` (same mean, different variance) + its near-null undetectability — the paper's *primary* case at 1-D |
| χ² invariance under a measurable equivalence (`p:nearnull`) | `ChiSquaredInvariance.lean` | `χ²(e#P‖e#Q) = χ²(P‖Q)` for `e : α ≃ᵐ β` — the whitening/rotation engine |
| χ² tensorization, binary + d-fold (`p:nearnull`) | `ChiSquaredTensor.lean`, `ChiSquaredPi.lean` | `χ²(P⊗P'‖Q⊗Q') = (1+χ²)(1+χ²')−1` and `χ²(⨂ᵢPᵢ‖⨂ᵢQᵢ) = ∏(1+χ²ᵢ)−1` (drops common coordinates; gives the `n`-survey product); also proves the `pi ≪ pi` primitive mathlib lacks |
| Isotropic multivariate Gaussian χ² (`p:nearnull`) | `MultivariateGaussianChiSquared.lean` | `χ²(N₀(·+c)‖N₀) = exp(‖c‖²) − 1` on `ℝⁿ` (identity-covariance mean shift) |
| **General-covariance multivariate mean-misreport near-null undetectability** (`p:nearnull`) | `MultivariateGaussianWhitening.lean` | `χ²(N(μ₁,LLᵀ)‖N(μ₂,LLᵀ)) = exp((μ₁−μ₂)ᵀ(LLᵀ)⁻¹(μ₁−μ₂)) − 1`, and `1 − √(…) ≤ P₁.real A + P₂.real Aᶜ` for every test |
| Axis-aligned multivariate spread χ² (`p:nearnull`) | `MultivariateGaussianScale.lean` | `χ²(N₀‖N₀ scaled) = ∏(cᵢ²/√(2cᵢ²−1))−1` via SCALING + tensorization; single-coordinate case `s²/√(2s²−1)−1` |
| **General multivariate spread-misreport near-null undetectability** (`p:nearnull`) | `MultivariateGaussianSpread.lean` | `χ²(N(0,I)‖N(0,I+uuᵀ)) = (1+‖u‖²)/√(1+2‖u‖²)−1` (rotate the perturbation onto a coordinate via `ext_of_charFunDual` uniqueness → the axis-aligned scaling), and `1 − √(…) ≤ P.real A + Q.real Aᶜ` for every test |
| **`p:nearnull` (i) COMPOSED** — the bound for the precision difference itself | `NearNullComposed.lean` | ONE theorem: `\|vᵀ(Σ_q⁻¹−Σ_ρ⁻¹)v\| ≤ c‖Av‖²` over matrices (Woodbury-factored middle `B` + matrix sandwich in one statement), the `Av = 0` endpoint exactly zero, and the PSD-difference form giving the paper's `c = max{a,b}`; hypotheses = the preamble's definitional relations (as `NearNullFactored` takes) + variational bounds on the middle factors |
| **`p:mapnearnull` (i) COMPOSED** — the tube, one theorem | `MapNearNullTube.lean` | `\|vᵀx̂ − φ(Πx̂)\| ≤ \|⟪Av,d⟫\|/μ ≤ ‖Av‖‖d‖/μ` chained from pinning + Cauchy–Schwarz + stationarity (differentiable 1-D penalty sections; kinked case stays on paper), + the `Av = 0` exact-collapse endpoint |
| **`p:nearnull` (ii) AT THE OPERATOR + n surveys** | `NearNullOperator.lean` | the spread whitening to ANY common nonsingular covariance (rank-one square root constructed); mean and spread misreport χ² + Le Cam undetectability at the data laws `N(Aμ, L_yL_yᵀ)`; iid tensorization `χ²_n = (1+χ²)ⁿ−1`; the machine crossover `(1+χ²)ⁿ < 5/4 ⟹ every n-survey test's summed error > 1/2`, both misreports; `Av = 0` endpoints recovering `c:nonident` |
| **`p:map`'s posterior-mean extension (discussion) — multivariate log-partition gradient + composition** | `LogPartitionGradient.lean` | `HasGradientAt Λ (conditional mean)` on an inner-product space under an explicit locally-uniform integrable envelope (mathlib dominated differentiation); `M > 0` proved, not hypothesized; and the composition theorem discharging `posterior_mean_collapse_of_strictConvex`'s `hΛ` hypothesis outright |
| **`p:nearnull` (i) HYPOTHESIS-FREE at the model** | `NearNullDefinitional.lean` | `Σ_post, G, S_y, Q` **defined** from `(A,Γ,Σ_ρ,Σ⋆)`; every preamble relation **proved** (two-Woodbury included); variational Loewner inverse-antitonicity by quadratic-form Cauchy–Schwarz; the bound with the paper's constant from PD/PSD + a variational spectral band |
| **The data law + certified χ² crossover** | `GaussianDataLaw.lean` | `y = Ax + ε ⟹ N(Aμ, L_yL_yᵀ)` by charFun uniqueness; `‖L_y⁻¹u‖² = ⟪u, S_y⁻¹u⟫`; `χ²(β) ≤ β²/2`; certified crossover `n·t² < 2/5 ⟹ summed error > 1/2` |
| **The data-covariance square root, BUILT** | `DataCovarianceSqrt.lean` | invertible self-adjoint `L_y` via `toEuclideanCLM + CFC.sqrt`; capstones `modelLaw_repr_closed` + mean/spread misreport undetectability + crossovers with **no** sqrt hypothesis; `Av = 0` blind endpoints hypothesis-free |
| **`p:map`'s posterior-mean extension (discussion) CLOSED** | `LogPartitionClosure.lean` | envelope from `E‖x_R‖ < ∞` + variational-PD tilt; strict convexity of `Λ` from the affine-span condition (L² Cauchy–Schwarz equality, no Hessians); composed collapse on model hypotheses only; + `E[Φ(a+bZ)] = Φ(a/√(1+b²))` |
| **`r:selfcheck` + sharpness witnesses** | `SelfCheckAndSharpness.lean` | SBC rank uniformity from exchangeability + a.s. distinctness; prior-independent rank law; power = size; the kinked-penalty witness (unique minimizer, no collapse, **no** `C·ε^κ` bound); sampler robustness / mixture anchor |
| **`p:mapnearnull` (i) at the paper's EXACT hypothesis + (ii)** | `MapNearNullSharp.lean` | the kinked/subgradient tube by secant slopes (sharp constant `μ`); the flat-direction witness exact (law `N(0, s/ε²)`); `E[Var(X\|m)] ≤ c²E[D²]` via condExpL2-as-projection (`E‖d‖² < ∞` a stated hypothesis) |
| **the function-space carrier (Sec 3) ASSEMBLED + `c:augment`'s pooled-archive freeze K-mixture** | `HilbertAssembly.lean` | full separable-Hilbert carrier; the disintegration from mathlib's `condKernel` (a THEOREM, not an input); curated fiber kernel = regularizer's a.e.; the pooled readout-tilted K-atom law of `eq:pooled`, each atom zero-width |
| **Pinsker BUILT + the paper's exact crossover** | `PinskerAndNonlinear.lean` | Pinsker from scratch (`\|P A − Q A\| ≤ √(KL/2)`); KL tensorization BUILT; 1-D Gaussian KL closed form; `eq:nearnull-cross`'s `n < 2/t²` **verbatim** (mean case hypothesis-free on Gaussian n-survey products; spread modulo the flagged per-survey KL bound `hKL`, its χ² counterpart proved); + `r:nonlinear`'s blind-derivative statement |
| **`eq:nearnull-kl` = and `eq:nearnull-cross` verbatim, BOTH misreports** | `SpreadKL.lean` | the spread misreport's exact per-survey Gaussian KL `= (t - log(1+t))/2` at `beta = w(Av)^T S_y^-1(Av)` (1-D variance KL closed form + the rank-one rotation + tensorization); the n-survey KL; the paper's `n < 2/t^2` crossover on the **physical record laws** with no hypotheses beyond the scenario -- discharges `PinskerAndNonlinear`'s formerly-flagged `hKL` |
| **the MEAN misreport's KL at the record law** | `MeanKL.lean` | per-survey `KL = beta/2` at `beta = w^2 (Av)^T S_y^-1 (Av)`, the n-survey tensorization, and `n*beta < 1 => summed error > 1/2` (n* = 1/beta; the mean case's Pinsker constant differs from the spread case's 2/t^2, and the module records that the n*beta<2 form is false) -- the companion of `SpreadKL`, so `eq:nearnull-cross` holds verbatim for BOTH misreports on the physical record laws |

**Not machine-checked (stated in the paper; proofs in the paper's supporting-results subsection).** After waves 1-3 the remainder
is short:

* `p:mapnearnull`(ii)'s **coverage tail** (`<= 2B*||Av|| E||d||/(alpha mu)`) and (iii)'s ridge-marginal
  computation; the **objective-comparison derivation of `E||d||^2 < inf`** (a stated hypothesis where
  (ii)'s variance bound is proved).
* `p:recover`(ii)'s **matrix-to-diagonal whitening reduction** (the diagonalized recursion and the
  matrix floor are both checked; the congruence between them is the supporting-results subsection's).
* **Dictionary items**, deliberate: spectral bands stated variationally (their reading as
  `lambda_min`/`lambda_max` is the eigenvalue dictionary); the identification of the Lean scenario
  (`EuclideanSpace`, quadratic forms, pushforward Gaussians) with the manuscript's matrix notation;
  and model-definition inputs (`hfac`/`hres`, the affine-span reading) that ARE the model, not gaps.

**The experimental claims of Section 5 are not formalized** — they are empirical.

Where the formalization surfaced a hypothesis the paper left implicit, the paper was updated to match
(e.g. the full-support / finite-divergence regularity in `p:law`, and the a.e. nonatomicity in `p:map`):
the paper's statements and proofs track the Lean, not the reverse.

## Build

Requires the Lean toolchain via [`elan`](https://github.com/leanprover/elan); the exact versions are
pinned in `lean-toolchain` and `lake-manifest.json`, so the formalization remains buildable.

```sh
lake exe cache get   # prebuilt mathlib oleans — do not compile mathlib from source
lake build           # the Lean kernel checks every proof
```

## Continuous integration

`ci/lean-ci-workflow.yml` is a GitHub Actions workflow that installs the toolchain, fetches the mathlib
cache, runs `lake build`, and prints `#print axioms` for the main theorems — so anyone can confirm at a
glance that the proofs are complete and `sorry`-free. **To activate it, move it to
`.github/workflows/lean.yml`** (it is parked here only because the push credential used during
development lacked GitHub's `workflow` scope).
