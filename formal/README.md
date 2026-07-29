# Lean 4 formalization

Machine-checked Lean 4 (`mathlib`) formalizations of the **theoretical** results of the paper
*"Priors learned from legacy reconstructions inherit undetectable overconfidence"*
(`paper/v2/manuscript.tex`).

## Scope against the current manuscript — read this first

This development **proves a superset of what the current paper states**, and it does **not** cover
everything the current paper states. Both directions matter to a reader checking the paper's
machine-checking claim, so both are spelled out here.

*Results proved here that the current manuscript no longer contains.* The scope tables below are
keyed to labels from a longer version. `c:fix`, `p:mean`, `r:ensemble`, `r:hilbert`, `r:sampler` and
`eq:pooled` were cut in a page-limit trim and appear in no section of `paper/v2`. Their Lean proofs
are untouched and still check; they simply have no counterpart to compare against right now.

*Results in the current manuscript that are **not** machine-checked here.* Wave 4 (2026-08-06)
closed `c:honest`, `p:audit` (exact parts), `p:certify` and `eq:covmu`; **wave 5 (2026-08-06) closed
`r:exchange` and the named `p:mapnearnull`/`p:recover` parts**. What remains proved in the paper
alone is deliberate and asymptotic — the full statement of it is the "Not machine-checked" section
further down, which is the authority; in summary:

| Manuscript result | Status |
|---|---|
| `p:audit`'s closing asymptotic `k ≈ 1+(z_{1−α}+z_{1−β})²/(2b log² r)`, and the rank-one search-penalty aside | paper only, **deliberate** — asymptotic approximations; the exact power and level content behind them is checked in `AuditPrice.lean` / `ExchangeRate.lean` |
| Dictionary items | deliberate — variational spectral bands read as `λ_min`/`λ_max`, and the identification of the Lean scenario with the manuscript's matrix notation |
| `c:invariant` — blind conditional as a loop invariant | assembled from `p:recover`(i),(ii), both checked |

Wave 5 modules: `ExchangeRate.lean` (the two currencies at one blind direction — the survey price
from the checked `SpreadKL`/`MeanKL` crossovers, the reference price from `AuditPrice`, and the
divergence of the first as `‖Av‖² → 0`), `MapNearNullCoverage.lean` ((ii)'s coverage tail, with the
`Av = 0` zero-coverage endpoint via mathlib's `condKernel`), `RidgeMarginal.lean` ((iii) plus the
`E‖d‖²` comparison and the 2-D counterexample showing the bound needs its strong-convexity
hypothesis), `RecoverWhitening.lean` (the whitening/pseudoinverse congruence reducing the matrix
recursion to the diagonal one `RecoverConvergence` solves — previously listed here as not checked).

Everything else the manuscript labels formal — the freeze, non-identifiability, the coverage law,
the single-best collapse, the recovery loop, the impossibility, the audit, the certificate, and
the coverage-against-damping bound — corresponds to a theorem below.

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
| Cor `c:fix` — fixed points (both directions) | `Fixpoint.lean` | a fixed point of the step ⟺ matching data marginals (χ²-rigidity) |
| Prop `p:mean` — posterior-mean collapse (structural) | `PosteriorMean.lean` | the posterior mean is a function of its resolved component; zero blind coverage (given the exp-family regularity) |
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
| `p:mean` — log-partition smoothness | `LogPartitionSmooth.lean` | `HasDerivAt Λ` by differentiation under the integral; `Λ' = (∫T·e^{τT})/M`. Injectivity in `PosteriorMeanRegularity.lean` |
| Prop `p:recover` (ii) — resolved convergence | `RecoverConvergence.lean` | fixed point, positivity, monotonicity, no-overshoot (`Ψ(d)−d = d²(d⋆−d)/(d+1)²`); MEAN error geometric via `|1−k| ≤ (1+λ)⁻¹`; COVARIANCE error geometric via `Ψ(d)−d⋆ = (d−d⋆)q(d)`, `q(d)=(2d+1)/(d+1)²<1`, with `[min d₀ d⋆, max d₀ d⋆]` invariant so the floor is permanent. the matrix→diagonal reduction (whitening) is closed in `RecoverWhitening.lean` |
| Rem `r:ensemble` — pooled archive | `BlindFreeze.lean` | a mixture of archives sharing a common blind conditional keeps it |
| Exp-family input to `p:mean` — `∇Λ` injective | `PosteriorMeanRegularity.lean` | on a real Hilbert space the gradient of a strictly convex function is strictly monotone, hence injective (segment restriction + 1D `strictMonoOn_deriv`); discharges `p:mean`'s injectivity hypothesis from strict convexity of the log-partition |
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
| **`p:mean` — multivariate log-partition gradient + composition** | `LogPartitionGradient.lean` | `HasGradientAt Λ (conditional mean)` on an inner-product space under an explicit locally-uniform integrable envelope (mathlib dominated differentiation); `M > 0` proved, not hypothesized; and the composition theorem discharging `posterior_mean_collapse_of_strictConvex`'s `hΛ` hypothesis outright |
| **`p:nearnull` (i) HYPOTHESIS-FREE at the model** | `NearNullDefinitional.lean` | `Σ_post, G, S_y, Q` **defined** from `(A,Γ,Σ_ρ,Σ⋆)`; every preamble relation **proved** (two-Woodbury included); variational Loewner inverse-antitonicity by quadratic-form Cauchy–Schwarz; the bound with the paper's constant from PD/PSD + a variational spectral band |
| **The data law + certified χ² crossover** | `GaussianDataLaw.lean` | `y = Ax + ε ⟹ N(Aμ, L_yL_yᵀ)` by charFun uniqueness; `‖L_y⁻¹u‖² = ⟪u, S_y⁻¹u⟫`; `χ²(β) ≤ β²/2`; certified crossover `n·t² < 2/5 ⟹ summed error > 1/2` |
| **The data-covariance square root, BUILT** | `DataCovarianceSqrt.lean` | invertible self-adjoint `L_y` via `toEuclideanCLM + CFC.sqrt`; capstones `modelLaw_repr_closed` + mean/spread misreport undetectability + crossovers with **no** sqrt hypothesis; `Av = 0` blind endpoints hypothesis-free |
| **`p:mean` CLOSED** | `LogPartitionClosure.lean` | envelope from `E‖x_R‖ < ∞` + variational-PD tilt; strict convexity of `Λ` from the affine-span condition (L² Cauchy–Schwarz equality, no Hessians); composed collapse on model hypotheses only; + `E[Φ(a+bZ)] = Φ(a/√(1+b²))` |
| **`r:selfcheck` + sharpness witnesses** | `SelfCheckAndSharpness.lean` | SBC rank uniformity from exchangeability + a.s. distinctness; prior-independent rank law; power = size; the kinked-penalty witness (unique minimizer, no collapse, **no** `C·ε^κ` bound); `r:sampler`'s mixture-freeze anchor |
| **`p:mapnearnull` (i) at the paper's EXACT hypothesis + (ii)** | `MapNearNullSharp.lean` | the kinked/subgradient tube by secant slopes (sharp constant `μ`); the flat-direction witness exact (law `N(0, s/ε²)`); `E[Var(X\|m)] ≤ c²E[D²]` via condExpL2-as-projection (`E‖d‖² < ∞` a stated hypothesis) |
| **`r:hilbert` ASSEMBLED + `r:ensemble` K-mixture** | `HilbertAssembly.lean` | full separable-Hilbert carrier; the disintegration from mathlib's `condKernel` (a THEOREM, not an input); curated fiber kernel = regularizer's a.e.; the pooled readout-tilted K-atom law of `eq:pooled`, each atom zero-width |
| **Pinsker BUILT + the paper's exact crossover** | `PinskerAndNonlinear.lean` | Pinsker from scratch (`\|P A − Q A\| ≤ √(KL/2)`); KL tensorization BUILT; 1-D Gaussian KL closed form; `eq:nearnull-cross`'s `n < 2/t²` **verbatim** (mean case hypothesis-free on Gaussian n-survey products; spread modulo the flagged per-survey KL bound `hKL`, its χ² counterpart proved); + `r:nonlinear`'s blind-derivative statement |
| **`eq:nearnull-kl` = and `eq:nearnull-cross` verbatim, BOTH misreports** | `SpreadKL.lean` | the spread misreport's exact per-survey Gaussian KL `= (t - log(1+t))/2` at `beta = w(Av)^T S_y^-1(Av)` (1-D variance KL closed form + the rank-one rotation + tensorization); the n-survey KL; the paper's `n < 2/t^2` crossover on the **physical record laws** with no hypotheses beyond the scenario -- discharges `PinskerAndNonlinear`'s formerly-flagged `hKL` |
| **the MEAN misreport's KL at the record law** | `MeanKL.lean` | per-survey `KL = beta/2` at `beta = w^2 (Av)^T S_y^-1 (Av)`, the n-survey tensorization, and `n*beta < 1 => summed error > 1/2` (n* = 1/beta; the mean case's Pinsker constant differs from the spread case's 2/t^2, and the module records that the n*beta<2 form is false) -- the companion of `SpreadKL`, so `eq:nearnull-cross` holds verbatim for BOTH misreports on the physical record laws |

| **Cor `c:honest` — no finite blind interval is an inference** | `HonestInterval.lean` | the density-bound coverage lemma `Pr(θ ∈ [lo W, hi W]) ≤ B·E[len]` via mathlib's `condKernel` disintegration; the Gaussian density maximum (BUILT — mathlib lacked it); constant information law + unbounded witness spread ⇒ `inf_p Pr_p(θ⋆∈Î) = 0`; uniform coverage `≥ 1−α` forces `E\|Î\| = ⊤`. Named modeling hypotheses (constant 𝒲-law, Gaussian fiber form, unbounded spread) each anchored to the checked `NonIdentifiability` witness |
| **Prop `p:audit` — the audit's exact laws, power, and level** | `AuditPrice.lean` | χ²(n) DEFINED as the pushforward law of `‖Z‖²`; sum and r²-scaling laws; the standardization pipelines (`W·√Σ = 1`, `CFC.sqrt (r²•Σ) = r•√Σ` BUILT); null `T ~ χ²(kb)`, alternative `T ~ r²χ²(kb)`; power `= χ²(kb)(Ioi (q/r²))` at any threshold; rejection monotone in `r` ⇒ level `α` over the composite null. The closing asymptotic `k`-approximation deliberately unformalized |
| **Prop `p:certify` — split-conformal validity of the certified band** | `CertifiedBand.lean` | the exact rank law `(j+1)/(k+1)` from exchangeability (reusing `SelfCheckAndSharpness`'s engine); the conformal event = rank event via order statistics; the band is `Icc (m−cs) (m+cs)`; the finiteness-threshold **iff** `⌈(k+1)(1−α)⌉ ≤ k ↔ k ≥ ⌈1/α⌉−1`; below threshold the infinite band covers trivially; the verbatim `certify` endpoint; conditional validity as a hypothesis-based corollary |
| **`eq:covmu` — coverage against damping** | `CoverageDamping.lean` | advertised fiber spread `≥ μ_v^{−1/2}` by an explicit defect identity (equality **iff** `v` is an eigenvector — exact at 1-D blind subspace, strict "understates" otherwise); the coverage bound `2Φ(z/(s⋆√μ_v))−1 ≤ C`; `μ_v(P+λI) = μ_v(P)+λ`; both monotonicity readings — the RHS strictly decreasing in a ridge, and the attained coverage nonincreasing via `NearNullDefinitional`'s variational Loewner antitonicity |

| **`p:mapnearnull`(ii) coverage tail** | `MapNearNullCoverage.lean` | conditional Markov per fiber; the exhibited interval `ψ ± ‖Av‖E[‖d‖\|w]/(αμ_v)` carries conditional mass ≥ 1−α; a minimal-length credible rule is no wider; density bound (REUSING `HonestInterval.coverage_compProd_le`) + tower give coverage ≤ `2B⋆‖Av‖E‖d‖/(αμ_v)`; the `Av = 0` endpoint is exactly zero; a `condKernel` variant makes the conditional's existence a theorem |
| **`r:exchange` — the two currencies at one witness** | `ExchangeRate.lean` | the `r = √(1+w/s⋆²)` dictionary (strictly monotone bijection); the survey-price divergence as a `Tendsto` (budget → ∞ as `‖Av‖² → 0`, from the checked `SpreadKL`/`MeanKL` crossovers), infinite at `Av = 0` (record laws EQUAL via the KL-zero Gibbs converse); operator-independence of the audit price as a theorem (`rfl` across arbitrary operators); the capstone packaging both prices at the same witness |
| **`p:mapnearnull`(iii) + the `E‖d‖²` input** | `RidgeMarginal.lean` | the ridge estimator IS the minimizer (proved, not assumed); push-through; `Var(vᵀx̂) ≤ (bound)·‖Ãv‖²/λ²` variationally with the `Av = 0` zero endpoint; the fully explicit 2-D coupled-precision counterexample with order-one blind marginal at `Av = 0`; the objective-comparison chain `misfit_comparison → expected_grad_sq_lt_top` discharging the `E‖d‖² < ∞` hypothesis |
| **`p:recover`(ii) whitening/congruence reduction** | `RecoverWhitening.lean` | the Woodbury rearrangement `AΣ_postAᵀ = D(D+Γ)⁻¹Γ`; the data-space recursion is autonomous (+ the mean step); whitening intertwines it with the recursion `RecoverConvergence` solves; interior fixed point unique; the pseudoinverse lift `P_RΣ_tP_R = A⁺D_t(A⁺)ᵀ` carrying the scalar convergence to the resolved block (`tendsto_resolved_covariance`/`_mean`) |

**Not machine-checked (stated in the paper; proofs in the appendix).** After waves 1-5 the remainder
is the deliberate set:

* `p:audit`'s closing **asymptotic** `k ≈ 1+(z_{1−α}+z_{1−β})²/(2b log² r)` and the discussion's
  rank-one **search-penalty** aside (`log b`) — asymptotic approximations, out of scope by design;
  the exact power/level content behind them is checked in `AuditPrice.lean`/`ExchangeRate.lean`.
* The **general-mollifier form of the smoothing floor** in `r:map-lg`'s appendix passage (the
  Minkowski/Lipschitz display `E[Var(vᵀx_B|r)] ≤ h²(c_B + Lm_R)²`) — the Gaussian case IS checked
  (`Coverage.tendsto_C_zero_atZero`), the general mollifier stays on paper.
* The **no-better-score aside** of `p:certify` ("no score adapted to the recorded data improves on
  this one") — an observation the certificate's validity never uses; no Lean counterpart.
* Two **classical proof-internal steps**: the several-surveys Gaussian **sufficiency reduction** in
  the proof of `c:augment` (the `t = Γ_r Lᵀ Γ̃⁻¹ ỹ` textbook computation), and the **EM
  descent-and-compactness assembly** in `p:recover`(ii) upgrading the checked floor + unique
  interior fixed point to convergence of the matrix iterates (named in `RecoverWhitening.lean`'s
  header); `p:mapnearnull`(i)'s solved **quadratic identity** (`vᵀx̂ − ψ = −⟪Av,d⟫/(vᵀPv)`) is a
  one-line consequence of the checked stationarity input, not separately stated in Lean.
* **Dictionary items**, deliberate: spectral bands stated variationally (their reading as
  `lambda_min`/`lambda_max` is the eigenvalue dictionary); the identification of the Lean scenario
  (`EuclideanSpace`, quadratic forms, pushforward Gaussians) with the manuscript's matrix notation;
  and model-definition inputs (`hfac`/`hres`, the affine-span reading) that ARE the model, not gaps.

Instance-level hypothesis accounting (e.g. `CertifiedBand`'s tie-free scores `hdist`,
`CoverageDamping`'s conditional-variance identification, `MapNearNullSharp`'s `hX`/`htube`) lives
in each module's header docstring; the rows above summarize. The 2026-08-07 text↔Lean assumptions
pass verified that the manuscript's stated hypotheses now carry every binder the Lean consumes.

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
