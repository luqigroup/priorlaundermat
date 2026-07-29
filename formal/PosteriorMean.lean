import SingleBestCollapse

/-!
# Proposition `p:mean` of the paper: posterior-mean archives collapse the same way

This file machine-checks the structural core of **Proposition `p:mean`** ("Posterior-mean archives
collapse the same way", linear operator, Gaussian noise, any prior with a finite first moment).

Thesis: under Gaussian noise `y∣x ~ 𝒩(Ax, Γ)`, the posterior-mean (MMSE) archive
`xhat(y) = 𝔼_reg[x∣y]` depends on `y` only through the resolved statistic `AᵀΓ⁻¹y`; the map from that
statistic to the resolved component `xhat_R` is the gradient of a strictly convex smooth function
(hence injective), so the **whole** reconstruction — blind component included — is a deterministic
function of its own resolved component.  The blind-fiber conditional is then a Dirac and the coverage
is zero, exactly as in `p:map` — with no penalty structure and no selection to fix (the mean is
unique).

## Results

* `posterior_mean_collapse` — the structural collapse: if `xhat = ψ ∘ T` factors through a resolved
  statistic `T` and its resolved part `P_R ∘ xhat = ρ ∘ T` is an **injective** image of `T`, then two
  measurements with the same resolved reconstruction give the same reconstruction (`xhat` is a
  deterministic function of its resolved component; zero within-fiber width).
* `posterior_mean_is_function_of_resolved` — packaged as `∃ φ, xhat = φ ∘ (P_R ∘ xhat)`.
* `posterior_mean_coverage_zero` — the coverage of the posterior-mean archive is zero: this is
  `SingleBestCollapse.coverage_zero` **verbatim**, since the report (any blind functional of `xhat(y)`)
  is a measurable function of the measurement — "the zero-coverage computation of `p:map` asks only
  that the report be a measurable function of the measurement, which `xhat(y)` is."

The injectivity `ρ` (the strict convexity of the log-partition `Λ`, i.e. the strictly monotone
gradient `∇Λ`) is the analytic content the paper derives; here it is an explicit hypothesis (see the
reconciliation).  The factoring `xhat = ψ ∘ T` through the resolved statistic `AᵀΓ⁻¹y` is the Gaussian
likelihood's exponential-family structure, likewise taken as the model input.

## Honesty

No `sorry`/`admit`.  `#print axioms` on each main theorem lists only `propext`, `Classical.choice`,
`Quot.sound` (no `sorryAx`).  Reconciliation audit at the end.
-/

open MeasureTheory ProbabilityTheory

namespace PosteriorMean

/-! ## The structural collapse -/

/-- **Posterior-mean / MMSE collapse (structural core).**

If the reconstruction `xhat` factors through a resolved statistic `T` (`xhat = ψ ∘ T`; for a Gaussian
likelihood `T y = AᵀΓ⁻¹y`), and its resolved component is an **injective** image of `T`
(`P_R ∘ xhat = ρ ∘ T` with `ρ` injective — the strictly monotone gradient `∇Λ` of the strictly convex
log-partition), then the whole reconstruction is a deterministic function of its own resolved
component: two measurements with the same resolved reconstruction give the **same** reconstruction
(zero within-fiber width, `C^sb = 0`). -/
theorem posterior_mean_collapse
    {𝓨 X S Rc : Type*} (xhat : 𝓨 → X) (T : 𝓨 → S) (ψ : S → X) (PR : X → Rc) (ρ : S → Rc)
    (hfac : ∀ y, xhat y = ψ (T y)) (hres : ∀ y, PR (xhat y) = ρ (T y)) (hρ : Function.Injective ρ)
    {y y' : 𝓨} (h : PR (xhat y) = PR (xhat y')) :
    xhat y = xhat y' := by
  have hT : T y = T y' := hρ (by rw [← hres y, ← hres y']; exact h)
  rw [hfac y, hfac y', hT]

/-- **The reconstruction is a function of its own resolved component**: `∃ φ, xhat = φ ∘ (P_R ∘ xhat)`.
The blind component is therefore a deterministic function of the resolved component — the blind-fiber
conditional is a Dirac. -/
theorem posterior_mean_is_function_of_resolved
    {𝓨 X S Rc : Type*} [Nonempty 𝓨] (xhat : 𝓨 → X) (T : 𝓨 → S) (ψ : S → X) (PR : X → Rc) (ρ : S → Rc)
    (hfac : ∀ y, xhat y = ψ (T y)) (hres : ∀ y, PR (xhat y) = ρ (T y)) (hρ : Function.Injective ρ) :
    ∃ φ : Rc → X, ∀ y, xhat y = φ (PR (xhat y)) := by
  classical
  -- `φ` sends a resolved value to the (well-defined) reconstruction over any measurement realizing it.
  refine ⟨fun r => if h : ∃ z, PR (xhat z) = r then xhat h.choose else xhat (Classical.arbitrary 𝓨),
    fun y => ?_⟩
  have hex : ∃ z, PR (xhat z) = PR (xhat y) := ⟨y, rfl⟩
  have hφ : (fun r => if h : ∃ z, PR (xhat z) = r then xhat h.choose
      else xhat (Classical.arbitrary 𝓨)) (PR (xhat y)) = xhat hex.choose := dif_pos hex
  rw [hφ]
  exact (posterior_mean_collapse xhat T ψ PR ρ hfac hres hρ hex.choose_spec).symm

/-! ## Coverage — reused verbatim from `p:map` -/

/-- **Zero coverage for the posterior-mean archive.** The collapse of `p:map` applies verbatim:
the report (any blind functional of `xhat(y)`) is a measurable function of the measurement, so — being
independent of a nonatomic truth given the resolved coordinate — it covers the truth with probability
zero.  This is exactly `SingleBestCollapse.coverage_zero`, re-exported for the posterior-mean report;
the mean being unique, there is no penalty structure and no selection to fix. -/
theorem posterior_mean_coverage_zero
    {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
    {θ report : Ω → ℝ} (hθ : Measurable θ) (hrep : Measurable report)
    (hindep : IndepFun θ report P) [NoAtoms (P.map θ)] :
    P {ω | θ ω = report ω} = 0 :=
  SingleBestCollapse.coverage_zero hθ hrep hindep

/-!
## Reconciliation — what `p:mean`'s formalization needs vs. what the paper states

**(a)** already in the text; **(b)** load-bearing, the paper's derived content taken as hypothesis;
**(c)** automatic.

* `xhat = ψ ∘ T` (the reconstruction factors through the resolved statistic `T = AᵀΓ⁻¹y`) — **(b)**:
  the paper derives this from the Gaussian likelihood's exponential-family form
  `h(y) exp(⟨t, x_R⟩ − ½ q(x_R))`, `t = AᵀΓ⁻¹y`; here it is the model input.  (Deriving it needs the
  Gaussian-density factorization — an analytic layer not formalized.)
* `P_R ∘ xhat = ρ ∘ T` with `ρ` **injective** — **(b)**: this is the crux the paper proves —
  `xhat_R = ∇Λ(t)` with `Λ` the strictly convex smooth log-partition, so `∇Λ` is strictly monotone
  hence injective (Hessian `∇²Λ = Cov(x_R∣t) ≻ 0` exactly when the resolved support affinely spans).
  Taken here as an explicit hypothesis; the strict-convexity / dominated-differentiation derivation
  is the analytic content not formalized.  The formalization isolates it cleanly: **injectivity of
  the statistic→resolved map is the entire non-trivial input**, and from it the collapse is
  elementary (`posterior_mean_collapse`).
* coverage — **(a)**, and literally reused: `p:map`'s `coverage_zero` needs only that the report be a
  measurable function of the measurement, which `xhat(y)` is (the paper says exactly this).  No new
  hypothesis; `posterior_mean_coverage_zero` is `SingleBestCollapse.coverage_zero`.

Surfaced: `p:mean`'s two structural inputs (factoring through `AᵀΓ⁻¹y`; injective statistic→resolved
map) are precisely the paper's Gaussian exponential-family + strict-convexity facts.  The elementary
collapse and the zero coverage then follow with no further assumption — the mean's uniqueness removing
the argmin-selection subtlety of `p:map`.

### Proof-level audit (Lean proof vs. the appendix argument)

The appendix proves `p:mean` in three analytic movements — (1) the Gaussian likelihood factors as
`h(y) exp(⟨t,x_R⟩ − ½q(x_R))`, `t = AᵀΓ⁻¹y`, so `x̂` is a function of `t`; (2) `Λ(τ)` is finite,
smooth, with `∇Λ = 𝔼[x_R∣τ] = x̂_R` and `∇²Λ = Cov(x_R∣τ) ≻ 0`; (3) a strictly monotone gradient is
injective, so `x̂_R` determines `t`, and with it `x̂ = φ(x̂_R)`; then coverage is `p:map`'s.  The Lean
proof surfaces:

* **(c) the appendix interleaves analysis and combinatorics; Lean separates them.** The genuine
  content is entirely (1)+(2): "`x̂` factors through `t`" and "`t ↦ x̂_R` is injective".  Given those,
  step (3) — the paper's sentence "a strictly monotone gradient is injective, so the resolved
  component determines the statistic, and with it the entire reconstruction" — is a **two-line,
  axiom-free** argument (`posterior_mean_collapse`: `ρ` injective and `P_R∘x̂ = ρ∘T` force
  `T y = T y'` from equal resolved parts, whence `x̂ y = ψ(T y) = x̂ y'`).  *Proposed appendix edit:*
  state (1),(2) as the two hypotheses and (3) as the elementary consequence, rather than folding the
  determinacy conclusion into the strict-convexity paragraph.
* **(a) coverage is reused verbatim, not re-argued.** The paper writes that the zero-coverage
  computation "asks only that the report be a measurable function of the measurement, which `x̂(y)`
  is."  Lean makes this literal: `posterior_mean_coverage_zero` **is**
  `SingleBestCollapse.coverage_zero` (`p:map`'s coverage lemma), applied with no new proof.  *Proposed
  appendix edit:* cite `p:map`'s coverage statement directly for `p:mean` rather than restating the
  integration; nothing about the mean enters the coverage beyond measurability.
* **(b) what the proof needs that the text states only in words.** Injectivity of `ρ = ∇Λ` and the
  factoring `x̂ = ψ ∘ T` are the two load-bearing inputs; the paper *derives* them (dominated
  differentiation of `Λ`; the exponential-family form), but the derivations are the analytic layer
  **not** formalized here — they are the explicit hypotheses `hρ`, `hfac`.  The formalization thus
  pins down exactly where analysis is unavoidable: only in establishing injectivity of the
  statistic→resolved map.
-/

end PosteriorMean

/-! ### Axiom audit

The doc comment above claims these rest only on Lean's three standard axioms. These commands
are what make that checkable rather than asserted: each must print `propext, Classical.choice,
Quot.sound` and never `sorryAx`. -/

#print axioms PosteriorMean.posterior_mean_collapse
#print axioms PosteriorMean.posterior_mean_coverage_zero
