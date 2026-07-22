/-
# `p:mean`'s multivariate log-partition gradient: differentiation under the integral on the
resolved subspace

`PosteriorMeanRegularity.posterior_mean_collapse_of_strictConvex` consumes
`hΛ : ∀ s, HasGradientAt Λfun (Λ s) s` — the **multivariate** smoothness of the resolved
log-partition — while `LogPartitionSmooth.lean` machine-checks only the one-parameter derivative.
This file supplies the missing multivariate half of `p:mean` ("Posterior-mean archives collapse the
same way"): for

    M(τ) = ∫ exp(⟪τ, X a⟫ − q a) dμ(a),      Λ(τ) = log M(τ),

on a real inner-product space `H` (`EuclideanSpace ℝ (Fin r)` in the paper — the resolved subspace
`row(A)` in an orthonormal basis; `X a` is the resolved coordinate `x_R` of the sample, `q` absorbs
the paper's `q(x_R)/2`), differentiating under the integral gives

    ∇Λ(τ) = (∫ exp(⟪τ, X a⟫ − q a) • X a dμ(a)) / M(τ),

the tilted (posterior) mean `𝔼[x_R ∣ τ]`.  The result is stated as `HasGradientAt` — not bare
`HasFDerivAt` — because that is the exact shape `posterior_mean_collapse_of_strictConvex`'s `hΛ`
consumes; the Fréchet derivative `∫ exp(…) • innerSL ℝ (X a) dμ` is identified with the gradient
through `InnerProductSpace.toDual` (`∫ c • innerSL(X) = toDual (∫ c • X)`), and that identification
is **proved** here (`integral_smul_innerSL_eq_toDual`), not assumed.

## Explicit hypotheses (the appendix's "dominated locally uniformly", stated exactly as mathlib's
`hasFDerivAt_integral_of_dominated_of_fderiv_le` wants them)

* `hX`, `hq` — a.e.-strong measurability of the coordinate `X` and the tilt `q`;
* `hM_int` — integrability of the integrand at the base point `τ₀`;
* `h_bound` / `bound_int` — an integrable envelope: `exp(⟪τ, X a⟫ − q a) * ‖X a‖ ≤ bound a` for
  all `τ` in a neighborhood `s ∈ 𝓝 τ₀`, `bound` integrable.  This is mathlib's `‖F' τ a‖ ≤ bound a`
  verbatim, because `‖exp(…) • innerSL ℝ (X a)‖ = exp(…) * ‖X a‖` (proved in `hasFDerivAt_M`);
* `hμ : μ ≠ 0` — from which `M(τ) > 0` is **proved** (`M_pos`, the integrand is strictly positive),
  not assumed.

## Composition with the collapse

`posterior_mean_collapse_of_logPartition` composes with
`PosteriorMeanRegularity.posterior_mean_collapse_of_strictConvex` (imported, not edited):
`hasGradientAt_logPartition_forall` discharges its `hΛ` hypothesis, so the only analytic input left
as a hypothesis is the strict convexity of `Λ` (the paper's affine-span condition
`∇²Λ = Cov(x_R ∣ τ) ≻ 0`) — exactly the hypothesis `PosteriorMeanRegularity` already takes.

No `sorry`/`admit`; `#print axioms` on each result at the end.
-/
import Mathlib
import PosteriorMeanRegularity

open MeasureTheory Filter InnerProductSpace
open scoped Topology InnerProductSpace

namespace LogPartitionGradient

variable {α : Type*} [MeasurableSpace α] {μ : Measure α}
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [CompleteSpace H]

/-- The multivariate partition function `M(τ) = ∫ exp(⟪τ, X a⟫ − q a) dμ(a)` of the tilted family
on the resolved subspace (`X a` = resolved coordinate `x_R`; `q` absorbs the paper's `q(x_R)/2`). -/
noncomputable def M (μ : Measure α) (X : α → H) (q : α → ℝ) (τ : H) : ℝ :=
  ∫ a, Real.exp (⟪τ, X a⟫_ℝ - q a) ∂μ

/-- The unnormalized tilted mean `∫ exp(⟪τ, X a⟫ − q a) • X a dμ(a)`, the numerator of `∇Λ`. -/
noncomputable def meanNum (μ : Measure α) (X : α → H) (q : α → ℝ) (τ : H) : H :=
  ∫ a, Real.exp (⟪τ, X a⟫_ℝ - q a) • X a ∂μ

/-- The log-partition `Λ(τ) = log M(τ)`. -/
noncomputable def Λ (μ : Measure α) (X : α → H) (q : α → ℝ) (τ : H) : ℝ :=
  Real.log (M μ X q τ)

/-- The tilted (posterior) mean `𝔼[x_R ∣ τ] = meanNum(τ) / M(τ)` — the map `p:mean` calls `∇Λ`. -/
noncomputable def condMean (μ : Measure α) (X : α → H) (q : α → ℝ) (τ : H) : H :=
  (M μ X q τ)⁻¹ • meanNum μ X q τ

/-! ## Pointwise facts -/

omit [MeasurableSpace α] [CompleteSpace H] in
/-- The integrand is Fréchet-differentiable in the natural parameter, pointwise in the sample,
with derivative `exp(⟪τ, X a⟫ − q a) • ⟪X a, ·⟫`. -/
lemma hasFDerivAt_integrand (X : α → H) (q : α → ℝ) (a : α) (τ : H) :
    HasFDerivAt (fun τ : H => Real.exp (⟪τ, X a⟫_ℝ - q a))
      (Real.exp (⟪τ, X a⟫_ℝ - q a) • innerSL ℝ (X a)) τ := by
  have h1 : HasFDerivAt (fun w : H => ⟪X a, w⟫_ℝ - q a) (innerSL ℝ (X a)) τ :=
    (innerSL ℝ (X a)).hasFDerivAt.sub_const (q a)
  have h2 := h1.exp
  have hfun : (fun w : H => Real.exp (⟪w, X a⟫_ℝ - q a))
      = fun w : H => Real.exp (⟪X a, w⟫_ℝ - q a) := by
    funext w; rw [real_inner_comm]
  rw [hfun, real_inner_comm]
  exact h2

omit [CompleteSpace H] in
/-- Measurability of the integrand, from measurability of the pieces. -/
lemma aestronglyMeasurable_integrand {X : α → H} {q : α → ℝ}
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ) (τ : H) :
    AEStronglyMeasurable (fun a => Real.exp (⟪τ, X a⟫_ℝ - q a)) μ := by
  have h1 : AEStronglyMeasurable (fun a => ⟪τ, X a⟫_ℝ) μ :=
    (Continuous.inner continuous_const continuous_id).comp_aestronglyMeasurable hX
  exact Real.continuous_exp.comp_aestronglyMeasurable (h1.sub hq)

omit [CompleteSpace H] in
/-- **Positivity of the partition function** (so `log` is differentiable there): proved from
`μ ≠ 0` and integrability, not assumed — the integrand is strictly positive. -/
theorem M_pos {X : α → H} {q : α → ℝ} {τ : H}
    (hM_int : Integrable (fun a => Real.exp (⟪τ, X a⟫_ℝ - q a)) μ) (hμ : μ ≠ 0) :
    0 < M μ X q τ := by
  have hsupp : Function.support (fun a => Real.exp (⟪τ, X a⟫_ℝ - q a)) = Set.univ :=
    Set.eq_univ_iff_forall.mpr fun a => (Real.exp_pos _).ne'
  have h : 0 < ∫ a, Real.exp (⟪τ, X a⟫_ℝ - q a) ∂μ := by
    rw [integral_pos_iff_support_of_nonneg (fun a => (Real.exp_pos _).le) hM_int, hsupp]
    exact Measure.measure_univ_pos.mpr hμ
  exact h

/-! ## Integrability of the derivative integrands (from the envelope) -/

omit [CompleteSpace H] in
/-- The vector integrand `a ↦ exp(⟪τ₀, X a⟫ − q a) • X a` is integrable under the envelope. -/
lemma integrable_smul_coordinate {X : α → H} {q : α → ℝ} {τ₀ : H} {s : Set H}
    (hs : s ∈ 𝓝 τ₀) (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    {bound : α → ℝ}
    (h_bound : ∀ᵐ a ∂μ, ∀ τ ∈ s, Real.exp (⟪τ, X a⟫_ℝ - q a) * ‖X a‖ ≤ bound a)
    (bound_int : Integrable bound μ) :
    Integrable (fun a => Real.exp (⟪τ₀, X a⟫_ℝ - q a) • X a) μ := by
  refine bound_int.mono' ((aestronglyMeasurable_integrand hX hq τ₀).smul hX) ?_
  filter_upwards [h_bound] with a ha
  rw [norm_smul, Real.norm_eq_abs, Real.abs_exp]
  exact ha τ₀ (mem_of_mem_nhds hs)

omit [CompleteSpace H] in
/-- The operator-valued integrand `a ↦ exp(⟪τ₀, X a⟫ − q a) • innerSL ℝ (X a)` is integrable
under the same envelope (`innerSL` is an isometry). -/
lemma integrable_smul_innerSL {X : α → H} {q : α → ℝ} {τ₀ : H} {s : Set H}
    (hs : s ∈ 𝓝 τ₀) (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    {bound : α → ℝ}
    (h_bound : ∀ᵐ a ∂μ, ∀ τ ∈ s, Real.exp (⟪τ, X a⟫_ℝ - q a) * ‖X a‖ ≤ bound a)
    (bound_int : Integrable bound μ) :
    Integrable (fun a => Real.exp (⟪τ₀, X a⟫_ℝ - q a) • innerSL ℝ (X a)) μ := by
  have hSL : AEStronglyMeasurable (fun a => innerSL ℝ (X a)) μ :=
    (innerSL ℝ (E := H)).continuous.comp_aestronglyMeasurable hX
  refine bound_int.mono' ((aestronglyMeasurable_integrand hX hq τ₀).smul hSL) ?_
  filter_upwards [h_bound] with a ha
  rw [norm_smul, Real.norm_eq_abs, Real.abs_exp, innerSL_apply_norm]
  exact ha τ₀ (mem_of_mem_nhds hs)

/-! ## The inner-product identification of the Fréchet derivative -/

/-- **The Fréchet derivative is the dual of the gradient**: `∫ c(a) • ⟪X a, ·⟫ dμ` is
`toDual (∫ c(a) • X a dμ)`.  This is the identification that lets the parametric-integral
derivative be read as a `HasGradientAt`. -/
lemma integral_smul_innerSL_eq_toDual {c : α → ℝ} {X : α → H}
    (hv : Integrable (fun a => c a • X a) μ)
    (hL : Integrable (fun a => c a • innerSL ℝ (X a)) μ) :
    ∫ a, c a • innerSL ℝ (X a) ∂μ
      = InnerProductSpace.toDual ℝ H (∫ a, c a • X a ∂μ) := by
  ext w
  rw [ContinuousLinearMap.integral_apply hL w, InnerProductSpace.toDual_apply_apply,
    real_inner_comm, ← integral_inner hv w]
  refine integral_congr_ae (Eventually.of_forall fun a => ?_)
  simp [innerSL_apply_apply, smul_eq_mul, real_inner_smul_right, real_inner_comm]

/-! ## Differentiation under the integral -/

omit [CompleteSpace H] in
/-- **The partition function is Fréchet-differentiable**, by mathlib's dominated
differentiation-under-the-integral (`hasFDerivAt_integral_of_dominated_of_fderiv_le`): the envelope
hypothesis is exactly that lemma's `‖F' τ a‖ ≤ bound a`, since
`‖exp(⟪τ, X a⟫ − q a) • innerSL ℝ (X a)‖ = exp(⟪τ, X a⟫ − q a) * ‖X a‖`. -/
theorem hasFDerivAt_M (X : α → H) (q : α → ℝ) {τ₀ : H} {s : Set H} (hs : s ∈ 𝓝 τ₀)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    (hM_int : Integrable (fun a => Real.exp (⟪τ₀, X a⟫_ℝ - q a)) μ)
    {bound : α → ℝ}
    (h_bound : ∀ᵐ a ∂μ, ∀ τ ∈ s, Real.exp (⟪τ, X a⟫_ℝ - q a) * ‖X a‖ ≤ bound a)
    (bound_int : Integrable bound μ) :
    HasFDerivAt (M μ X q)
      (∫ a, Real.exp (⟪τ₀, X a⟫_ℝ - q a) • innerSL ℝ (X a) ∂μ) τ₀ := by
  have hF_meas : ∀ᶠ τ in 𝓝 τ₀,
      AEStronglyMeasurable (fun a => Real.exp (⟪τ, X a⟫_ℝ - q a)) μ :=
    Eventually.of_forall fun τ => aestronglyMeasurable_integrand hX hq τ
  have hSL : AEStronglyMeasurable (fun a => innerSL ℝ (X a)) μ :=
    (innerSL ℝ (E := H)).continuous.comp_aestronglyMeasurable hX
  have hF'_meas :
      AEStronglyMeasurable (fun a => Real.exp (⟪τ₀, X a⟫_ℝ - q a) • innerSL ℝ (X a)) μ :=
    (aestronglyMeasurable_integrand hX hq τ₀).smul hSL
  have h_bound' : ∀ᵐ a ∂μ, ∀ τ ∈ s,
      ‖Real.exp (⟪τ, X a⟫_ℝ - q a) • innerSL ℝ (X a)‖ ≤ bound a := by
    filter_upwards [h_bound] with a ha τ hτ
    rw [norm_smul, Real.norm_eq_abs, Real.abs_exp, innerSL_apply_norm]
    exact ha τ hτ
  have h_diff : ∀ᵐ a ∂μ, ∀ τ ∈ s,
      HasFDerivAt (fun τ : H => Real.exp (⟪τ, X a⟫_ℝ - q a))
        (Real.exp (⟪τ, X a⟫_ℝ - q a) • innerSL ℝ (X a)) τ :=
    Eventually.of_forall fun a τ _ => hasFDerivAt_integrand X q a τ
  exact hasFDerivAt_integral_of_dominated_of_fderiv_le hs hF_meas hM_int hF'_meas
    h_bound' bound_int h_diff

/-- **The partition function has a gradient** — the unnormalized tilted mean.  The Fréchet form of
`hasFDerivAt_M` is converted through `integral_smul_innerSL_eq_toDual`. -/
theorem hasGradientAt_M (X : α → H) (q : α → ℝ) {τ₀ : H} {s : Set H} (hs : s ∈ 𝓝 τ₀)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    (hM_int : Integrable (fun a => Real.exp (⟪τ₀, X a⟫_ℝ - q a)) μ)
    {bound : α → ℝ}
    (h_bound : ∀ᵐ a ∂μ, ∀ τ ∈ s, Real.exp (⟪τ, X a⟫_ℝ - q a) * ‖X a‖ ≤ bound a)
    (bound_int : Integrable bound μ) :
    HasGradientAt (M μ X q) (meanNum μ X q τ₀) τ₀ := by
  have hv := integrable_smul_coordinate hs hX hq h_bound bound_int
  have hL := integrable_smul_innerSL hs hX hq h_bound bound_int
  have key : InnerProductSpace.toDual ℝ H (meanNum μ X q τ₀)
      = ∫ a, Real.exp (⟪τ₀, X a⟫_ℝ - q a) • innerSL ℝ (X a) ∂μ :=
    (integral_smul_innerSL_eq_toDual hv hL).symm
  rw [hasGradientAt_iff_hasFDerivAt, key]
  exact hasFDerivAt_M X q hs hX hq hM_int h_bound bound_int

/-- **`p:mean`'s multivariate smoothness input, machine-checked.**  Under the domination
hypotheses, the log-partition `Λ(τ) = log ∫ exp(⟪τ, X a⟫ − q a) dμ` has at `τ₀` the gradient

    ∇Λ(τ₀) = (∫ exp(⟪τ₀, X a⟫ − q a) • X a dμ) / M(τ₀) = condMean τ₀,

the tilted (posterior) mean `𝔼[x_R ∣ τ₀]`.  Positivity of `M` is supplied by `M_pos` from
`μ ≠ 0`, so `log` differentiates with no extra nondegeneracy hypothesis. -/
theorem hasGradientAt_logPartition (X : α → H) (q : α → ℝ) {τ₀ : H} {s : Set H}
    (hs : s ∈ 𝓝 τ₀)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    (hM_int : Integrable (fun a => Real.exp (⟪τ₀, X a⟫_ℝ - q a)) μ)
    {bound : α → ℝ}
    (h_bound : ∀ᵐ a ∂μ, ∀ τ ∈ s, Real.exp (⟪τ, X a⟫_ℝ - q a) * ‖X a‖ ≤ bound a)
    (bound_int : Integrable bound μ) (hμ : μ ≠ 0) :
    HasGradientAt (Λ μ X q) (condMean μ X q τ₀) τ₀ := by
  have hM_pos : 0 < M μ X q τ₀ := M_pos hM_int hμ
  have h1 : HasFDerivAt (M μ X q)
      (InnerProductSpace.toDual ℝ H (meanNum μ X q τ₀)) τ₀ :=
    (hasGradientAt_M X q hs hX hq hM_int h_bound bound_int).hasFDerivAt
  have h2 := h1.log hM_pos.ne'
  have key : InnerProductSpace.toDual ℝ H (condMean μ X q τ₀)
      = (M μ X q τ₀)⁻¹ • InnerProductSpace.toDual ℝ H (meanNum μ X q τ₀) := by
    ext w
    simp [condMean, InnerProductSpace.toDual_apply_apply, smul_eq_mul]
  rw [hasGradientAt_iff_hasFDerivAt, key]
  exact h2

/-! ## The exact shape `posterior_mean_collapse_of_strictConvex` consumes -/

/-- The `∀ s, HasGradientAt Λfun (Λ s) s` form: under a locally-uniform integrable envelope at
**every** natural parameter (the appendix's "dominated locally uniformly because `q` is positive
definite"), the log-partition has the tilted mean as gradient everywhere.  This is verbatim the
`hΛ` hypothesis of `PosteriorMean.posterior_mean_collapse_of_strictConvex`. -/
theorem hasGradientAt_logPartition_forall (X : α → H) (q : α → ℝ)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ) (hμ : μ ≠ 0)
    (dom : ∀ τ₀ : H, ∃ s ∈ 𝓝 τ₀, ∃ bound : α → ℝ,
      Integrable (fun a => Real.exp (⟪τ₀, X a⟫_ℝ - q a)) μ ∧
      (∀ᵐ a ∂μ, ∀ τ ∈ s, Real.exp (⟪τ, X a⟫_ℝ - q a) * ‖X a‖ ≤ bound a) ∧
      Integrable bound μ) :
    ∀ τ, HasGradientAt (Λ μ X q) (condMean μ X q τ) τ := by
  intro τ
  obtain ⟨s, hs, bound, hM_int, h_bound, bound_int⟩ := dom τ
  exact hasGradientAt_logPartition X q hs hX hq hM_int h_bound bound_int hμ

/-- **The composition, demonstrated** (importing `PosteriorMeanRegularity`, not editing it):
`p:mean`'s collapse with the gradient hypothesis *discharged* by dominated differentiation.
If the archive's reconstruction factors through the resolved statistic (`hfac`), its resolved
component is the tilted mean of the log-partition built here (`hres`), and the log-partition is
strictly convex (`hconv` — the paper's affine-span condition, the one analytic input still taken
as a hypothesis, exactly as in `PosteriorMeanRegularity`), then equal resolved components force
equal reconstructions: the blind-fiber conditional is a Dirac.  The former `hΛ` hypothesis is now
`hasGradientAt_logPartition_forall`. -/
theorem posterior_mean_collapse_of_logPartition
    {𝓨 W : Type*} (xhat : 𝓨 → W) (T : 𝓨 → H) (ψ : H → W) (PR : W → H)
    (X : α → H) (q : α → ℝ)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ) (hμ : μ ≠ 0)
    (dom : ∀ τ₀ : H, ∃ s ∈ 𝓝 τ₀, ∃ bound : α → ℝ,
      Integrable (fun a => Real.exp (⟪τ₀, X a⟫_ℝ - q a)) μ ∧
      (∀ᵐ a ∂μ, ∀ τ ∈ s, Real.exp (⟪τ, X a⟫_ℝ - q a) * ‖X a‖ ≤ bound a) ∧
      Integrable bound μ)
    (hconv : StrictConvexOn ℝ Set.univ (Λ μ X q))
    (hfac : ∀ y, xhat y = ψ (T y))
    (hres : ∀ y, PR (xhat y) = condMean μ X q (T y))
    {y y' : 𝓨} (h : PR (xhat y) = PR (xhat y')) :
    xhat y = xhat y' :=
  PosteriorMean.posterior_mean_collapse_of_strictConvex xhat T ψ PR (condMean μ X q)
    hfac hres hconv (hasGradientAt_logPartition_forall X q hX hq hμ dom) h

end LogPartitionGradient

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms LogPartitionGradient.hasFDerivAt_integrand
#print axioms LogPartitionGradient.M_pos
#print axioms LogPartitionGradient.integral_smul_innerSL_eq_toDual
#print axioms LogPartitionGradient.hasFDerivAt_M
#print axioms LogPartitionGradient.hasGradientAt_M
#print axioms LogPartitionGradient.hasGradientAt_logPartition
#print axioms LogPartitionGradient.hasGradientAt_logPartition_forall
#print axioms LogPartitionGradient.posterior_mean_collapse_of_logPartition
