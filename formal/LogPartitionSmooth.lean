/-
# `p:mean`'s remaining input: smoothness of the exponential-family log-partition

`PosteriorMeanRegularity.lean` supplies the *injectivity* of `∇Λ` from strict convexity.  The paper's
`app:formal` listed the other half — the **existence** of `∇Λ`, "under dominated differentiation" —
as not machine-checked.  This file supplies it.

For the natural parameter `τ` and sufficient statistic `T`, the partition function and log-partition
are

    M(τ) = ∫ exp(τ · T a) dμ(a),    Λ(τ) = log M(τ),

and differentiating under the integral gives `Λ'(τ) = (∫ T a · exp(τ · T a) dμ) / M(τ)` — the
exponential family's mean-value map, whose injectivity is what `p:mean` consumes.

The domination hypotheses are taken as explicit hypotheses, exactly as the appendix states them
("under dominated differentiation"), and discharged against mathlib's
`hasDerivAt_integral_of_dominated_loc_of_deriv_le`.  Nothing is assumed that the appendix does not
assume; what is added is that the passage from those hypotheses to `HasDerivAt Λ` is now checked
rather than asserted.

No `sorry`/`admit`; `#print axioms` on each result at the end.
-/
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

open MeasureTheory Filter
open scoped Topology

namespace LogPartition

/-- The integrand is differentiable in the natural parameter, pointwise in the sample. -/
lemma hasDerivAt_integrand (T : α → ℝ) (a : α) (τ : ℝ) :
    HasDerivAt (fun τ => Real.exp (τ * T a)) (T a * Real.exp (τ * T a)) τ := by
  have h : HasDerivAt (fun τ : ℝ => τ * T a) (T a) τ := by
    simpa using (hasDerivAt_id τ).mul_const (T a)
  simpa [mul_comm] using h.exp

variable {α : Type*} [MeasurableSpace α] {μ : Measure α}

/-- The partition function of a one-parameter exponential family with sufficient statistic `T`. -/
noncomputable def M (μ : Measure α) (T : α → ℝ) (τ : ℝ) : ℝ := ∫ a, Real.exp (τ * T a) ∂μ

/-- The log-partition. -/
noncomputable def Λ (μ : Measure α) (T : α → ℝ) (τ : ℝ) : ℝ := Real.log (M μ T τ)

/-- **The partition function is differentiable**, with the derivative obtained by differentiating
under the integral sign. -/
theorem hasDerivAt_M (T : α → ℝ) {τ₀ : ℝ} {s : Set ℝ} (hs : s ∈ 𝓝 τ₀) {bound : α → ℝ}
    (hF_meas : ∀ᶠ τ in 𝓝 τ₀, AEStronglyMeasurable (fun a => Real.exp (τ * T a)) μ)
    (hF_int : Integrable (fun a => Real.exp (τ₀ * T a)) μ)
    (hF'_meas : AEStronglyMeasurable (fun a => T a * Real.exp (τ₀ * T a)) μ)
    (h_bound : ∀ᵐ a ∂μ, ∀ τ ∈ s, ‖T a * Real.exp (τ * T a)‖ ≤ bound a)
    (bound_int : Integrable bound μ) :
    HasDerivAt (M μ T) (∫ a, T a * Real.exp (τ₀ * T a) ∂μ) τ₀ := by
  have h_diff : ∀ᵐ a ∂μ, ∀ τ ∈ s,
      HasDerivAt (fun τ => Real.exp (τ * T a)) (T a * Real.exp (τ * T a)) τ := by
    filter_upwards with a τ _
    exact hasDerivAt_integrand T a τ
  exact (hasDerivAt_integral_of_dominated_loc_of_deriv_le hs hF_meas hF_int hF'_meas
    h_bound bound_int h_diff).2

/-- **`p:mean`'s smoothness input, machine-checked.**  Under the appendix's domination hypotheses and
a nondegenerate partition function, the log-partition is differentiable with

    Λ'(τ₀) = (∫ T a · exp(τ₀ · T a) dμ) / M(τ₀),

the exponential family's mean-value map.  Combined with `PosteriorMeanRegularity`'s injectivity from
strict convexity, this closes the regularity `p:mean` consumes. -/
theorem hasDerivAt_logPartition (T : α → ℝ) {τ₀ : ℝ} {s : Set ℝ} (hs : s ∈ 𝓝 τ₀) {bound : α → ℝ}
    (hF_meas : ∀ᶠ τ in 𝓝 τ₀, AEStronglyMeasurable (fun a => Real.exp (τ * T a)) μ)
    (hF_int : Integrable (fun a => Real.exp (τ₀ * T a)) μ)
    (hF'_meas : AEStronglyMeasurable (fun a => T a * Real.exp (τ₀ * T a)) μ)
    (h_bound : ∀ᵐ a ∂μ, ∀ τ ∈ s, ‖T a * Real.exp (τ * T a)‖ ≤ bound a)
    (bound_int : Integrable bound μ)
    (hpos : M μ T τ₀ ≠ 0) :
    HasDerivAt (Λ μ T) ((∫ a, T a * Real.exp (τ₀ * T a) ∂μ) / M μ T τ₀) τ₀ :=
  (hasDerivAt_M T hs hF_meas hF_int hF'_meas h_bound bound_int).log hpos

end LogPartition

-- #print axioms audit (p:mean smoothness)
#print axioms LogPartition.hasDerivAt_integrand
#print axioms LogPartition.hasDerivAt_M
#print axioms LogPartition.hasDerivAt_logPartition
