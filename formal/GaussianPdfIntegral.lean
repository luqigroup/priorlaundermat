import Mathlib

open MeasureTheory ProbabilityTheory Real
open scoped NNReal

namespace GaussianPdf

/-- The integral of `pdf₁² / pdf₂` for two Gaussians with common variance `v` and means
`m₁, m₂` equals `exp ((m₁ - m₂)² / v)`.

Pointwise the integrand equals `exp ((m₁ - m₂)² / v) · gaussianPDFReal (2 m₁ - m₂) v`, so the
integral collapses to the constant times the total mass `1` of a Gaussian density. -/
theorem integral_gaussianPDFReal_sq_div_gaussianPDFReal
    (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    ∫ x, gaussianPDFReal m₁ v x ^ 2 / gaussianPDFReal m₂ v x ∂volume
      = Real.exp ((m₁ - m₂) ^ 2 / (v : ℝ)) := by
  -- Positivity facts for the variance coercion.
  have hvR : (0 : ℝ) < (v : ℝ) :=
    lt_of_le_of_ne v.coe_nonneg (Ne.symm (NNReal.coe_ne_zero.mpr hv))
  have hvR' : (v : ℝ) ≠ 0 := hvR.ne'
  -- Pointwise identity: the integrand is a constant multiple of a Gaussian density.
  have hpt : ∀ x : ℝ, gaussianPDFReal m₁ v x ^ 2 / gaussianPDFReal m₂ v x
      = Real.exp ((m₁ - m₂) ^ 2 / (v : ℝ)) * gaussianPDFReal (2 * m₁ - m₂) v x := by
    intro x
    have hden : gaussianPDFReal m₂ v x ≠ 0 := (gaussianPDFReal_pos m₂ v x hv).ne'
    rw [div_eq_iff hden]
    simp only [gaussianPDFReal]
    set s : ℝ := (Real.sqrt (2 * π * (v : ℝ)))⁻¹ with hsdef
    -- Combine the exponential factors on each side.
    have hL : ∀ a : ℝ, (s * Real.exp a) ^ 2 = s ^ 2 * Real.exp (a + a) := by
      intro a; rw [mul_pow, pow_two (Real.exp a), ← Real.exp_add]
    have hR : ∀ c d b : ℝ,
        Real.exp c * (s * Real.exp d) * (s * Real.exp b) = s ^ 2 * Real.exp (c + d + b) := by
      intro c d b; rw [Real.exp_add, Real.exp_add]; ring
    rw [hL, hR]
    congr 1
    rw [Real.exp_eq_exp]
    field_simp
    ring
  -- Reduce the integral to a constant times the total mass of a Gaussian density.
  simp_rw [hpt]
  rw [integral_const_mul, integral_gaussianPDFReal_eq_one (2 * m₁ - m₂) hv, mul_one]

end GaussianPdf

#print axioms GaussianPdf.integral_gaussianPDFReal_sq_div_gaussianPDFReal
