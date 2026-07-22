import Mathlib
import ChiSquaredTV
import GaussianPdfIntegral

/-!
# χ² of two same-variance Gaussians, and the quantitative near-null undetectability

This is the final wiring lemma of the near-null layer. It plugs the closed-form Gaussian pdf-ratio
integral (`GaussianPdfIntegral.lean`) into the χ²-controls-TV Le Cam engine (`ChiSquaredTV.lean`) to
obtain, sorry-free, the quantitative statement `p:nearnull`: as the two means converge the total
testing error tends to `1`, so no test distinguishes two Gaussians whose means differ by a near-null
amount.

## Results
* `chi2_gaussianReal` — `χ²(N(m₁,v) ‖ N(m₂,v)) = exp((m₁-m₂)²/v) - 1`.
* `integrable_chi2_gaussianReal` — the finiteness side-condition feeding the Le Cam bound.
* `NearNull.gaussianReal_near_null_undetectable` — the payoff:
  `1 - √(exp((m₁-m₂)²/v) - 1) ≤ P₁.real A + P₂.real Aᶜ` for every test `A`.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each of the three results lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory ProbabilityTheory Real
open scoped NNReal ENNReal

/-- Pointwise: `pdf₁² / pdf₂` is a constant multiple of the Gaussian density with mean `2m₁ - m₂`.
The same identity underlies `GaussianPdf.integral_gaussianPDFReal_sq_div_gaussianPDFReal`; reproved
here to feed integrability. -/
private lemma gaussianPDFReal_sq_div_eq (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    gaussianPDFReal m₁ v x ^ 2 / gaussianPDFReal m₂ v x
      = Real.exp ((m₁ - m₂) ^ 2 / (v : ℝ)) * gaussianPDFReal (2 * m₁ - m₂) v x := by
  have hden : gaussianPDFReal m₂ v x ≠ 0 := (gaussianPDFReal_pos m₂ v x hv).ne'
  rw [div_eq_iff hden]
  simp only [gaussianPDFReal]
  set s : ℝ := (Real.sqrt (2 * π * (v : ℝ)))⁻¹ with hsdef
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

/-- `pdf₁² / pdf₂` is integrable: it is a constant times a Gaussian density. -/
private lemma integrable_gaussianPDFReal_sq_div (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    Integrable (fun x => gaussianPDFReal m₁ v x ^ 2 / gaussianPDFReal m₂ v x) volume :=
  ((integrable_gaussianPDFReal (2 * m₁ - m₂) v).const_mul
      (Real.exp ((m₁ - m₂) ^ 2 / (v : ℝ)))).congr
    (ae_of_all _ fun x => (gaussianPDFReal_sq_div_eq m₁ m₂ hv x).symm)

/-- Expand the change-of-measure integrand: `p₂ · (p₁/p₂ − 1)² = p₁²/p₂ − 2 p₁ + p₂`. -/
private lemma gaussianPDFReal_density_expand (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    gaussianPDFReal m₂ v x * (gaussianPDFReal m₁ v x / gaussianPDFReal m₂ v x - 1) ^ 2
      = gaussianPDFReal m₁ v x ^ 2 / gaussianPDFReal m₂ v x
          - 2 * gaussianPDFReal m₁ v x + gaussianPDFReal m₂ v x := by
  have hden : gaussianPDFReal m₂ v x ≠ 0 := (gaussianPDFReal_pos m₂ v x hv).ne'
  field_simp
  ring

/-- The Radon–Nikodym density `d(N(m₁,v))/d(N(m₂,v))` is a.e. the pointwise pdf ratio `p₁/p₂`.
Chain rule `Measure.rnDeriv_mul_rnDeriv` (through `volume`) plus `rnDeriv_gaussianReal`. -/
private lemma rnDeriv_toReal_ae (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    (fun x => ((gaussianReal m₁ v).rnDeriv (gaussianReal m₂ v) x).toReal)
      =ᵐ[gaussianReal m₂ v]
    (fun x => gaussianPDFReal m₁ v x / gaussianPDFReal m₂ v x) := by
  have hP1v : gaussianReal m₁ v ≪ volume := gaussianReal_absolutelyContinuous m₁ hv
  have hP2v : gaussianReal m₂ v ≪ volume := gaussianReal_absolutelyContinuous m₂ hv
  have hvP2 : volume ≪ gaussianReal m₂ v := gaussianReal_absolutelyContinuous' m₂ hv
  have hP1P2 : gaussianReal m₁ v ≪ gaussianReal m₂ v := hP1v.trans hvP2
  -- a.e.[volume] version, then transferred to `gaussianReal m₂ v` via `hP2v`
  refine hP2v.ae_eq ?_
  have hchain := Measure.rnDeriv_mul_rnDeriv (μ := gaussianReal m₁ v)
    (ν := gaussianReal m₂ v) (κ := volume) hP1P2
  have h1v := rnDeriv_gaussianReal m₁ v
  have h2v := rnDeriv_gaussianReal m₂ v
  filter_upwards [hchain, h1v, h2v] with x hx h1 h2
  rw [Pi.mul_apply, h1, h2] at hx
  -- hx : (rnDeriv) x * gaussianPDF m₂ v x = gaussianPDF m₁ v x  (in ℝ≥0∞)
  have hpos2 : (0 : ℝ) < gaussianPDFReal m₂ v x := gaussianPDFReal_pos m₂ v x hv
  have hxR := congrArg ENNReal.toReal hx
  simp only [ENNReal.toReal_mul, toReal_gaussianPDF] at hxR
  -- hxR : (rnDeriv) x |>.toReal * gaussianPDFReal m₂ v x = gaussianPDFReal m₁ v x
  rw [eq_div_iff hpos2.ne']
  exact hxR

/-- **(A) χ² of two same-variance Gaussians.**
`χ²(N(m₁,v) ‖ N(m₂,v)) = exp((m₁ - m₂)² / v) - 1`. -/
theorem chi2_gaussianReal (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    NearNull.chi2 (gaussianReal m₁ v) (gaussianReal m₂ v)
      = Real.exp ((m₁ - m₂) ^ 2 / (v : ℝ)) - 1 := by
  -- rewrite the integrand of χ² by the pdf ratio, then change measure to `volume`
  have hstep1 : NearNull.chi2 (gaussianReal m₁ v) (gaussianReal m₂ v)
      = ∫ x, (gaussianPDFReal m₁ v x / gaussianPDFReal m₂ v x - 1) ^ 2 ∂(gaussianReal m₂ v) := by
    unfold NearNull.chi2
    apply integral_congr_ae
    filter_upwards [rnDeriv_toReal_ae m₁ m₂ hv] with x hx
    rw [hx]
  rw [hstep1, integral_gaussianReal_eq_integral_smul hv]
  simp only [smul_eq_mul]
  simp_rw [gaussianPDFReal_density_expand m₁ m₂ hv]
  -- split the three integrable summands and evaluate each in closed form
  have hIa := integrable_gaussianPDFReal_sq_div m₁ m₂ hv
  have hIb : Integrable (fun x => 2 * gaussianPDFReal m₁ v x) volume :=
    (integrable_gaussianPDFReal m₁ v).const_mul 2
  have hIc : Integrable (fun x => gaussianPDFReal m₂ v x) volume :=
    integrable_gaussianPDFReal m₂ v
  -- term-mode splits (unify integrands up to defeq, unlike `rw`)
  have h1 : ∫ x, (gaussianPDFReal m₁ v x ^ 2 / gaussianPDFReal m₂ v x
        - 2 * gaussianPDFReal m₁ v x + gaussianPDFReal m₂ v x) ∂volume
      = (∫ x, (gaussianPDFReal m₁ v x ^ 2 / gaussianPDFReal m₂ v x
          - 2 * gaussianPDFReal m₁ v x) ∂volume) + ∫ x, gaussianPDFReal m₂ v x ∂volume :=
    integral_add (hIa.sub hIb) hIc
  have h2 : ∫ x, (gaussianPDFReal m₁ v x ^ 2 / gaussianPDFReal m₂ v x
        - 2 * gaussianPDFReal m₁ v x) ∂volume
      = (∫ x, gaussianPDFReal m₁ v x ^ 2 / gaussianPDFReal m₂ v x ∂volume)
        - ∫ x, 2 * gaussianPDFReal m₁ v x ∂volume :=
    integral_sub hIa hIb
  rw [h1, h2, integral_const_mul,
    GaussianPdf.integral_gaussianPDFReal_sq_div_gaussianPDFReal m₁ m₂ hv,
    integral_gaussianPDFReal_eq_one m₁ hv, integral_gaussianPDFReal_eq_one m₂ hv]
  ring

/-- **(B) Finiteness of the χ² integrand** (the side-condition feeding the Le Cam bound). -/
theorem integrable_chi2_gaussianReal (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    Integrable (fun x => (((gaussianReal m₁ v).rnDeriv (gaussianReal m₂ v) x).toReal - 1) ^ 2)
      (gaussianReal m₂ v) := by
  -- move to the pdf-ratio integrand, change measure to `volume`, and split into integrable pieces
  have hG : Integrable
      (fun x => (gaussianPDFReal m₁ v x / gaussianPDFReal m₂ v x - 1) ^ 2) (gaussianReal m₂ v) := by
    rw [gaussianReal_of_var_ne_zero m₂ hv,
      integrable_withDensity_iff_integrable_smul' (measurable_gaussianPDF m₂ v)
        (ae_of_all _ fun _ => gaussianPDF_lt_top)]
    simp only [smul_eq_mul, toReal_gaussianPDF]
    exact (((integrable_gaussianPDFReal_sq_div m₁ m₂ hv).sub
        ((integrable_gaussianPDFReal m₁ v).const_mul 2)).add
        (integrable_gaussianPDFReal m₂ v)).congr
      (ae_of_all _ fun x => (gaussianPDFReal_density_expand m₁ m₂ hv x).symm)
  refine hG.congr ?_
  filter_upwards [rnDeriv_toReal_ae m₁ m₂ hv] with x hx
  rw [hx]

namespace NearNull

/-- **(C) Quantitative near-null undetectability.** Instantiating the Le Cam two-point χ² bound at the
two same-variance Gaussians: for every test `A`,
`1 - √(exp((m₁ - m₂)²/v) - 1) ≤ P₁.real A + P₂.real Aᶜ`. As `m₁ → m₂` the left side `→ 1`, so no test
distinguishes two Gaussians whose means differ by a near-null amount. -/
theorem gaussianReal_near_null_undetectable (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0)
    {A : Set ℝ} (hA : MeasurableSet A) :
    1 - Real.sqrt (Real.exp ((m₁ - m₂) ^ 2 / (v : ℝ)) - 1)
      ≤ (gaussianReal m₁ v).real A + (gaussianReal m₂ v).real Aᶜ := by
  have hP1v : gaussianReal m₁ v ≪ volume := gaussianReal_absolutelyContinuous m₁ hv
  have hvP2 : volume ≪ gaussianReal m₂ v := gaussianReal_absolutelyContinuous' m₂ hv
  have hP1P2 : gaussianReal m₁ v ≪ gaussianReal m₂ v := hP1v.trans hvP2
  have h := NearNull.lecam_two_point_chi2 hP1P2 (integrable_chi2_gaussianReal m₁ m₂ hv) hA
  rwa [chi2_gaussianReal m₁ m₂ hv] at h

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms chi2_gaussianReal
#print axioms integrable_chi2_gaussianReal
#print axioms NearNull.gaussianReal_near_null_undetectable
