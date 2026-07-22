import Mathlib
import ChiSquaredTV
import GaussianPdfIntegral

/-!
# χ² of two same-mean Gaussians with mismatched variances, and near-null undetectability

Companion of `GaussianChiSquared.lean` (the mean-shift case). Here the two Gaussians share the mean
`m` but differ in variance (`v₁ ≠ v₂`). This is the **spread-misreport** near-null case — the paper's
primary one: a curated prior reports a variance `v₁` on the blind subspace that is only a near-null
amount away from the calibrated `v₂`, and no test detects the difference.

The only new analytic content is the pdf-ratio integral with mismatched variances: writing
`σ² := v₁ v₂ / (2 v₂ - v₁)` (positive when `v₁ < 2 v₂`), the pointwise ratio `p₁²/p₂` is a constant
multiple of a Gaussian density of variance `σ²`, so `∫ p₁²/p₂ = v₂ / √(v₁ (2 v₂ - v₁))`. Everything
else mirrors `GaussianChiSquared.lean` verbatim.

## Results
* `chi2_gaussianReal_var` —
  `χ²(N(m,v₁) ‖ N(m,v₂)) = v₂ / √(v₁ (2 v₂ - v₁)) - 1`.
* `integrable_chi2_gaussianReal_var` — the finiteness side-condition feeding the Le Cam bound.
* `NearNull.gaussianReal_var_near_null_undetectable` — the payoff:
  `1 - √(v₂/√(v₁ (2 v₂ - v₁)) - 1) ≤ P₁.real A + P₂.real Aᶜ` for every test `A`. As `v₁ → v₂` the χ²
  `→ 0`, so the bound `→ 1` and no test distinguishes the two spreads.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each of the three results lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory ProbabilityTheory Real
open scoped NNReal ENNReal

/-- Pointwise: `pdf₁² / pdf₂` (same mean `m`, variances `v₁`, `v₂`) is a constant multiple of the
Gaussian density of variance `w = v₁ v₂ / (2 v₂ - v₁)`. The constant is
`v₂ / √(v₁ (2 v₂ - v₁))`. -/
private lemma gaussianPDFReal_sq_div_eq_var (m : ℝ) {v₁ v₂ w : ℝ≥0}
    (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0) (hlt : (v₁ : ℝ) < 2 * v₂)
    (hwval : (w : ℝ) = (v₁ : ℝ) * v₂ / (2 * v₂ - v₁)) (x : ℝ) :
    gaussianPDFReal m v₁ x ^ 2 / gaussianPDFReal m v₂ x
      = ((v₂ : ℝ) / Real.sqrt ((v₁ : ℝ) * (2 * v₂ - v₁))) * gaussianPDFReal m w x := by
  have hv₁pos : (0 : ℝ) < v₁ := lt_of_le_of_ne v₁.coe_nonneg (Ne.symm (NNReal.coe_ne_zero.mpr hv₁))
  have hv₂pos : (0 : ℝ) < v₂ := lt_of_le_of_ne v₂.coe_nonneg (Ne.symm (NNReal.coe_ne_zero.mpr hv₂))
  have hdiffpos : (0 : ℝ) < 2 * v₂ - v₁ := by linarith
  have hdiffne : (2 * (v₂ : ℝ) - v₁) ≠ 0 := hdiffpos.ne'
  have hv₁ne : (v₁ : ℝ) ≠ 0 := hv₁pos.ne'
  have hv₂ne : (v₂ : ℝ) ≠ 0 := hv₂pos.ne'
  have hden : gaussianPDFReal m v₂ x ≠ 0 := (gaussianPDFReal_pos m v₂ x hv₂).ne'
  rw [div_eq_iff hden]
  simp only [gaussianPDFReal]
  set s₁ : ℝ := (Real.sqrt (2 * π * (v₁ : ℝ)))⁻¹ with hs₁
  set s₂ : ℝ := (Real.sqrt (2 * π * (v₂ : ℝ)))⁻¹ with hs₂
  set sw : ℝ := (Real.sqrt (2 * π * (w : ℝ)))⁻¹ with hsw
  have hL : ∀ a : ℝ, (s₁ * Real.exp a) ^ 2 = s₁ ^ 2 * Real.exp (a + a) := by
    intro a; rw [mul_pow, pow_two (Real.exp a), ← Real.exp_add]
  have hR : ∀ cc d b : ℝ,
      cc * (sw * Real.exp d) * (s₂ * Real.exp b) = cc * sw * s₂ * Real.exp (d + b) := by
    intro cc d b; rw [Real.exp_add]; ring
  have hexp : -(x - m) ^ 2 / (2 * (v₁ : ℝ)) + -(x - m) ^ 2 / (2 * (v₁ : ℝ))
      = -(x - m) ^ 2 / (2 * (w : ℝ)) + -(x - m) ^ 2 / (2 * (v₂ : ℝ)) := by
    rw [hwval]; field_simp; ring
  rw [hL, hR, hexp]
  congr 1
  rw [hs₁, hsw, hs₂]
  have hA : (0 : ℝ) ≤ (v₁ : ℝ) * (2 * v₂ - v₁) := mul_nonneg hv₁pos.le hdiffpos.le
  have hA2 : (0 : ℝ) ≤ (v₁ : ℝ) * (2 * v₂ - v₁) * (2 * π * (w : ℝ)) := mul_nonneg hA (by positivity)
  have hABC : Real.sqrt ((v₁ : ℝ) * (2 * v₂ - v₁)) * Real.sqrt (2 * π * (w : ℝ))
        * Real.sqrt (2 * π * (v₂ : ℝ)) = 2 * π * (v₁ : ℝ) * (v₂ : ℝ) := by
    rw [← Real.sqrt_mul hA, ← Real.sqrt_mul hA2,
      show (v₁ : ℝ) * (2 * v₂ - v₁) * (2 * π * (w : ℝ)) * (2 * π * (v₂ : ℝ))
          = (2 * π * (v₁ : ℝ) * (v₂ : ℝ)) ^ 2 by rw [hwval]; field_simp]
    exact Real.sqrt_sq (by positivity)
  rw [inv_pow, Real.sq_sqrt (by positivity : (0 : ℝ) ≤ 2 * π * (v₁ : ℝ)),
    show (v₂ : ℝ) / Real.sqrt ((v₁ : ℝ) * (2 * v₂ - v₁)) * (Real.sqrt (2 * π * (w : ℝ)))⁻¹
          * (Real.sqrt (2 * π * (v₂ : ℝ)))⁻¹
        = (v₂ : ℝ) / (Real.sqrt ((v₁ : ℝ) * (2 * v₂ - v₁)) * Real.sqrt (2 * π * (w : ℝ))
            * Real.sqrt (2 * π * (v₂ : ℝ))) by ring,
    hABC, inv_eq_one_div,
    div_eq_div_iff (mul_ne_zero (mul_ne_zero two_ne_zero Real.pi_ne_zero) hv₁ne)
      (mul_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero Real.pi_ne_zero) hv₁ne) hv₂ne)]
  ring

/-- The integral of `pdf₁² / pdf₂` (same mean, variances `v₁`, `v₂` with `v₁ < 2 v₂`) equals the
closed-form constant `v₂ / √(v₁ (2 v₂ - v₁))`. -/
private lemma integral_gaussianPDFReal_sq_div_var (m : ℝ) {v₁ v₂ : ℝ≥0}
    (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0) (hlt : (v₁ : ℝ) < 2 * v₂) :
    ∫ x, gaussianPDFReal m v₁ x ^ 2 / gaussianPDFReal m v₂ x ∂volume
      = (v₂ : ℝ) / Real.sqrt ((v₁ : ℝ) * (2 * v₂ - v₁)) := by
  have hv₁pos : (0 : ℝ) < v₁ := lt_of_le_of_ne v₁.coe_nonneg (Ne.symm (NNReal.coe_ne_zero.mpr hv₁))
  have hv₂pos : (0 : ℝ) < v₂ := lt_of_le_of_ne v₂.coe_nonneg (Ne.symm (NNReal.coe_ne_zero.mpr hv₂))
  have hdiffpos : (0 : ℝ) < 2 * v₂ - v₁ := by linarith
  have hσpos : (0 : ℝ) < (v₁ : ℝ) * v₂ / (2 * v₂ - v₁) := div_pos (mul_pos hv₁pos hv₂pos) hdiffpos
  have hwval : ((Real.toNNReal ((v₁ : ℝ) * v₂ / (2 * v₂ - v₁)) : ℝ≥0) : ℝ)
      = (v₁ : ℝ) * v₂ / (2 * v₂ - v₁) := Real.coe_toNNReal _ hσpos.le
  have hwne : Real.toNNReal ((v₁ : ℝ) * v₂ / (2 * v₂ - v₁)) ≠ 0 := (Real.toNNReal_pos.mpr hσpos).ne'
  simp_rw [gaussianPDFReal_sq_div_eq_var m hv₁ hv₂ hlt hwval]
  rw [integral_const_mul, integral_gaussianPDFReal_eq_one m hwne, mul_one]

/-- `pdf₁² / pdf₂` is integrable: it is a constant times a Gaussian density. -/
private lemma integrable_gaussianPDFReal_sq_div_var (m : ℝ) {v₁ v₂ : ℝ≥0}
    (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0) (hlt : (v₁ : ℝ) < 2 * v₂) :
    Integrable (fun x => gaussianPDFReal m v₁ x ^ 2 / gaussianPDFReal m v₂ x) volume := by
  have hv₁pos : (0 : ℝ) < v₁ := lt_of_le_of_ne v₁.coe_nonneg (Ne.symm (NNReal.coe_ne_zero.mpr hv₁))
  have hv₂pos : (0 : ℝ) < v₂ := lt_of_le_of_ne v₂.coe_nonneg (Ne.symm (NNReal.coe_ne_zero.mpr hv₂))
  have hdiffpos : (0 : ℝ) < 2 * v₂ - v₁ := by linarith
  have hσpos : (0 : ℝ) < (v₁ : ℝ) * v₂ / (2 * v₂ - v₁) := div_pos (mul_pos hv₁pos hv₂pos) hdiffpos
  have hwval : ((Real.toNNReal ((v₁ : ℝ) * v₂ / (2 * v₂ - v₁)) : ℝ≥0) : ℝ)
      = (v₁ : ℝ) * v₂ / (2 * v₂ - v₁) := Real.coe_toNNReal _ hσpos.le
  exact ((integrable_gaussianPDFReal m (Real.toNNReal ((v₁ : ℝ) * v₂ / (2 * v₂ - v₁)))).const_mul
      ((v₂ : ℝ) / Real.sqrt ((v₁ : ℝ) * (2 * v₂ - v₁)))).congr
    (ae_of_all _ fun x => (gaussianPDFReal_sq_div_eq_var m hv₁ hv₂ hlt hwval x).symm)

/-- Expand the change-of-measure integrand: `p₂ · (p₁/p₂ − 1)² = p₁²/p₂ − 2 p₁ + p₂`. -/
private lemma gaussianPDFReal_density_expand_var (m : ℝ) {v₁ v₂ : ℝ≥0} (hv₂ : v₂ ≠ 0) (x : ℝ) :
    gaussianPDFReal m v₂ x * (gaussianPDFReal m v₁ x / gaussianPDFReal m v₂ x - 1) ^ 2
      = gaussianPDFReal m v₁ x ^ 2 / gaussianPDFReal m v₂ x
          - 2 * gaussianPDFReal m v₁ x + gaussianPDFReal m v₂ x := by
  have hden : gaussianPDFReal m v₂ x ≠ 0 := (gaussianPDFReal_pos m v₂ x hv₂).ne'
  field_simp
  ring

/-- The Radon–Nikodym density `d(N(m,v₁))/d(N(m,v₂))` is a.e. the pointwise pdf ratio `p₁/p₂`.
Chain rule `Measure.rnDeriv_mul_rnDeriv` (through `volume`) plus `rnDeriv_gaussianReal`; the equal
means play no role. -/
private lemma rnDeriv_toReal_ae_var (m : ℝ) {v₁ v₂ : ℝ≥0} (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0) :
    (fun x => ((gaussianReal m v₁).rnDeriv (gaussianReal m v₂) x).toReal)
      =ᵐ[gaussianReal m v₂]
    (fun x => gaussianPDFReal m v₁ x / gaussianPDFReal m v₂ x) := by
  have hP1v : gaussianReal m v₁ ≪ volume := gaussianReal_absolutelyContinuous m hv₁
  have hP2v : gaussianReal m v₂ ≪ volume := gaussianReal_absolutelyContinuous m hv₂
  have hvP2 : volume ≪ gaussianReal m v₂ := gaussianReal_absolutelyContinuous' m hv₂
  have hP1P2 : gaussianReal m v₁ ≪ gaussianReal m v₂ := hP1v.trans hvP2
  refine hP2v.ae_eq ?_
  have hchain := Measure.rnDeriv_mul_rnDeriv (μ := gaussianReal m v₁)
    (ν := gaussianReal m v₂) (κ := volume) hP1P2
  have h1 := rnDeriv_gaussianReal m v₁
  have h2 := rnDeriv_gaussianReal m v₂
  filter_upwards [hchain, h1, h2] with x hx hh1 hh2
  rw [Pi.mul_apply, hh1, hh2] at hx
  have hpos2 : (0 : ℝ) < gaussianPDFReal m v₂ x := gaussianPDFReal_pos m v₂ x hv₂
  have hxR := congrArg ENNReal.toReal hx
  simp only [ENNReal.toReal_mul, toReal_gaussianPDF] at hxR
  rw [eq_div_iff hpos2.ne']
  exact hxR

/-- **(A) χ² of two same-mean Gaussians with mismatched variances.**
`χ²(N(m,v₁) ‖ N(m,v₂)) = v₂ / √(v₁ (2 v₂ - v₁)) - 1`. -/
theorem chi2_gaussianReal_var (m : ℝ) {v₁ v₂ : ℝ≥0} (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0)
    (hlt : (v₁ : ℝ) < 2 * v₂) :
    NearNull.chi2 (gaussianReal m v₁) (gaussianReal m v₂)
      = (v₂ : ℝ) / Real.sqrt ((v₁ : ℝ) * (2 * v₂ - v₁)) - 1 := by
  have hstep1 : NearNull.chi2 (gaussianReal m v₁) (gaussianReal m v₂)
      = ∫ x, (gaussianPDFReal m v₁ x / gaussianPDFReal m v₂ x - 1) ^ 2 ∂(gaussianReal m v₂) := by
    unfold NearNull.chi2
    apply integral_congr_ae
    filter_upwards [rnDeriv_toReal_ae_var m hv₁ hv₂] with x hx
    rw [hx]
  rw [hstep1, integral_gaussianReal_eq_integral_smul hv₂]
  simp only [smul_eq_mul]
  simp_rw [gaussianPDFReal_density_expand_var m hv₂]
  have hIa := integrable_gaussianPDFReal_sq_div_var m hv₁ hv₂ hlt
  have hIb : Integrable (fun x => 2 * gaussianPDFReal m v₁ x) volume :=
    (integrable_gaussianPDFReal m v₁).const_mul 2
  have hIc : Integrable (fun x => gaussianPDFReal m v₂ x) volume :=
    integrable_gaussianPDFReal m v₂
  have h1 : ∫ x, (gaussianPDFReal m v₁ x ^ 2 / gaussianPDFReal m v₂ x
        - 2 * gaussianPDFReal m v₁ x + gaussianPDFReal m v₂ x) ∂volume
      = (∫ x, (gaussianPDFReal m v₁ x ^ 2 / gaussianPDFReal m v₂ x
          - 2 * gaussianPDFReal m v₁ x) ∂volume) + ∫ x, gaussianPDFReal m v₂ x ∂volume :=
    integral_add (hIa.sub hIb) hIc
  have h2 : ∫ x, (gaussianPDFReal m v₁ x ^ 2 / gaussianPDFReal m v₂ x
        - 2 * gaussianPDFReal m v₁ x) ∂volume
      = (∫ x, gaussianPDFReal m v₁ x ^ 2 / gaussianPDFReal m v₂ x ∂volume)
        - ∫ x, 2 * gaussianPDFReal m v₁ x ∂volume :=
    integral_sub hIa hIb
  rw [h1, h2, integral_const_mul,
    integral_gaussianPDFReal_sq_div_var m hv₁ hv₂ hlt,
    integral_gaussianPDFReal_eq_one m hv₁, integral_gaussianPDFReal_eq_one m hv₂]
  ring

/-- **(B) Finiteness of the χ² integrand** (the side-condition feeding the Le Cam bound). -/
theorem integrable_chi2_gaussianReal_var (m : ℝ) {v₁ v₂ : ℝ≥0} (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0)
    (hlt : (v₁ : ℝ) < 2 * v₂) :
    Integrable (fun x => (((gaussianReal m v₁).rnDeriv (gaussianReal m v₂) x).toReal - 1) ^ 2)
      (gaussianReal m v₂) := by
  have hG : Integrable
      (fun x => (gaussianPDFReal m v₁ x / gaussianPDFReal m v₂ x - 1) ^ 2) (gaussianReal m v₂) := by
    rw [gaussianReal_of_var_ne_zero m hv₂,
      integrable_withDensity_iff_integrable_smul' (measurable_gaussianPDF m v₂)
        (ae_of_all _ fun _ => gaussianPDF_lt_top)]
    simp only [smul_eq_mul, toReal_gaussianPDF]
    exact (((integrable_gaussianPDFReal_sq_div_var m hv₁ hv₂ hlt).sub
        ((integrable_gaussianPDFReal m v₁).const_mul 2)).add
        (integrable_gaussianPDFReal m v₂)).congr
      (ae_of_all _ fun x => (gaussianPDFReal_density_expand_var m hv₂ x).symm)
  refine hG.congr ?_
  filter_upwards [rnDeriv_toReal_ae_var m hv₁ hv₂] with x hx
  rw [hx]

namespace NearNull

/-- **(C) Quantitative near-null undetectability (spread misreport).** Instantiating the Le Cam
two-point χ² bound at the two same-mean Gaussians: for every test `A`,
`1 - √(v₂/√(v₁ (2 v₂ - v₁)) - 1) ≤ P₁.real A + P₂.real Aᶜ`. As `v₁ → v₂` the χ² `→ 0`, so the left
side `→ 1` and no test distinguishes two Gaussians whose spreads differ by a near-null amount. -/
theorem gaussianReal_var_near_null_undetectable (m : ℝ) {v₁ v₂ : ℝ≥0} (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0)
    (hlt : (v₁ : ℝ) < 2 * v₂) {A : Set ℝ} (hA : MeasurableSet A) :
    1 - Real.sqrt ((v₂ : ℝ) / Real.sqrt ((v₁ : ℝ) * (2 * v₂ - v₁)) - 1)
      ≤ (gaussianReal m v₁).real A + (gaussianReal m v₂).real Aᶜ := by
  have hP1v : gaussianReal m v₁ ≪ volume := gaussianReal_absolutelyContinuous m hv₁
  have hvP2 : volume ≪ gaussianReal m v₂ := gaussianReal_absolutelyContinuous' m hv₂
  have hP1P2 : gaussianReal m v₁ ≪ gaussianReal m v₂ := hP1v.trans hvP2
  have h := NearNull.lecam_two_point_chi2 hP1P2 (integrable_chi2_gaussianReal_var m hv₁ hv₂ hlt) hA
  rwa [chi2_gaussianReal_var m hv₁ hv₂ hlt] at h

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms chi2_gaussianReal_var
#print axioms integrable_chi2_gaussianReal_var
#print axioms NearNull.gaussianReal_var_near_null_undetectable
