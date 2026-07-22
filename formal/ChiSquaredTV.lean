import Mathlib
import NearNullLeCam

/-!
# χ² controls total variation: the closeness `ε` feeding the Le Cam engine

`NearNullLeCam.lean` reduces the near-null undetectability to a closeness input: if every event's
probability differs by at most `ε` between the two data laws, no test beats error `1 - ε`. This file
supplies that `ε` from the **χ²-divergence**, with no Pinsker (absent from mathlib): via Cauchy–Schwarz
(Jensen for a probability measure), every event satisfies `|P A − Q A| ≤ √(χ²(P‖Q))`, where
`χ²(P‖Q) = ∫ (dP/dQ − 1)² dQ`. Instantiated at the two near-null Gaussians (whose χ² is a closed-form
Gaussian integral, `GaussianPdfIntegral.lean`), this closes the near-null rate.

## Results
* `sq_integral_le_integral_sq` — Jensen for a probability measure: `(∫ f dQ)² ≤ ∫ f² dQ`.
* `chi2` — the χ²-divergence `∫ (dP/dQ − 1)² dQ`.
* `abs_measureReal_diff_le_sqrt_chi2` — `|P.real A − Q.real A| ≤ √(χ²(P‖Q))` for every measurable `A`.
* `lecam_two_point_chi2` — the Le Cam bound with `ε = √(χ²)`: every test has total error `≥ 1 − √(χ²)`.

## Honesty
No `sorry`/`admit`; `#print axioms` on each result lists only `propext`, `Classical.choice`,
`Quot.sound`.
-/

open MeasureTheory Set Real

namespace NearNull

variable {Ω : Type*} [MeasurableSpace Ω] {P Q : Measure Ω}
  [IsProbabilityMeasure P] [IsProbabilityMeasure Q]

/-- **Jensen for a probability measure.** `(∫ f dQ)² ≤ ∫ f² dQ`, from `0 ≤ ∫ (f − ∫f)² dQ`. -/
theorem sq_integral_le_integral_sq (f : Ω → ℝ) (hf : Integrable f Q)
    (hf2 : Integrable (fun x => f x ^ 2) Q) :
    (∫ x, f x ∂Q) ^ 2 ≤ ∫ x, f x ^ 2 ∂Q := by
  set c := ∫ x, f x ∂Q with hc
  have hcf : Integrable (fun x => 2 * c * f x) Q := hf.const_mul (2 * c)
  have hg : Integrable (fun x => f x ^ 2 - 2 * c * f x) Q := hf2.sub hcf
  have hadd : ∫ x, (f x ^ 2 - 2 * c * f x + c ^ 2) ∂Q
      = (∫ x, (f x ^ 2 - 2 * c * f x) ∂Q) + ∫ _, (c ^ 2 : ℝ) ∂Q :=
    integral_add hg (integrable_const _)
  have hsub : ∫ x, (f x ^ 2 - 2 * c * f x) ∂Q = (∫ x, f x ^ 2 ∂Q) - ∫ x, 2 * c * f x ∂Q :=
    integral_sub hf2 hcf
  have hconst : ∫ _, (c ^ 2 : ℝ) ∂Q = c ^ 2 := by
    rw [integral_const]; simp
  have key : ∫ x, (f x - c) ^ 2 ∂Q = (∫ x, f x ^ 2 ∂Q) - c ^ 2 := by
    calc ∫ x, (f x - c) ^ 2 ∂Q
        = ∫ x, (f x ^ 2 - 2 * c * f x + c ^ 2) ∂Q :=
          integral_congr_ae (Filter.Eventually.of_forall fun x => by ring)
      _ = (∫ x, f x ^ 2 ∂Q) - ∫ x, 2 * c * f x ∂Q + c ^ 2 := by rw [hadd, hsub, hconst]
      _ = (∫ x, f x ^ 2 ∂Q) - 2 * c * (∫ x, f x ∂Q) + c ^ 2 := by rw [integral_const_mul]
      _ = (∫ x, f x ^ 2 ∂Q) - c ^ 2 := by rw [← hc]; ring
  have hnn : 0 ≤ ∫ x, (f x - c) ^ 2 ∂Q := integral_nonneg fun x => sq_nonneg _
  rw [key] at hnn; linarith

/-- The **χ²-divergence** of `P` from `Q` via the density `dP/dQ`: `∫ (dP/dQ − 1)² dQ`. -/
noncomputable def chi2 (P Q : Measure Ω) : ℝ := ∫ x, ((P.rnDeriv Q x).toReal - 1) ^ 2 ∂Q

/-- **χ² controls the event difference.** For every measurable `A`,
`|P.real A − Q.real A| ≤ √(χ²(P‖Q))`. The integrability hypothesis (`(dP/dQ − 1)² ∈ L¹(Q)`, i.e. χ²
finite) holds for the near-null Gaussians. -/
theorem abs_measureReal_diff_le_sqrt_chi2 (hPQ : P ≪ Q)
    (hr2 : Integrable (fun x => ((P.rnDeriv Q x).toReal - 1) ^ 2) Q)
    {A : Set Ω} (hA : MeasurableSet A) :
    |P.real A - Q.real A| ≤ Real.sqrt (chi2 P Q) := by
  set r := fun x => (P.rnDeriv Q x).toReal with hrdef
  have hrint : Integrable r Q := Measure.integrable_toReal_rnDeriv
  have hr1 : Integrable (fun x => r x - 1) Q := hrint.sub (integrable_const 1)
  -- P.real A − Q.real A = ∫_A (r − 1) dQ
  have hPA : P.real A = ∫ x in A, r x ∂Q := (Measure.setIntegral_toReal_rnDeriv hPQ A).symm
  have hQA : Q.real A = ∫ _ in A, (1 : ℝ) ∂Q := by
    rw [setIntegral_const, smul_eq_mul, mul_one, measureReal_def]
  have hdiff : P.real A - Q.real A = ∫ x in A, (r x - 1) ∂Q := by
    rw [hPA, hQA, ← integral_sub hrint.restrict ((integrable_const (1 : ℝ)).restrict)]
  -- |∫_A (r−1)| ≤ ∫_A |r−1| ≤ ∫ |r−1| dQ
  have h1 : |P.real A - Q.real A| ≤ ∫ x in A, |r x - 1| ∂Q := by
    rw [hdiff]; exact abs_integral_le_integral_abs
  have h2 : ∫ x in A, |r x - 1| ∂Q ≤ ∫ x, |r x - 1| ∂Q :=
    setIntegral_le_integral hr1.abs (Filter.Eventually.of_forall fun _ => abs_nonneg _)
  -- (∫ |r−1|)² ≤ ∫ (r−1)² = χ²
  have hJ : (∫ x, |r x - 1| ∂Q) ^ 2 ≤ chi2 P Q := by
    have := sq_integral_le_integral_sq (fun x => |r x - 1|) hr1.abs (by simpa [sq_abs] using hr2)
    simpa [chi2, sq_abs, hrdef] using this
  have hnn : 0 ≤ ∫ x, |r x - 1| ∂Q := integral_nonneg fun _ => abs_nonneg _
  have h3 : ∫ x, |r x - 1| ∂Q ≤ Real.sqrt (chi2 P Q) :=
    calc ∫ x, |r x - 1| ∂Q = Real.sqrt ((∫ x, |r x - 1| ∂Q) ^ 2) := (Real.sqrt_sq hnn).symm
      _ ≤ Real.sqrt (chi2 P Q) := Real.sqrt_le_sqrt hJ
  linarith [h1, h2, h3]

/-- **Le Cam two-point bound with `ε = √(χ²)`.** Every test between `P` and `Q` has total error at
least `1 − √(χ²(P‖Q))`. Combined with a χ² bound this is the near-null undetectability rate. -/
theorem lecam_two_point_chi2 (hPQ : P ≪ Q)
    (hr2 : Integrable (fun x => ((P.rnDeriv Q x).toReal - 1) ^ 2) Q)
    {A : Set Ω} (hA : MeasurableSet A) :
    1 - Real.sqrt (chi2 P Q) ≤ P.real A + Q.real Aᶜ :=
  lecam_two_point (fun B hB => abs_measureReal_diff_le_sqrt_chi2 hPQ hr2 hB) hA

end NearNull

-- #print axioms audit
#print axioms NearNull.sq_integral_le_integral_sq
#print axioms NearNull.abs_measureReal_diff_le_sqrt_chi2
#print axioms NearNull.lecam_two_point_chi2
