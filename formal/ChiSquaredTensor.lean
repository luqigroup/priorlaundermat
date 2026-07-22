import Mathlib
import ChiSquaredTV

/-!
# χ² tensorizes over independent products

The near-null closeness `ε = √(χ²)` (see `ChiSquaredTV.lean`) is supplied for a *single* coordinate by
a closed-form Gaussian integral. This file records the multiplicative tensorization identity that turns a
per-coordinate χ² into the χ² of the joint law across independent coordinates:
`χ²(P⊗P' ‖ Q⊗Q') = (χ²(P‖Q) + 1)(χ²(P'‖Q') + 1) − 1`.

The route is entirely measure-theoretic:
* the product density factorizes (`Measure.prod_withDensity` through `withDensity_rnDeriv_eq` /
  `Measure.rnDeriv_withDensity`), so `d(P⊗P')/d(Q⊗Q') (x,y) =ᵐ (dP/dQ)(x)·(dP'/dQ')(y)`;
* expanding `(ab−1)² = a²b² − 2ab + 1` and applying Fubini (`integral_prod_mul`) turns the joint integral
  into a product of one-dimensional integrals;
* each marginal `∫ dP/dQ dQ = 1` (`Measure.integral_toReal_rnDeriv`, `probReal_univ`), and
  `∫ (dP/dQ)² dQ = χ²(P‖Q) + 1`.

## Results
* `NearNull.chi2_prod` — the tensorization identity above.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms NearNull.chi2_prod` lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure

namespace NearNull

/-- `∫ (dP/dQ)² dQ = χ²(P‖Q) + 1`, from the expansion `(a−1)² = a² − 2a + 1` and `∫ dP/dQ dQ = 1`. -/
private lemma integral_rnDeriv_sq {Ω : Type*} [MeasurableSpace Ω] (P Q : Measure Ω)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q] (hPQ : P ≪ Q)
    (hInt : Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q) :
    ∫ x, ((P.rnDeriv Q x).toReal) ^ 2 ∂Q = NearNull.chi2 P Q + 1 := by
  have ha : Integrable (fun x => (P.rnDeriv Q x).toReal) Q := Measure.integrable_toReal_rnDeriv
  have hmean : ∫ x, (P.rnDeriv Q x).toReal ∂Q = 1 := by
    rw [integral_toReal_rnDeriv hPQ]; exact probReal_univ
  have hg : Integrable
      (fun x => (P.rnDeriv Q x).toReal ^ 2 - 2 * (P.rnDeriv Q x).toReal) Q :=
    hInt.sub (ha.const_mul 2)
  have hchi : NearNull.chi2 P Q = (∫ x, (P.rnDeriv Q x).toReal ^ 2 ∂Q) - 1 := by
    unfold NearNull.chi2
    calc ∫ x, ((P.rnDeriv Q x).toReal - 1) ^ 2 ∂Q
        = ∫ x, ((P.rnDeriv Q x).toReal ^ 2 - 2 * (P.rnDeriv Q x).toReal + 1) ∂Q :=
          integral_congr_ae (Filter.Eventually.of_forall fun x => by ring)
      _ = (∫ x, ((P.rnDeriv Q x).toReal ^ 2 - 2 * (P.rnDeriv Q x).toReal) ∂Q)
            + ∫ _, (1 : ℝ) ∂Q :=
          integral_add hg (integrable_const 1)
      _ = ((∫ x, (P.rnDeriv Q x).toReal ^ 2 ∂Q) - ∫ x, 2 * (P.rnDeriv Q x).toReal ∂Q)
            + ∫ _, (1 : ℝ) ∂Q := by
          rw [integral_sub hInt (ha.const_mul 2)]
      _ = (∫ x, (P.rnDeriv Q x).toReal ^ 2 ∂Q) - 1 := by
          rw [integral_const_mul, hmean]; simp; ring
  rw [hchi]; ring

/-- **χ² tensorizes across an independent product.**
`χ²(P⊗P' ‖ Q⊗Q') = (χ²(P‖Q) + 1)(χ²(P'‖Q') + 1) − 1`. The hypotheses `hInt`/`hInt'` say each
one-dimensional χ² is finite (`∫ (dP/dQ)² dQ < ∞`), which holds for the near-null Gaussians. -/
theorem chi2_prod {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (P Q : Measure α) (P' Q' : Measure β)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    [IsProbabilityMeasure P'] [IsProbabilityMeasure Q']
    (hPQ : P ≪ Q) (hPQ' : P' ≪ Q')
    (hInt : Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q)
    (hInt' : Integrable (fun y => ((P'.rnDeriv Q' y).toReal) ^ 2) Q') :
    NearNull.chi2 (P.prod P') (Q.prod Q')
      = (NearNull.chi2 P Q + 1) * (NearNull.chi2 P' Q' + 1) - 1 := by
  -- measurable densities
  have hr : Measurable (P.rnDeriv Q) := Measure.measurable_rnDeriv P Q
  have hr' : Measurable (P'.rnDeriv Q') := Measure.measurable_rnDeriv P' Q'
  -- (1) the product density factorizes
  have hPeq : Q.withDensity (P.rnDeriv Q) = P := withDensity_rnDeriv_eq P Q hPQ
  have hP'eq : Q'.withDensity (P'.rnDeriv Q') = P' := withDensity_rnDeriv_eq P' Q' hPQ'
  have hprod : P.prod P'
      = (Q.prod Q').withDensity (fun p => P.rnDeriv Q p.1 * P'.rnDeriv Q' p.2) := by
    conv_lhs => rw [← hPeq, ← hP'eq]
    exact prod_withDensity hr hr'
  have hrn : (P.prod P').rnDeriv (Q.prod Q')
      =ᵐ[Q.prod Q'] fun p => P.rnDeriv Q p.1 * P'.rnDeriv Q' p.2 := by
    rw [hprod]
    exact rnDeriv_withDensity (Q.prod Q') ((hr.comp measurable_fst).mul (hr'.comp measurable_snd))
  have hrnR : (fun p => ((P.prod P').rnDeriv (Q.prod Q') p).toReal)
      =ᵐ[Q.prod Q'] fun p => (P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal := by
    filter_upwards [hrn] with p hp
    rw [hp, ENNReal.toReal_mul]
  -- (2) rewrite the joint χ² integrand by the factorized density
  have hchi_prod : NearNull.chi2 (P.prod P') (Q.prod Q')
      = ∫ p, ((P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal - 1) ^ 2 ∂(Q.prod Q') := by
    unfold NearNull.chi2
    apply integral_congr_ae
    filter_upwards [hrnR] with p hp
    rw [hp]
  -- one-dimensional integrability
  have haQ : Integrable (fun x => (P.rnDeriv Q x).toReal) Q := Measure.integrable_toReal_rnDeriv
  have hbQ : Integrable (fun y => (P'.rnDeriv Q' y).toReal) Q' := Measure.integrable_toReal_rnDeriv
  -- product integrability (product of L¹ functions on a product measure)
  have hI1 : Integrable
      (fun p : α × β => (P.rnDeriv Q p.1).toReal ^ 2 * (P'.rnDeriv Q' p.2).toReal ^ 2)
      (Q.prod Q') := hInt.mul_prod hInt'
  have hI2 : Integrable
      (fun p : α × β => (P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal)
      (Q.prod Q') := haQ.mul_prod hbQ
  -- expand `(ab − 1)² = a²b² − 2ab + 1`
  have hexpand : ∀ p : α × β,
      ((P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal - 1) ^ 2
        = (P.rnDeriv Q p.1).toReal ^ 2 * (P'.rnDeriv Q' p.2).toReal ^ 2
          - 2 * ((P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal) + 1 := fun p => by ring
  have hstep : ∫ p, ((P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal - 1) ^ 2 ∂(Q.prod Q')
      = ∫ p, ((P.rnDeriv Q p.1).toReal ^ 2 * (P'.rnDeriv Q' p.2).toReal ^ 2
          - 2 * ((P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal) + 1) ∂(Q.prod Q') :=
    integral_congr_ae (Filter.Eventually.of_forall hexpand)
  -- term-mode Fubini splits (explicit expected types; not `rw`)
  have hsplit1 : ∫ p, ((P.rnDeriv Q p.1).toReal ^ 2 * (P'.rnDeriv Q' p.2).toReal ^ 2
          - 2 * ((P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal) + 1) ∂(Q.prod Q')
      = (∫ p, ((P.rnDeriv Q p.1).toReal ^ 2 * (P'.rnDeriv Q' p.2).toReal ^ 2
          - 2 * ((P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal)) ∂(Q.prod Q'))
        + ∫ _, (1 : ℝ) ∂(Q.prod Q') :=
    integral_add (hI1.sub (hI2.const_mul 2)) (integrable_const 1)
  have hsplit2 : ∫ p, ((P.rnDeriv Q p.1).toReal ^ 2 * (P'.rnDeriv Q' p.2).toReal ^ 2
          - 2 * ((P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal)) ∂(Q.prod Q')
      = (∫ p, (P.rnDeriv Q p.1).toReal ^ 2 * (P'.rnDeriv Q' p.2).toReal ^ 2 ∂(Q.prod Q'))
        - ∫ p, 2 * ((P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal) ∂(Q.prod Q') :=
    integral_sub hI1 (hI2.const_mul 2)
  -- Fubini factorizations
  have hprodI1 : ∫ p, (P.rnDeriv Q p.1).toReal ^ 2 * (P'.rnDeriv Q' p.2).toReal ^ 2 ∂(Q.prod Q')
      = (∫ x, (P.rnDeriv Q x).toReal ^ 2 ∂Q) * ∫ y, (P'.rnDeriv Q' y).toReal ^ 2 ∂Q' :=
    integral_prod_mul (fun x => (P.rnDeriv Q x).toReal ^ 2) (fun y => (P'.rnDeriv Q' y).toReal ^ 2)
  have hprodI2 : ∫ p, (P.rnDeriv Q p.1).toReal * (P'.rnDeriv Q' p.2).toReal ∂(Q.prod Q')
      = (∫ x, (P.rnDeriv Q x).toReal ∂Q) * ∫ y, (P'.rnDeriv Q' y).toReal ∂Q' :=
    integral_prod_mul (fun x => (P.rnDeriv Q x).toReal) (fun y => (P'.rnDeriv Q' y).toReal)
  -- one-dimensional values
  have hone : ∫ _p : α × β, (1 : ℝ) ∂(Q.prod Q') = 1 := by simp
  have hA2 : ∫ x, (P.rnDeriv Q x).toReal ^ 2 ∂Q = NearNull.chi2 P Q + 1 :=
    integral_rnDeriv_sq P Q hPQ hInt
  have hB2 : ∫ y, (P'.rnDeriv Q' y).toReal ^ 2 ∂Q' = NearNull.chi2 P' Q' + 1 :=
    integral_rnDeriv_sq P' Q' hPQ' hInt'
  have hA1 : ∫ x, (P.rnDeriv Q x).toReal ∂Q = 1 := by
    rw [integral_toReal_rnDeriv hPQ]; exact probReal_univ
  have hB1 : ∫ y, (P'.rnDeriv Q' y).toReal ∂Q' = 1 := by
    rw [integral_toReal_rnDeriv hPQ']; exact probReal_univ
  -- assemble
  rw [hchi_prod, hstep, hsplit1, hsplit2, hprodI1, integral_const_mul, hprodI2,
    hone, hA1, hB1, hA2, hB2]
  ring

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.chi2_prod
