import Mathlib
import ChiSquaredTV

/-!
# χ² is invariant under a bijective bimeasurable change of variables

The χ²-divergence `χ²(P‖Q) = ∫ (dP/dQ − 1)² dQ` (`NearNull.chi2`, `ChiSquaredTV.lean`) depends only on
the density `dP/dQ`, which transports covariantly under a measurable equivalence `e : α ≃ᵐ β`. Hence
`χ²(e₊P ‖ e₊Q) = χ²(P‖Q)`: the near-null closeness bound is coordinate-free — measuring it in the
resolved coordinates of the operator, or in any other bimeasurable parametrization, gives the same value.

## Result
* `NearNull.chi2_map_measurableEquiv` — `χ²(P.map e ‖ Q.map e) = χ²(P ‖ Q)`.

## Honesty
No `sorry`/`admit`; `#print axioms` lists only `propext`, `Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure

/-- **χ² is invariant under a measurable equivalence.** For a bijective bimeasurable change of
variables `e : α ≃ᵐ β`, `χ²(P.map e ‖ Q.map e) = χ²(P ‖ Q)`. The χ² depends only on the density
`dP/dQ`, which pushes forward to `(dP/dQ) ∘ e⁻¹`; the change-of-variables in the integral cancels the
Jacobian. -/
theorem NearNull.chi2_map_measurableEquiv {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (P Q : Measure α) [IsProbabilityMeasure P] [IsProbabilityMeasure Q] (hPQ : P ≪ Q)
    (e : α ≃ᵐ β) :
    NearNull.chi2 (P.map e) (Q.map e) = NearNull.chi2 P Q := by
  haveI hPe : IsProbabilityMeasure (P.map e) := isProbabilityMeasure_map e.measurable.aemeasurable
  haveI hQe : IsProbabilityMeasure (Q.map e) := isProbabilityMeasure_map e.measurable.aemeasurable
  have hr_meas : Measurable (P.rnDeriv Q) := Measure.measurable_rnDeriv P Q
  set r : α → ENNReal := P.rnDeriv Q with hr
  have hf_meas : Measurable (fun y => r (e.symm y)) := hr_meas.comp e.symm.measurable
  -- Push-forward / withDensity commutation: `(Q.withDensity r).map e = (Q.map e).withDensity (r∘e⁻¹)`.
  have hcomm : (Q.withDensity r).map e = (Q.map e).withDensity (fun y => r (e.symm y)) := by
    refine Measure.ext fun s hs => ?_
    rw [Measure.map_apply e.measurable hs, withDensity_apply _ (e.measurable hs),
        withDensity_apply _ hs, setLIntegral_map hs hf_meas e.measurable]
    simp only [MeasurableEquiv.symm_apply_apply]
  -- `Q.withDensity (dP/dQ) = P`, so `P.map e = (Q.map e).withDensity (r∘e⁻¹)`.
  have hwd : Q.withDensity r = P := by rw [hr]; exact Measure.withDensity_rnDeriv_eq P Q hPQ
  have hPmap : P.map e = (Q.map e).withDensity (fun y => r (e.symm y)) := by rw [← hwd, hcomm]
  -- Read off the transported density.
  have hrn : (P.map e).rnDeriv (Q.map e) =ᵐ[Q.map e] (fun y => r (e.symm y)) := by
    rw [hPmap]; exact Measure.rnDeriv_withDensity (Q.map e) hf_meas
  have hg : AEStronglyMeasurable (fun y : β => ((r (e.symm y)).toReal - 1) ^ 2) (Q.map e) :=
    ((hf_meas.ennreal_toReal.sub measurable_const).pow_const 2).aestronglyMeasurable
  calc NearNull.chi2 (P.map e) (Q.map e)
      = ∫ y, (((P.map e).rnDeriv (Q.map e) y).toReal - 1) ^ 2 ∂(Q.map e) := rfl
    _ = ∫ y, ((r (e.symm y)).toReal - 1) ^ 2 ∂(Q.map e) := by
        refine integral_congr_ae ?_
        filter_upwards [hrn] with y hy
        rw [hy]
    _ = ∫ x, ((r (e.symm (e x))).toReal - 1) ^ 2 ∂Q :=
        integral_map e.measurable.aemeasurable hg
    _ = ∫ x, ((r x).toReal - 1) ^ 2 ∂Q := by simp only [MeasurableEquiv.symm_apply_apply]
    _ = NearNull.chi2 P Q := by rw [hr]; rfl

-- #print axioms audit
#print axioms NearNull.chi2_map_measurableEquiv
