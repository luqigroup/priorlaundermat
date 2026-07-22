import Mathlib
import ChiSquaredTV
import ChiSquaredInvariance
import ChiSquaredTensor

/-!
# χ² tensorizes over a finite product of independent coordinates

The two-factor tensorization identity `χ²(P⊗P' ‖ Q⊗Q') = (χ²(P‖Q)+1)(χ²(P'‖Q')+1) − 1`
(`NearNull.chi2_prod`, `ChiSquaredTensor.lean`) inducts to the `d`-fold product over a finite index
`Fin n`:
`χ²(⨂ᵢ Pᵢ ‖ ⨂ᵢ Qᵢ) = (∏ᵢ (χ²(Pᵢ‖Qᵢ)+1)) − 1`.
This is the many-coordinate near-null closeness budget: independent coordinates multiply the
per-coordinate `1+χ²` factors, so the joint χ² of a full acquisition is fixed by the marginals.

The route is induction on `n`, splitting off coordinate `0` with the measure-preserving equivalence
`MeasurableEquiv.piFinSuccAbove` and applying `chi2_prod` to head × tail. Two auxiliary facts are
carried through the induction: absolute continuity of the product laws (`pi_ac`) and the L²
integrability of the joint density (needed to feed `chi2_prod` at the next step), so the induction
proves the identity together with `∫ (d(⨂P)/d(⨂Q))² d(⨂Q) < ∞`.

## Results
* `NearNull.chi2_pi` — the `d`-fold tensorization identity above.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms NearNull.chi2_pi` lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure Finset

universe u

namespace NearNull

/-- `χ²(μ‖μ) = 0`: the density `dμ/dμ =ᵐ 1`, so the integrand `(1−1)² = 0`. -/
private lemma chi2_self {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [SigmaFinite μ] :
    NearNull.chi2 μ μ = 0 := by
  unfold NearNull.chi2
  rw [integral_congr_ae (g := fun _ => (0 : ℝ))]
  · simp
  · filter_upwards [Measure.rnDeriv_self μ] with x hx
    rw [hx]; norm_num

/-- The squared joint density is `Q⊗Q'`-integrable, from the factorization
`d(P⊗P')/d(Q⊗Q') (x,y) =ᵐ (dP/dQ)(x)·(dP'/dQ')(y)` and `hInt.mul_prod hInt'`. -/
private lemma integrable_rnDeriv_sq_prod {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (P Q : Measure α) (P' Q' : Measure β)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    [IsProbabilityMeasure P'] [IsProbabilityMeasure Q']
    (hPQ : P ≪ Q) (hPQ' : P' ≪ Q')
    (hInt : Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q)
    (hInt' : Integrable (fun y => ((P'.rnDeriv Q' y).toReal) ^ 2) Q') :
    Integrable (fun p => (((P.prod P').rnDeriv (Q.prod Q') p).toReal) ^ 2) (Q.prod Q') := by
  have hr : Measurable (P.rnDeriv Q) := Measure.measurable_rnDeriv P Q
  have hr' : Measurable (P'.rnDeriv Q') := Measure.measurable_rnDeriv P' Q'
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
  have hI1 : Integrable
      (fun p : α × β => (P.rnDeriv Q p.1).toReal ^ 2 * (P'.rnDeriv Q' p.2).toReal ^ 2)
      (Q.prod Q') := hInt.mul_prod hInt'
  apply hI1.congr
  filter_upwards [hrn] with p hp
  rw [hp, ENNReal.toReal_mul, mul_pow]

/-- Transport of the squared-density integrability across a measurable equivalence `e`: the density
pushes forward to `(dP/dQ)∘e⁻¹`, and integrability is invariant under the bimeasurable `e`. -/
private lemma integrable_rnDeriv_sq_map {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (P Q : Measure α) [IsProbabilityMeasure P] [IsProbabilityMeasure Q] (hPQ : P ≪ Q)
    (e : α ≃ᵐ β)
    (hmap : Integrable (fun y => (((P.map e).rnDeriv (Q.map e) y).toReal) ^ 2) (Q.map e)) :
    Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q := by
  haveI hPe : IsProbabilityMeasure (P.map e) := isProbabilityMeasure_map e.measurable.aemeasurable
  haveI hQe : IsProbabilityMeasure (Q.map e) := isProbabilityMeasure_map e.measurable.aemeasurable
  have hr_meas : Measurable (P.rnDeriv Q) := Measure.measurable_rnDeriv P Q
  set r : α → ENNReal := P.rnDeriv Q with hr
  have hf_meas : Measurable (fun y => r (e.symm y)) := hr_meas.comp e.symm.measurable
  have hcomm : (Q.withDensity r).map e = (Q.map e).withDensity (fun y => r (e.symm y)) := by
    refine Measure.ext fun s hs => ?_
    rw [Measure.map_apply e.measurable hs, withDensity_apply _ (e.measurable hs),
        withDensity_apply _ hs, setLIntegral_map hs hf_meas e.measurable]
    simp only [MeasurableEquiv.symm_apply_apply]
  have hwd : Q.withDensity r = P := by rw [hr]; exact Measure.withDensity_rnDeriv_eq P Q hPQ
  have hPmap : P.map e = (Q.map e).withDensity (fun y => r (e.symm y)) := by rw [← hwd, hcomm]
  have hrn : (P.map e).rnDeriv (Q.map e) =ᵐ[Q.map e] (fun y => r (e.symm y)) := by
    rw [hPmap]; exact Measure.rnDeriv_withDensity (Q.map e) hf_meas
  have hmap2 : Integrable (fun y => ((r (e.symm y)).toReal) ^ 2) (Q.map e) := by
    apply hmap.congr
    filter_upwards [hrn] with y hy
    rw [hy]
  have hemb : MeasurableEmbedding e := e.measurableEmbedding
  rw [hemb.integrable_map_iff] at hmap2
  have heq : ((fun y => ((r (e.symm y)).toReal) ^ 2) ∘ e) = fun x => ((r x).toReal) ^ 2 := by
    funext x; simp only [Function.comp_apply, MeasurableEquiv.symm_apply_apply]
  rwa [heq] at hmap2

/-- The product laws are absolutely continuous coordinatewise. Induction on `n`: the empty product is a
`dirac`, and the successor step splits off coordinate `0` and pushes `AbsolutelyContinuous.prod`
back through the measure-preserving equivalence. -/
theorem pi_ac : ∀ (n : ℕ) (X : Fin n → Type u) [∀ i, MeasurableSpace (X i)]
    (P Q : ∀ i, Measure (X i)) [∀ i, IsProbabilityMeasure (P i)] [∀ i, IsProbabilityMeasure (Q i)]
    (_hPQ : ∀ i, P i ≪ Q i),
    Measure.pi P ≪ Measure.pi Q := by
  intro n
  induction n with
  | zero =>
    intro X _ P Q _ _ _hPQ
    rw [Measure.pi_of_empty P, Measure.pi_of_empty Q]
  | succ n ih =>
    intro X _ P Q _ _ hPQ
    set e := MeasurableEquiv.piFinSuccAbove X 0 with he
    have he_P := measurePreserving_piFinSuccAbove P 0
    have he_Q := measurePreserving_piFinSuccAbove Q 0
    have hstep : (Measure.pi P).map e ≪ (Measure.pi Q).map e := by
      rw [he_P.map_eq, he_Q.map_eq]
      exact (hPQ 0).prod (ih _ _ _ (fun j => hPQ _))
    have hback := hstep.map e.symm.measurable
    rwa [e.map_symm_map, e.map_symm_map] at hback

/-- Induction workhorse: the `d`-fold χ² identity **together with** the L² integrability of the joint
density (the latter is exactly what `chi2_prod` needs from the tail at the next step). -/
private theorem chi2_pi_aux : ∀ (n : ℕ) (X : Fin n → Type u) [∀ i, MeasurableSpace (X i)]
    (P Q : ∀ i, Measure (X i)) [∀ i, IsProbabilityMeasure (P i)] [∀ i, IsProbabilityMeasure (Q i)]
    (_hPQ : ∀ i, P i ≪ Q i)
    (_hInt : ∀ i, Integrable (fun x => (((P i).rnDeriv (Q i) x).toReal) ^ 2) (Q i)),
    NearNull.chi2 (Measure.pi P) (Measure.pi Q)
        = (∏ i, (NearNull.chi2 (P i) (Q i) + 1)) - 1
      ∧ Integrable
        (fun x => ((Measure.pi P).rnDeriv (Measure.pi Q) x).toReal ^ 2) (Measure.pi Q) := by
  intro n
  induction n with
  | zero =>
    intro X _ P Q _ _ _hPQ _hInt
    have hpieq : Measure.pi P = Measure.pi Q := by
      rw [Measure.pi_of_empty P, Measure.pi_of_empty Q]
    refine ⟨?_, ?_⟩
    · rw [hpieq, chi2_self]
      simp
    · rw [hpieq]
      apply (integrable_const (1 : ℝ)).congr
      filter_upwards [Measure.rnDeriv_self (Measure.pi Q)] with x hx
      rw [hx]; norm_num
  | succ n ih =>
    intro X _ P Q _ _ hPQ hInt
    set e := MeasurableEquiv.piFinSuccAbove X 0 with he
    have he_P := measurePreserving_piFinSuccAbove P 0
    have he_Q := measurePreserving_piFinSuccAbove Q 0
    have hac_pi : Measure.pi P ≪ Measure.pi Q := pi_ac _ X P Q hPQ
    have hac_tail : Measure.pi (fun j => P (Fin.succAbove 0 j))
        ≪ Measure.pi (fun j => Q (Fin.succAbove 0 j)) :=
      pi_ac _ _ _ _ (fun j => hPQ _)
    have ihTail := ih (fun j => X (Fin.succAbove 0 j))
      (fun j => P (Fin.succAbove 0 j)) (fun j => Q (Fin.succAbove 0 j))
      (fun j => hPQ (Fin.succAbove 0 j)) (fun j => hInt (Fin.succAbove 0 j))
    refine ⟨?_, ?_⟩
    · -- the χ² identity
      have key : NearNull.chi2 (Measure.pi P) (Measure.pi Q)
          = (NearNull.chi2 (P 0) (Q 0) + 1)
            * (NearNull.chi2 (Measure.pi (fun j => P (Fin.succAbove 0 j)))
                (Measure.pi (fun j => Q (Fin.succAbove 0 j))) + 1) - 1 := by
        rw [← chi2_map_measurableEquiv (Measure.pi P) (Measure.pi Q) hac_pi e,
            he_P.map_eq, he_Q.map_eq]
        exact chi2_prod (P 0) (Q 0) _ _ (hPQ 0) hac_tail (hInt 0) ihTail.2
      rw [key, ihTail.1, Fin.prod_univ_succAbove (fun i => NearNull.chi2 (P i) (Q i) + 1) 0]
      ring
    · -- the L² integrability
      apply integrable_rnDeriv_sq_map (Measure.pi P) (Measure.pi Q) hac_pi e
      rw [he_P.map_eq, he_Q.map_eq]
      exact integrable_rnDeriv_sq_prod (P 0) (Q 0) _ _ (hPQ 0) hac_tail (hInt 0) ihTail.2

/-- **χ² tensorizes across a finite independent product.**
`χ²(⨂ᵢ Pᵢ ‖ ⨂ᵢ Qᵢ) = (∏ᵢ (χ²(Pᵢ‖Qᵢ)+1)) − 1` over `i : Fin n`. Each hypothesis says the coordinate
`Pᵢ ≪ Qᵢ` and its one-dimensional χ² is finite (`∫ (dPᵢ/dQᵢ)² dQᵢ < ∞`), which holds for the
near-null Gaussians. -/
theorem chi2_pi {n : ℕ} {X : Fin n → Type*} [∀ i, MeasurableSpace (X i)]
    (P Q : ∀ i, Measure (X i)) [∀ i, IsProbabilityMeasure (P i)] [∀ i, IsProbabilityMeasure (Q i)]
    (hPQ : ∀ i, P i ≪ Q i)
    (hInt : ∀ i, Integrable (fun x => (((P i).rnDeriv (Q i) x).toReal) ^ 2) (Q i)) :
    NearNull.chi2 (Measure.pi P) (Measure.pi Q)
      = (∏ i, (NearNull.chi2 (P i) (Q i) + 1)) - 1 :=
  (chi2_pi_aux n X P Q hPQ hInt).1

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.chi2_pi
