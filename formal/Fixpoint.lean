import PriorLaundering
import NonIdentifiability
import Mathlib.MeasureTheory.Function.AEEqOfLIntegral
import Mathlib.MeasureTheory.Measure.Decomposition.RadonNikodym
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Prod

/-!
# Corollary `c:fix` of the paper: the fixed points of the EM step

This file machine-checks the tractable core of **Corollary `c:fix`** ("The assumptions the step
leaves unchanged", regularizer of full support): the EM step `L[reg] = curatedPrior` leaves a
regularizer unchanged iff its data marginal matches the truth's,
`q_reg = reg ⟺ reg(y) = truth(y)` (eq:fixedpoint).

Reusing Theorem 1's Part A (`PriorLaundering.pLaw_partA`, `curatedPrior = reg.withDensity
(emReweight)`), the mechanism is: `L[reg] = reg` iff the EM reweight is `reg`-a.e. `1`.

## Results

* `fixed_point_iff_reweight_ae_one` — `curatedPrior κ μ ν = μ ↔ emReweight κ μ ν =ᵐ[μ] 1`
  (fixed point ⟺ trivial reweight).
* `reweight_ae_one_of_data_marginal_eq` — if `reg`'s data marginal equals the truth's
  (`κ ∘ₘ ν = κ ∘ₘ μ`), the reweight is `reg`-a.e. `1` (each `∫ d(κx)/d(reg(y)) d(reg(y)) = (κx) univ
  = 1`).
* `data_marginal_eq_imp_fixed_point` — hence `κ ∘ₘ ν = κ ∘ₘ μ ⟹ curatedPrior κ μ ν = μ`: the
  matching-data-marginal direction of eq:fixedpoint (`⟸`; the paper notes full support is needed
  **only** for the converse).
* `chiSq_rigidity` — the χ²-rigidity lemma `∫ (dP/dQ)² dQ = 1 ⟹ P = Q` (`χ²(P‖Q) = 0 ⟹ P = Q`).
* `mixture_withDensity` — the mixture Radon–Nikodym identity (Tonelli): the density of a mixture is
  the mixture of densities.
* `fixed_point_imp_data_marginal_eq` — the **full-support converse** `q_reg = reg ⟹ reg(y) =
  truth(y)` (`⟹`), now fully machine-checked: under full support (`h_dom`, `truth ≪ reg`) a fixed
  point forces `χ²(truth‖reg) = 0` (via `mixture_withDensity` + Tonelli + `emReweight =ᵐ 1`), whence
  `chiSq_rigidity` gives matching data marginals.
* `resolved_marginal_eq_imp_fixed_point` — for a likelihood factoring through the resolved
  coordinate (reusing `c:nonident`), matching resolved marginals suffices: the fixed-point set
  contains every prior sharing the truth's resolved marginal, "the assumptions that differ from it
  only in the directions the measurements cannot see."

Both directions of eq:fixedpoint are thus machine-checked: `⟸` (`data_marginal_eq_imp_fixed_point`)
and the full-support `⟹` (`fixed_point_imp_data_marginal_eq`).

## Honesty

No `sorry`/`admit`.  `#print axioms` on each main theorem lists only `propext`, `Classical.choice`,
`Quot.sound` (no `sorryAx`).  Reconciliation audit at the end.
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace Fixpoint

/-! ## A Tonelli helper: a mixture of `withDensity`s is a single `withDensity` of the mixture -/

/-- **Mixture of `withDensity`s (Tonelli).** Averaging `Q.withDensity (f x)` over `x ∼ ξ` gives
`Q.withDensity` of the pointwise ξ-average of the densities:
`∫_x Q.withDensity(f x) dξ(x) = Q.withDensity (y ↦ ∫_x f x y dξ(x))`.  This is the mixture
Radon–Nikodym identity — the density of a mixture is the mixture of densities — proved by Tonelli
(`lintegral_lintegral_swap`). -/
lemma mixture_withDensity {Ω' 𝓧' : Type*} {_mΩ : MeasurableSpace Ω'} {_m𝓧 : MeasurableSpace 𝓧'}
    (ξ : Measure Ω') [SFinite ξ] (Q : Measure 𝓧') [SFinite Q]
    {f : Ω' → 𝓧' → ℝ≥0∞} (hf : Measurable (Function.uncurry f)) :
    ξ.bind (fun x => Q.withDensity (f x)) = Q.withDensity (fun y => ∫⁻ x, f x y ∂ξ) := by
  have hmeas : Measurable (fun x => Q.withDensity (f x)) := by
    refine Measure.measurable_of_measurable_coe _ (fun t ht => ?_)
    simp_rw [withDensity_apply _ ht]
    exact hf.lintegral_prod_right
  ext s hs
  rw [Measure.bind_apply hs hmeas.aemeasurable, withDensity_apply _ hs]
  simp_rw [withDensity_apply _ hs]
  exact lintegral_lintegral_swap hf.aemeasurable

/-! ## The χ²-rigidity (for `c:fix`'s full-support converse) -/

/-- **χ²-rigidity.** If `P ≪ Q` are probability measures and the χ²-integral
`∫ (dP/dQ)² dQ = 1` (i.e. `χ²(P‖Q) = 0`), then `P = Q`.  (Expand `∫ (dP/dQ − 1)² dQ =
∫ (dP/dQ)² − 2∫(dP/dQ) + ∫1 = 1 − 2 + 1 = 0`, so `dP/dQ =ᵐ 1`, whence `P = Q.withDensity 1 = Q`.) -/
theorem chiSq_rigidity {α : Type*} {_ : MeasurableSpace α} (P Q : Measure α)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q] (hPQ : P ≪ Q)
    (hint : Integrable (fun y => (P.rnDeriv Q y).toReal ^ 2) Q)
    (hchi : ∫ y, (P.rnDeriv Q y).toReal ^ 2 ∂Q = 1) : P = Q := by
  have hf_int : Integrable (fun y => (P.rnDeriv Q y).toReal) Q :=
    integrableOn_univ.mp (Measure.integrableOn_toReal_rnDeriv (ν := Q) (measure_ne_top P _))
  have hf1 : ∫ y, (P.rnDeriv Q y).toReal ∂Q = 1 := by
    rw [Measure.integral_toReal_rnDeriv hPQ, measureReal_def, measure_univ, ENNReal.toReal_one]
  have hI2 : Integrable (fun y => 2 * (P.rnDeriv Q y).toReal) Q := hf_int.const_mul 2
  have heq : (fun y => ((P.rnDeriv Q y).toReal - 1) ^ 2)
      = fun y => ((P.rnDeriv Q y).toReal ^ 2 - 2 * (P.rnDeriv Q y).toReal) + 1 := by
    funext y; ring
  have hfi : Integrable (fun y => ((P.rnDeriv Q y).toReal - 1) ^ 2) Q := by
    rw [heq]; exact (hint.sub hI2).add (integrable_const 1)
  have hexp : ∫ y, ((P.rnDeriv Q y).toReal - 1) ^ 2 ∂Q = 0 := by
    calc ∫ y, ((P.rnDeriv Q y).toReal - 1) ^ 2 ∂Q
        = ∫ y, ((P.rnDeriv Q y).toReal ^ 2 - 2 * (P.rnDeriv Q y).toReal) + 1 ∂Q :=
          integral_congr_ae (Filter.Eventually.of_forall fun y => by ring)
      _ = (∫ y, (P.rnDeriv Q y).toReal ^ 2 - 2 * (P.rnDeriv Q y).toReal ∂Q) + ∫ _y, (1 : ℝ) ∂Q :=
          integral_add (hint.sub hI2) (integrable_const 1)
      _ = ((∫ y, (P.rnDeriv Q y).toReal ^ 2 ∂Q) - ∫ y, 2 * (P.rnDeriv Q y).toReal ∂Q)
            + ∫ _y, (1 : ℝ) ∂Q := by rw [integral_sub hint hI2]
      _ = 0 := by
          rw [integral_const_mul, hchi, hf1, integral_const, measureReal_def, measure_univ,
            ENNReal.toReal_one]
          norm_num
  have hf_ae1 : (fun y => (P.rnDeriv Q y).toReal) =ᵐ[Q] 1 := by
    have hz := (integral_eq_zero_iff_of_nonneg (fun y => sq_nonneg _) hfi).mp hexp
    filter_upwards [hz] with y hy
    have : (P.rnDeriv Q y).toReal - 1 = 0 := by
      have := sq_eq_zero_iff.mp hy; simpa using this
    simp only [Pi.one_apply]; linarith
  have hrn : P.rnDeriv Q =ᵐ[Q] 1 := by
    filter_upwards [hf_ae1, Measure.rnDeriv_lt_top P Q] with y hy hylt
    have hy' : (P.rnDeriv Q y).toReal = 1 := by simpa using hy
    simp only [Pi.one_apply]
    rw [← ENNReal.ofReal_toReal hylt.ne, hy', ENNReal.ofReal_one]
  rw [← Measure.withDensity_rnDeriv_eq P Q hPQ, withDensity_congr_ae hrn, withDensity_one]

/-! ## The fixed-point characterization (abstract) -/

section Abstract

variable {Ω 𝓧 : Type*} [MeasurableSpace Ω] [MeasurableSpace 𝓧]
  [StandardBorelSpace Ω] [Nonempty Ω]
  [MeasurableSpace.CountableOrCountablyGenerated Ω 𝓧]
  (κ : Kernel Ω 𝓧) [IsMarkovKernel κ]
  (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]

omit [StandardBorelSpace Ω] [Nonempty Ω] [IsProbabilityMeasure μ] in
/-- Measurability of the EM reweight (for abstract parameter space, reproving Part A's internal
step). -/
private lemma measurable_emReweight : Measurable (PriorLaundering.emReweight κ μ ν) := by
  unfold PriorLaundering.emReweight
  exact (Kernel.measurable_rnDeriv κ (Kernel.const _ (κ ∘ₘ μ))).lintegral_kernel_prod_right'
    (κ := Kernel.const _ (κ ∘ₘ ν))

/-- **The fixed points of the EM step are the trivial reweights.**
`L[reg] = reg` iff the EM reweight is `reg`-a.e. `1`. -/
theorem fixed_point_iff_reweight_ae_one
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ) :
    PriorLaundering.curatedPrior κ μ ν = μ ↔ PriorLaundering.emReweight κ μ ν =ᵐ[μ] 1 := by
  rw [PriorLaundering.pLaw_partA κ μ ν h_dom h_ac]
  constructor
  · intro h
    have hh : μ.withDensity (PriorLaundering.emReweight κ μ ν) = μ.withDensity 1 := by
      rw [h, withDensity_one]
    exact (withDensity_eq_iff_of_sigmaFinite
      (measurable_emReweight κ μ ν).aemeasurable measurable_one.aemeasurable).mp hh
  · intro h
    rw [withDensity_congr_ae h, withDensity_one]

omit [StandardBorelSpace Ω] [Nonempty Ω] [IsProbabilityMeasure ν] in
/-- **Matching data marginals ⟹ trivial reweight.** If `reg`'s data marginal equals the truth's,
`emReweight` is `reg`-a.e. `1`: each `∫ d(κ x)/d(reg(y)) d(truth(y)) = ∫ d(κx)/d(reg(y)) d(reg(y))
= (κ x) univ = 1`. -/
theorem reweight_ae_one_of_data_marginal_eq
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_eq : κ ∘ₘ ν = κ ∘ₘ μ) :
    PriorLaundering.emReweight κ μ ν =ᵐ[μ] 1 := by
  filter_upwards [h_dom] with x hx
  simp only [Pi.one_apply]
  unfold PriorLaundering.emReweight
  rw [h_eq]
  have hrn := Kernel.rnDeriv_eq_rnDeriv_measure (κ := κ) (η := Kernel.const Ω (κ ∘ₘ μ)) (a := x)
  rw [Kernel.const_apply] at hrn
  rw [lintegral_congr_ae hrn, Measure.lintegral_rnDeriv hx, measure_univ]

/-- **`c:fix` (`⟸`): matching data marginals give a fixed point.**
`κ ∘ₘ ν = κ ∘ₘ μ ⟹ curatedPrior κ μ ν = μ`. -/
theorem data_marginal_eq_imp_fixed_point
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_eq : κ ∘ₘ ν = κ ∘ₘ μ) :
    PriorLaundering.curatedPrior κ μ ν = μ :=
  (fixed_point_iff_reweight_ae_one κ μ ν h_dom (by rw [h_eq])).mpr
    (reweight_ae_one_of_data_marginal_eq κ μ ν h_dom h_eq)

omit [StandardBorelSpace Ω] [Nonempty Ω] [MeasurableSpace.CountableOrCountablyGenerated Ω 𝓧] in
/-- **`c:fix` (`⟹`), the full-support converse's rigidity core: vanishing χ² ⟹ matching data
marginals.** If the χ²-divergence of the truth's data marginal from the regularizer's vanishes
(`∫ (d truth(y)/d reg(y))² d reg(y) = 1`) then the two data marginals coincide, `truth(y) = reg(y)`.
The fixed point `q_reg = reg` yields this χ² condition under full support, by integrating the
fixed-point equation against the truth (Tonelli) — the `∫ truth(y)²/reg(y) dy = 1` step (see the
reconciliation for the one measure-theoretic identity that step still needs). -/
theorem data_marginal_eq_of_chiSq_one
    (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ)
    (hint : Integrable (fun y => ((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y).toReal ^ 2) (κ ∘ₘ μ))
    (hchi : ∫ y, ((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y).toReal ^ 2 ∂(κ ∘ₘ μ) = 1) :
    κ ∘ₘ ν = κ ∘ₘ μ :=
  chiSq_rigidity (κ ∘ₘ ν) (κ ∘ₘ μ) h_ac hint hchi

/-- **`c:fix` (`⟹`): the full-support converse, fully machine-checked.** Under full support
(`h_dom`: each likelihood `κ x ≪ reg`'s data marginal; `hνμ`: `truth ≪ reg` on the parameters) a
fixed point `L[reg] = reg` forces the data marginals to coincide, `truth(y) = reg(y)`.

The proof closes the "fixed point ⟹ `χ²(truth‖reg) = 0`" step the earlier commit had flagged:
`L[reg] = reg` gives `emReweight =ᵐ[reg] 1`, hence `=ᵐ[truth] 1` (via `truth ≪ reg`).  The truth's
data-marginal density is the ν-mixture of the per-`x` likelihood densities
(`mixture_withDensity`: `d(κ ∘ₘ ν)/d(κ ∘ₘ μ)(y) =ᵐ ∫ d(κ x)/d(κ ∘ₘ μ)(y) dν(x)`).  Integrating
`emReweight` against `ν` and swapping the order (Tonelli, `lintegral_lintegral_swap`) yields exactly
`∫ (d(κ ∘ₘ ν)/d(κ ∘ₘ μ))² d(κ ∘ₘ μ) = 1`, and `chiSq_rigidity` concludes. -/
theorem fixed_point_imp_data_marginal_eq
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ) (hνμ : ν ≪ μ)
    (hfix : PriorLaundering.curatedPrior κ μ ν = μ) :
    κ ∘ₘ ν = κ ∘ₘ μ := by
  haveI : IsProbabilityMeasure (κ ∘ₘ μ) :=
    ⟨by rw [Measure.bind_apply MeasurableSet.univ κ.aemeasurable]; simp⟩
  haveI : IsProbabilityMeasure (κ ∘ₘ ν) :=
    ⟨by rw [Measure.bind_apply MeasurableSet.univ κ.aemeasurable]; simp⟩
  -- fixed point ⟹ trivial reweight, transported from `reg` to `truth`
  have hrwμ : PriorLaundering.emReweight κ μ ν =ᵐ[μ] 1 :=
    (fixed_point_iff_reweight_ae_one κ μ ν h_dom h_ac).mp hfix
  have hrwν : PriorLaundering.emReweight κ μ ν =ᵐ[ν] 1 := hνμ.ae_le hrwμ
  -- the (jointly measurable) kernel reweight density and its ν-mixture
  have hDmeas : Measurable (Function.uncurry (κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)))) :=
    Kernel.measurable_rnDeriv κ (Kernel.const Ω (κ ∘ₘ μ))
  have hHmeas : Measurable (fun y => ∫⁻ x, κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y ∂ν) :=
    hDmeas.lintegral_prod_left
  -- ν-a.e., `κ x ≪ reg(y)` and there `κ x = reg(y).withDensity (d(κx)/d reg(y))`
  have hκwd : (⇑κ) =ᵐ[ν]
      fun x => (κ ∘ₘ μ).withDensity (κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x) := by
    filter_upwards [hνμ.ae_le h_dom] with x hx
    have hrn := Kernel.rnDeriv_eq_rnDeriv_measure (κ := κ) (η := Kernel.const Ω (κ ∘ₘ μ)) (a := x)
    rw [Kernel.const_apply] at hrn
    rw [withDensity_congr_ae hrn, Measure.withDensity_rnDeriv_eq (κ x) (κ ∘ₘ μ) hx]
  -- the truth's data marginal is `reg`'s data marginal reweighted by the ν-mixture density
  have hPwd : κ ∘ₘ ν
      = (κ ∘ₘ μ).withDensity (fun y => ∫⁻ x, κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y ∂ν) := by
    rw [Measure.bind_congr_right hκwd, mixture_withDensity ν (κ ∘ₘ μ) hDmeas]
  have hrnH : (κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ)
      =ᵐ[κ ∘ₘ μ] fun y => ∫⁻ x, κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y ∂ν := by
    rw [hPwd]; exact Measure.rnDeriv_withDensity (κ ∘ₘ μ) hHmeas
  -- the χ² integral (in `ℝ≥0∞`) equals `1`
  have hchi_enn : ∫⁻ y, ((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y) ^ 2 ∂(κ ∘ₘ μ) = 1 := by
    calc ∫⁻ y, ((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y) ^ 2 ∂(κ ∘ₘ μ)
        = ∫⁻ y, (∫⁻ x, κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y ∂ν) ^ 2 ∂(κ ∘ₘ μ) :=
          lintegral_congr_ae (hrnH.mono fun y hy => by simp only [hy])
      _ = ∫⁻ y, (∫⁻ x, κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y ∂ν) ∂(κ ∘ₘ ν) := by
          rw [hPwd, lintegral_withDensity_eq_lintegral_mul (κ ∘ₘ μ) hHmeas hHmeas]
          refine lintegral_congr (fun y => ?_)
          simp only [Pi.mul_apply, pow_two]
      _ = ∫⁻ x, PriorLaundering.emReweight κ μ ν x ∂ν := by
          rw [← lintegral_lintegral_swap (μ := ν) (ν := κ ∘ₘ ν)
            (f := κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ))) hDmeas.aemeasurable]
          rfl
      _ = 1 := by
          have h1 : PriorLaundering.emReweight κ μ ν =ᵐ[ν] fun _ => (1 : ℝ≥0∞) := by
            filter_upwards [hrwν] with x hx; simpa using hx
          rw [lintegral_congr_ae h1, lintegral_one, measure_univ]
  -- transfer to the real χ² integral and conclude by rigidity
  have hf2meas : AEMeasurable (fun y => ((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y) ^ 2) (κ ∘ₘ μ) :=
    ((Measure.measurable_rnDeriv _ _).pow_const 2).aemeasurable
  have hInt' : Integrable (fun y => (((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y) ^ 2).toReal) (κ ∘ₘ μ) :=
    integrable_toReal_of_lintegral_ne_top hf2meas (by rw [hchi_enn]; exact ENNReal.one_ne_top)
  have hfun : (fun y => (((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y) ^ 2).toReal)
      = fun y => ((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y).toReal ^ 2 := by
    funext y; rw [ENNReal.toReal_pow]
  have hInt : Integrable (fun y => ((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y).toReal ^ 2) (κ ∘ₘ μ) := hfun ▸ hInt'
  have hchi_real : ∫ y, ((κ ∘ₘ ν).rnDeriv (κ ∘ₘ μ) y).toReal ^ 2 ∂(κ ∘ₘ μ) = 1 := by
    have h := integral_toReal hf2meas (ae_lt_top' hf2meas (by rw [hchi_enn]; exact ENNReal.one_ne_top))
    simp_rw [ENNReal.toReal_pow] at h
    rw [h, hchi_enn, ENNReal.toReal_one]
  exact chiSq_rigidity (κ ∘ₘ ν) (κ ∘ₘ μ) h_ac hInt hchi_real

end Abstract

/-! ## The linear-operator reading: resolved marginal alone (reusing `c:nonident`) -/

section Resolved

variable {R B 𝓧 : Type*} [MeasurableSpace R] [MeasurableSpace B] [MeasurableSpace 𝓧]
  [StandardBorelSpace (R × B)] [Nonempty (R × B)]
  [MeasurableSpace.CountableOrCountablyGenerated (R × B) 𝓧]

/-- **`c:fix` for a factoring likelihood — matching resolved marginals suffice.**
If the likelihood reaches `x` only through the resolved coordinate (`κ x = η x.1`, i.e.
`p(y∣x)=p(y∣x_R)`), then any prior whose resolved marginal equals the truth's is a fixed point.  So
the fixed-point set contains the truth and every assumption differing from it only on the blind
subspace — the non-identifiable directions of `c:nonident`. -/
theorem resolved_marginal_eq_imp_fixed_point
    (κ : Kernel (R × B) 𝓧) [IsMarkovKernel κ] (η : Kernel R 𝓧) (hκ : ∀ x, κ x = η x.1)
    (μ ν : Measure (R × B)) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ)
    (h : ν.map Prod.fst = μ.map Prod.fst) :
    PriorLaundering.curatedPrior κ μ ν = μ :=
  data_marginal_eq_imp_fixed_point κ μ ν h_dom
    (NonIdentifiability.data_marginal_eq_of_resolved_eq κ η hκ ν μ h)

end Resolved

/-!
## Reconciliation — what `c:fix`'s formalization needs vs. what the paper states

**(a)** already in the text; **(b)** load-bearing but not stated; **(c)** automatic in the setting.

* the EM step is `curatedPrior κ μ ν` and `curatedPrior = reg.withDensity (emReweight)` — **(a)**:
  Theorem 1's Part A (eq:emstep), reused directly.  The fixed-point condition `L[reg]=reg` is then
  `withDensity (emReweight) = reg`, i.e. `emReweight =ᵐ[reg] 1` — the trivial-reweight core.
* `h_dom`, `h_ac` — **(a)**: the two absolute-continuity conditions of Part A (`h_ac` is automatic
  when the data marginals match).
* `SigmaFinite μ` (for `withDensity_eq_iff_of_sigmaFinite`) — **(c)**: `reg` is a probability
  measure.
* **The converse `⟹` is now fully machine-checked** (`fixed_point_imp_data_marginal_eq`).  The
  paper's converse "integrate the fixed-point equation against the truth to get
  `∫ truth(y)²/reg(y) dy = 1` (`χ²(truth‖reg) = 0`), forcing `reg(y) = truth(y)`" is closed in two
  reusable pieces:
  (i) **χ²-rigidity** `∫ (dP/dQ)² dQ = 1 ⟹ P = Q` (`chiSq_rigidity`), taking the honest full-support
  hypothesis as `P ≪ Q` (`truth(y) ≪ reg(y)`) and the χ² finiteness as an `Integrable` hypothesis
  (both discharged inside the converse from `∫⁻ (dP/dQ)² dQ = 1 < ∞`);
  (ii) **fixed point ⟹ χ² = 1** — the Tonelli step, now proved: `L[reg]=reg` gives `emReweight =ᵐ
  [reg] 1`, hence `=ᵐ[truth] 1` (via `truth ≪ reg`, hypothesis `hνμ`); the truth's data-marginal
  density is the ν-mixture of the per-`x` likelihood densities (`mixture_withDensity`:
  `d(κ ∘ₘ ν)/d(κ ∘ₘ μ)(y) =ᵐ ∫ d(κ x)/d(κ ∘ₘ μ)(y) dν(x)`); integrating `emReweight` against `ν`
  and swapping (`lintegral_lintegral_swap`) yields `∫⁻ (dP/dQ)² d(κ ∘ₘ μ) = 1`, which transfers to
  the real χ² integral by `integral_toReal`/`ENNReal.toReal_pow`.
* the converse's hypotheses — **(a)/(b)**: `h_dom` and `truth ≪ reg` (as parameter-level `hνμ : ν ≪
  μ`) are exactly the paper's "full support"; the `Integrable`/finite-χ² premise of `chiSq_rigidity`
  is discharged, not assumed, in `fixed_point_imp_data_marginal_eq`.
* `resolved_marginal_eq_imp_fixed_point` reuses `c:nonident`'s resolved-only data marginal — **(a)**:
  the paper's "for a linear operator the full-support fixed points are the priors whose resolved
  marginal is the truth's" (the `⟸` half, exact).

No hidden facts: **both** directions of eq:fixedpoint — `⟸`, the trivial-reweight characterization,
the χ²-rigidity, and the full-support `⟹` — are machine-checked with no `sorry` and axioms
`{propext, Classical.choice, Quot.sound}` only.

### Proof-level audit (Lean proof vs. the appendix argument)

The paper's appendix proves eq:fixedpoint in one movement: integrate the fixed-point equation against
the truth to get `∫ truth(y)²/reg(y) dy = 1`, read this as `χ²(truth‖reg) = 0`, conclude
`reg(y)=truth(y)`.  The Lean proof takes a different, more granular route, which surfaces:

* **(a) an intermediate the appendix skips.** Lean factors `c:fix` through
  `L[reg]=reg ⟺ emReweight =ᵐ[reg] 1` (`fixed_point_iff_reweight_ae_one`), proved from Part A's
  `withDensity` form plus the injectivity of `withDensity` (`withDensity_eq_iff_of_sigmaFinite`).
  The appendix never isolates this "fixed point ⟺ trivial multiplicative update" step — it goes
  straight to the `χ²`.  *Proposed appendix edit:* state the trivial-reweight equivalence first
  (it is the definition of a Vardi/NPMLE fixed point), then the data-marginal characterization; the
  two are logically separate and only the second needs full support.
* **(b) a hypothesis the appendix leans on silently.** The `⟸` direction ("matching data marginals
  give a fixed point") is glossed in the paper as `g(x)=∫p(y∣x)dy=1`.  The Lean proof shows this
  identity is `∫ d(κ x)/d(reg(y)) d(reg(y)) = (κ x)(univ) = 1`, and the middle equality is
  `Measure.lintegral_rnDeriv`, which **requires `κ x ≪ reg(y)`** (each likelihood dominated by
  `reg`'s data marginal — the `h_dom` hypothesis, i.e. `reg` of full support on the data).  So even
  the "easy" direction needs full support in the guise of `h_dom`; the paper attributes full support
  only to the converse.  *Proposed appendix edit:* note that `g(x)=1` uses `κ x ≪ reg(y)` (density
  integrates back to unit mass), so full support underlies `⟸` as well, via `h_dom`.
* **(b) the Kernel-vs-measure RN-derivative.** Lean must pass from the *kernel* reweight density
  (`Kernel.rnDeriv`) to the *measure* RN-derivative `∂(κ x)/∂(reg(y))` (`rnDeriv_eq_rnDeriv_measure`)
  before the total-mass identity applies — a step the paper's `p(y∣x)/reg(y)` notation elides.
-/

end Fixpoint

/-! ### Axiom audit

The doc comment above claims these rest only on Lean's three standard axioms. These commands
are what make that checkable rather than asserted: each must print `propext, Classical.choice,
Quot.sound` and never `sorryAx`. -/

#print axioms Fixpoint.data_marginal_eq_imp_fixed_point
#print axioms Fixpoint.fixed_point_imp_data_marginal_eq
