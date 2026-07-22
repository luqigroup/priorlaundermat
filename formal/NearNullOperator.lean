import Mathlib
import ChiSquaredTV
import ChiSquaredInvariance
import ChiSquaredPi
import GaussianChiSquared
import MultivariateGaussianChiSquared
import MultivariateGaussianWhitening
import MultivariateGaussianScale
import MultivariateGaussianSpread

/-!
# Operator instantiation of `p:nearnull`(ii) and the `n`-survey crossover

The χ²/Le Cam cluster proves the mean-misreport χ² at any common covariance
(`MultivariateGaussianWhitening.lean`), the spread-misreport χ² at the **standard** Gaussian base
(`MultivariateGaussianSpread.lean`), and tensorization over independent coordinates
(`ChiSquaredPi.lean`) — but no statement mentions the forward operator or `n` surveys. This file
closes that gap in three steps, following `p:nearnull`(ii) of `paper/manuscript.tex`.

**(3a) General-covariance spread χ².** The same affine whitening `y ↦ L⁻¹(y − μ)` that
`MultivariateGaussianWhitening.lean` used for the mean case conjugates
`(N(μ, LLᵀ), N(μ, LLᵀ + (Lu)(Lu)ᵀ))` to `(N₀, N₀∘M)` with `M Mᵀ = I + u uᵀ`, so χ² invariance
lands on the rank-one statement: `χ²(N(μ,S) ‖ N(μ, S + (Lu)(Lu)ᵀ)) = (1+‖u‖²)/√(1+2‖u‖²) − 1`.

**(3b) Operator instantiation.** With data law `N(Aμ, S_y)`, `S_y = Ly Lyᵀ` for an invertible square
root `Ly`, a mean misreport `μ ↦ μ + w v` moves the data marginal to `N(Aμ + w·Av, S_y)` with
`χ² = exp(β) − 1`, `β = w²‖Ly⁻¹(Av)‖²`; a spread misreport `Σ⋆ ↦ Σ⋆ + w v vᵀ` moves it to
`N(Aμ, S_y + w (Av)(Av)ᵀ)` with `χ² = (1+β)/√(1+2β) − 1`, `β = w‖Ly⁻¹(Av)‖²`. On paper
`‖Ly⁻¹(Av)‖² = (Av)ᵀ S_y⁻¹ (Av)`, so these are the manuscript's closed forms. **What is taken as
given rather than machine-checked:** the theorems are stated directly on the Gaussian pushforward
laws `N(Aμ, Ly Lyᵀ)` etc.; the derivation that the record `y = A x + ε` of a truth `x ∼ N(μ, Σ⋆)`
under independent noise `ε ∼ N(0, Γ)` has exactly this marginal with `Ly Lyᵀ = A Σ⋆ Aᵀ + Γ` is the
paper's (standard) computation, not reproved here — the covariance enters only through its square
root `Ly`. The rank-one square root `M` (with `M Mᵀ = I + u uᵀ`) is *constructed*
(`exists_rank1_sqrt`), so the spread hypotheses are satisfiable for every direction.

**(3c) `n`-survey crossover.** For `n` iid surveys the tensorization gives
`χ²ₙ = (1+χ²)ⁿ − 1`, so by the Le Cam two-point χ² bound every test's summed type-I + type-II error
is at least `1 − √((1+χ²)ⁿ − 1)`; whenever `(1+χ²)ⁿ < 5/4` the summed error exceeds `1/2` — the
machine form of the paper's crossover `n < n⋆`. At `Av = 0` the two data marginals coincide, χ² is
`0`, and every test's summed error is at least `1` for every `n` — the exact non-identifiability of
`c:nonident` recovered as the endpoint.

## Results
* `NearNull.chi2_nonneg` — `0 ≤ χ²(P‖Q)`.
* `NearNull.exists_rank1_sqrt` — a self-adjoint `M` with `M Mᵀ = I + u uᵀ` exists for every `u`.
* `NearNull.chi2_gaussian_spread_rank1` — **(3a)**: `χ²(N(μ,LLᵀ) ‖ N(μ,LLᵀ)∘bump) = (1+‖u‖²)/√(1+2‖u‖²) − 1`.
* `NearNull.gaussian_spread_rank1_ac` / `_integrable` / `_undetectable` — its side conditions and Le Cam payoff.
* `NearNull.gaussian_mean_shift_ac` / `_integrable` — the mean case's side conditions at general covariance.
* `NearNull.operator_mean_misreport_chi2` / `_undetectable` — **(3b)** mean: `χ² = exp(w²‖Ly⁻¹(Av)‖²) − 1`.
* `NearNull.operator_spread_misreport_chi2` / `_undetectable` — **(3b)** spread: `χ² = (1+β)/√(1+2β) − 1`, `β = w‖Ly⁻¹(Av)‖²`.
* `NearNull.operator_mean_endpoint` / `operator_spread_endpoint` — at `Av = 0` the marginals coincide: summed error `≥ 1`.
* `NearNull.operator_mean_endpoint_chi2` / `operator_spread_endpoint_chi2` — at `Av = 0` the χ² is `0`.
* `NearNull.chi2_iid` — **(3c)**: `χ²(Pⁿ ‖ Qⁿ) = (1+χ²)ⁿ − 1`.
* `NearNull.nsurvey_lecam` — every `n`-survey test's summed error `≥ 1 − √((1+χ²)ⁿ − 1)`.
* `NearNull.nsurvey_crossover` — if `(1+χ²)ⁿ < 5/4`, every `n`-survey test's summed error `> 1/2`.
* `NearNull.operator_mean_nsurvey_undetectable` / `operator_spread_nsurvey_undetectable` — the bound at the operator's closed forms.
* `NearNull.operator_mean_crossover` / `operator_spread_crossover` — the operator-level crossover.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each public result lists only `propext`,
`Classical.choice`, `Quot.sound`. The identification of the pushforward laws with the physical data
law `y = Ax + ε` (and of `‖Ly⁻¹(Av)‖²` with `(Av)ᵀS_y⁻¹(Av)`) is stated in this docstring as the
paper's step, not silently claimed as machine-checked.
-/

open MeasureTheory Measure ProbabilityTheory
open scoped NNReal RealInnerProductSpace

universe u

namespace NearNull

/-! ### (0) Small generic facts -/

/-- **χ² is nonnegative** — it is the integral of a square. -/
theorem chi2_nonneg {Ω : Type*} [MeasurableSpace Ω] (P Q : Measure Ω) :
    0 ≤ NearNull.chi2 P Q :=
  integral_nonneg fun _ => sq_nonneg _

/-- `(r − 1)²` is `Q`-integrable when `r²` is (with `r = dP/dQ`), via `(r−1)² = r² − 2r + 1`. -/
private lemma integrable_rnDeriv_sub_one_sq {Ω : Type*} [MeasurableSpace Ω]
    (P Q : Measure Ω) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hsq : Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q) :
    Integrable (fun x => ((P.rnDeriv Q x).toReal - 1) ^ 2) Q := by
  have h2 : Integrable (fun x => (P.rnDeriv Q x).toReal) Q := Measure.integrable_toReal_rnDeriv
  refine ((hsq.sub (h2.const_mul 2)).add (integrable_const (1 : ℝ))).congr ?_
  filter_upwards with x
  simp only [Pi.add_apply, Pi.sub_apply]
  ring

/-! ### (1) Reusable L²-integrability transport (local re-proofs of `ChiSquaredPi` privates) -/

/-- The squared joint density is `Q⊗Q'`-integrable. Copied from `ChiSquaredPi.lean` (private there). -/
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

/-- Transport of the squared-density integrability across a measurable equivalence `e`. Copied from
`ChiSquaredPi.lean` (private there). -/
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

/-- **L² integrability of the density tensorizes.** Copied from `MultivariateGaussianWhitening.lean`
(private there). -/
private theorem integrable_rnDeriv_sq_pi :
    ∀ (n : ℕ) (X : Fin n → Type u) [∀ i, MeasurableSpace (X i)]
      (P Q : ∀ i, Measure (X i)) [∀ i, IsProbabilityMeasure (P i)] [∀ i, IsProbabilityMeasure (Q i)]
      (_hPQ : ∀ i, P i ≪ Q i)
      (_hInt : ∀ i, Integrable (fun x => (((P i).rnDeriv (Q i) x).toReal) ^ 2) (Q i)),
      Integrable (fun x => ((Measure.pi P).rnDeriv (Measure.pi Q) x).toReal ^ 2) (Measure.pi Q) := by
  intro n
  induction n with
  | zero =>
    intro X _ P Q _ _ _hPQ _hInt
    have hpieq : Measure.pi P = Measure.pi Q := by
      rw [Measure.pi_of_empty P, Measure.pi_of_empty Q]
    rw [hpieq]
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
    apply integrable_rnDeriv_sq_map (Measure.pi P) (Measure.pi Q) hac_pi e
    rw [he_P.map_eq, he_Q.map_eq]
    exact integrable_rnDeriv_sq_prod (P 0) (Q 0) _ _ (hPQ 0) hac_tail (hInt 0) ihTail

/-! ### (2) Mean-case privates copied from `MultivariateGaussianWhitening.lean` -/

/-- **Finiteness of the squared 1-D Gaussian density.** Copied from
`MultivariateGaussianWhitening.lean` (private there). -/
private lemma integrable_rnDeriv_sq_gaussianReal (m : ℝ) :
    Integrable
      (fun x => (((gaussianReal m 1).rnDeriv (gaussianReal (0 : ℝ) 1) x).toReal) ^ 2)
      (gaussianReal (0 : ℝ) 1) := by
  have hv1 : (1 : ℝ≥0) ≠ 0 := one_ne_zero
  have h1 := integrable_chi2_gaussianReal m 0 hv1
  have h2 : Integrable
      (fun x => ((gaussianReal m 1).rnDeriv (gaussianReal (0 : ℝ) 1) x).toReal)
      (gaussianReal (0 : ℝ) 1) := Measure.integrable_toReal_rnDeriv
  refine ((h1.add (h2.const_mul 2)).sub (integrable_const (1 : ℝ))).congr ?_
  filter_upwards with x
  simp only [Pi.add_apply, Pi.sub_apply]
  ring

/-- **L² integrability of `d(N₀(·+δ))/d(N₀)`.** Copied from `MultivariateGaussianWhitening.lean`
(private there). -/
private lemma integrable_rnDeriv_sq_stdGaussian_add {n : ℕ} (δ : EuclideanSpace ℝ (Fin n)) :
    Integrable
      (fun x => (((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)).rnDeriv
        (stdGaussian (EuclideanSpace ℝ (Fin n))) x).toReal ^ 2)
      (stdGaussian (EuclideanSpace ℝ (Fin n))) := by
  classical
  have hv1 : (1 : ℝ≥0) ≠ 0 := one_ne_zero
  set e₀ : (Fin n → ℝ) ≃ᵐ EuclideanSpace ℝ (Fin n) := MeasurableEquiv.toLp 2 (Fin n → ℝ) with he₀
  set δ' : Fin n → ℝ := WithLp.ofLp δ with hδ'
  have hac_i : ∀ i : Fin n, gaussianReal (δ' i) 1 ≪ gaussianReal (0 : ℝ) 1 := fun i =>
    (gaussianReal_absolutelyContinuous (δ' i) hv1).trans (gaussianReal_absolutelyContinuous' 0 hv1)
  have hInt_i : ∀ i : Fin n, Integrable
      (fun x => (((gaussianReal (δ' i) 1).rnDeriv (gaussianReal (0 : ℝ) 1) x).toReal) ^ 2)
      (gaussianReal (0 : ℝ) 1) := fun i => integrable_rnDeriv_sq_gaussianReal (δ' i)
  have hpi : Integrable
      (fun x => ((Measure.pi (fun i => gaussianReal (δ' i) 1)).rnDeriv
        (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)) x).toReal ^ 2)
      (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)) :=
    integrable_rnDeriv_sq_pi n (fun _ => ℝ) (fun i => gaussianReal (δ' i) 1)
      (fun _ => gaussianReal (0 : ℝ) 1) hac_i hInt_i
  have hstd : stdGaussian (EuclideanSpace ℝ (Fin n))
      = (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map e₀ := by
    rw [he₀, MeasurableEquiv.coe_toLp]; exact map_pi_eq_stdGaussian.symm
  have hconj : ((· + δ) ∘ e₀) = (e₀ ∘ (· + δ')) := by
    funext x
    simp only [Function.comp_apply, he₀, MeasurableEquiv.coe_toLp, hδ', WithLp.toLp_add,
      WithLp.toLp_ofLp]
  have hshift : (stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)
      = ((Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map (· + δ')).map e₀ := by
    rw [hstd, Measure.map_map (by fun_prop) e₀.measurable,
        Measure.map_map e₀.measurable (by fun_prop), hconj]
  have hpimap : (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map (· + δ')
      = Measure.pi (fun i => gaussianReal (δ' i) 1) := by
    have hfun : ((· + δ') : (Fin n → ℝ) → (Fin n → ℝ)) = (fun x i => x i + δ' i) := rfl
    rw [hfun, pi_map_pi (fun i => (measurable_add_const (δ' i)).aemeasurable)]
    congr 1
    funext i
    rw [gaussianReal_map_add_const, zero_add]
  have hM₁' : (stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)
      = (Measure.pi (fun i => gaussianReal (δ' i) 1)).map e₀ := by rw [hshift, hpimap]
  have hM₁map : ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)).map e₀.symm
      = Measure.pi (fun i => gaussianReal (δ' i) 1) := by rw [hM₁', e₀.map_symm_map]
  have hM₂map : (stdGaussian (EuclideanSpace ℝ (Fin n))).map e₀.symm
      = Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1) := by rw [hstd, e₀.map_symm_map]
  have hac_std : (stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)
      ≪ stdGaussian (EuclideanSpace ℝ (Fin n)) := stdGaussian_add_absolutelyContinuous δ
  haveI hprob_add :
      IsProbabilityMeasure ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  have key : Integrable
      (fun y => ((((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)).map e₀.symm).rnDeriv
        ((stdGaussian (EuclideanSpace ℝ (Fin n))).map e₀.symm) y).toReal ^ 2)
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map e₀.symm) := by
    rw [hM₁map, hM₂map]; exact hpi
  exact integrable_rnDeriv_sq_map ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ))
    (stdGaussian (EuclideanSpace ℝ (Fin n))) hac_std e₀.symm key

/-- **Affine whitening** (mean case). Copied from `MultivariateGaussianWhitening.lean`
(private there). -/
private lemma affine_whitening {n : ℕ} (μ₁ μ₂ : EuclideanSpace ℝ (Fin n))
    (L : EuclideanSpace ℝ (Fin n) ≃L[ℝ] EuclideanSpace ℝ (Fin n)) :
    ∃ e : EuclideanSpace ℝ (Fin n) ≃ᵐ EuclideanSpace ℝ (Fin n),
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₁ + L x)).map e
          = (stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + L.symm (μ₁ - μ₂))
      ∧ ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x)).map e
          = stdGaussian (EuclideanSpace ℝ (Fin n))
      ∧ (stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₁ + L x)
          ≪ (stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x) := by
  classical
  set δ : EuclideanSpace ℝ (Fin n) := L.symm (μ₁ - μ₂) with hδ
  set f₁ : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin n) := fun x => μ₁ + L x with hf₁
  set f₂ : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin n) := fun x => μ₂ + L x with hf₂
  have hf₁_meas : Measurable f₁ := by fun_prop
  have hf₂_meas : Measurable f₂ := by fun_prop
  have hadd_meas : Measurable (fun x : EuclideanSpace ℝ (Fin n) => x + δ) := by fun_prop
  set e : EuclideanSpace ℝ (Fin n) ≃ᵐ EuclideanSpace ℝ (Fin n) :=
    ((Homeomorph.subRight μ₂).trans L.symm.toHomeomorph).toMeasurableEquiv with he
  have he_apply : ∀ y : EuclideanSpace ℝ (Fin n), e y = L.symm (y - μ₂) := fun y => rfl
  have hef₁ : (e ∘ f₁) = (· + δ) := by
    funext x
    simp only [Function.comp_apply, he_apply, hf₁]
    rw [show μ₁ + L x - μ₂ = L x + (μ₁ - μ₂) from by abel]
    rw [map_add, ContinuousLinearEquiv.symm_apply_apply, ← hδ]
  have hef₂ : (e ∘ f₂) = id := by
    funext x
    simp only [Function.comp_apply, he_apply, hf₂, id_eq]
    rw [show μ₂ + L x - μ₂ = L x from by abel, ContinuousLinearEquiv.symm_apply_apply]
  have hcomp : (f₂ ∘ (· + δ)) = f₁ := by
    funext x
    simp only [Function.comp_apply, hf₁, hf₂]
    rw [show L (x + δ) = L x + (μ₁ - μ₂) from by
        rw [map_add, hδ, ContinuousLinearEquiv.apply_symm_apply]]
    abel
  refine ⟨e, ?_, ?_, ?_⟩
  · rw [Measure.map_map e.measurable hf₁_meas, hef₁]
  · rw [Measure.map_map e.measurable hf₂_meas, hef₂, Measure.map_id]
  · have hf₁eq : (stdGaussian (EuclideanSpace ℝ (Fin n))).map f₁
        = ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)).map f₂ := by
      rw [Measure.map_map hf₂_meas hadd_meas, hcomp]
    rw [hf₁eq]
    exact (stdGaussian_add_absolutelyContinuous δ).map hf₂_meas

/-- **The mean shift at general covariance is absolutely continuous:**
`N(μ₁, LLᵀ) ≪ N(μ₂, LLᵀ)` (as `stdGaussian` pushforwards). -/
theorem gaussian_mean_shift_ac {n : ℕ} (μ₁ μ₂ : EuclideanSpace ℝ (Fin n))
    (L : EuclideanSpace ℝ (Fin n) ≃L[ℝ] EuclideanSpace ℝ (Fin n)) :
    (stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₁ + L x)
      ≪ (stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x) :=
  (affine_whitening μ₁ μ₂ L).choose_spec.2.2

/-- **L² integrability of the mean-shift density at general covariance** (the Le Cam
side-condition), by transporting the standard-level fact through the whitening. -/
theorem gaussian_mean_shift_integrable {n : ℕ} (μ₁ μ₂ : EuclideanSpace ℝ (Fin n))
    (L : EuclideanSpace ℝ (Fin n) ≃L[ℝ] EuclideanSpace ℝ (Fin n)) :
    Integrable
      (fun y => ((((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₁ + L x)).rnDeriv
        ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x)) y).toReal) ^ 2)
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x)) := by
  classical
  obtain ⟨e, hM₁, hM₂, hac⟩ := affine_whitening μ₁ μ₂ L
  haveI : IsProbabilityMeasure
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₁ + L x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI : IsProbabilityMeasure
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  apply integrable_rnDeriv_sq_map _ _ hac e
  rw [hM₁, hM₂]
  exact integrable_rnDeriv_sq_stdGaussian_add (L.symm (μ₁ - μ₂))

/-! ### (3) The rank-one square root exists -/

/-- **Existence of the rank-one covariance square root.** For every `u` there is an `M` with
`M Mᵀ = I + u uᵀ` in quadratic-form terms: `‖Mᵀ ξ‖² = ‖ξ‖² + ⟪u,ξ⟫²` for all `ξ`. Witness:
`M = I + c · u uᵀ` with `c = (√(1+‖u‖²) − 1)/‖u‖²` (self-adjoint; `M = I` at `u = 0`). This makes
the spread-misreport hypotheses of the theorems below satisfiable for every direction. -/
theorem exists_rank1_sqrt {m : ℕ} (u : E m) :
    ∃ M : (E m) →L[ℝ] (E m),
      ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2 := by
  classical
  rcases eq_or_ne u 0 with hu | hu
  · refine ⟨ContinuousLinearMap.id ℝ (E m), fun ξ => ?_⟩
    rw [ContinuousLinearMap.adjoint_id, ContinuousLinearMap.id_apply, hu]
    simp
  · have hnu : (0 : ℝ) < ‖u‖ ^ 2 := by
      have h := norm_pos_iff.mpr hu
      positivity
    set c : ℝ := (Real.sqrt (1 + ‖u‖ ^ 2) - 1) / ‖u‖ ^ 2 with hc
    set M : (E m) →L[ℝ] (E m) :=
      ContinuousLinearMap.id ℝ (E m) + c • ((innerSL ℝ u).smulRight u) with hMdef
    have hM_apply : ∀ x : E m, M x = x + (c * inner ℝ u x) • u := by
      intro x
      simp only [hMdef, _root_.add_apply, ContinuousLinearMap.id_apply, _root_.smul_apply,
        ContinuousLinearMap.smulRight_apply, innerSL_apply_apply]
      rw [smul_smul]
    have hMsa : ContinuousLinearMap.adjoint M = M := by
      have h : M = ContinuousLinearMap.adjoint M := by
        rw [ContinuousLinearMap.eq_adjoint_iff]
        intro x y
        simp only [hMdef, _root_.add_apply, ContinuousLinearMap.id_apply, _root_.smul_apply,
          ContinuousLinearMap.smulRight_apply, innerSL_apply_apply]
        simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right]
        rw [real_inner_comm x u]
        ring
      exact h.symm
    have hct : c * ‖u‖ ^ 2 = Real.sqrt (1 + ‖u‖ ^ 2) - 1 := by
      rw [hc, div_mul_cancel₀ _ (ne_of_gt hnu)]
    have hs : Real.sqrt (1 + ‖u‖ ^ 2) ^ 2 = 1 + ‖u‖ ^ 2 := Real.sq_sqrt (by nlinarith)
    have hkey : 2 * c + c ^ 2 * ‖u‖ ^ 2 = 1 := by
      have h2 : (2 * c + c ^ 2 * ‖u‖ ^ 2) * ‖u‖ ^ 2 = 1 * ‖u‖ ^ 2 := by
        have hexp : (2 * c + c ^ 2 * ‖u‖ ^ 2) * ‖u‖ ^ 2
            = 2 * (c * ‖u‖ ^ 2) + (c * ‖u‖ ^ 2) ^ 2 := by ring
        rw [hexp, hct]
        nlinarith [hs]
      exact mul_right_cancel₀ (ne_of_gt hnu) h2
    refine ⟨M, fun ξ => ?_⟩
    rw [hMsa, hM_apply, norm_add_sq_real, real_inner_smul_right, norm_smul, Real.norm_eq_abs,
      mul_pow, sq_abs, real_inner_comm ξ u]
    linear_combination (inner ℝ ξ u) ^ 2 * hkey

/-! ### (4) (3a) General-covariance spread misreport: whitening of the rank-one bump -/

/-- **Spread whitening.** The affine whitening `e : y ↦ L⁻¹(y − μ)` conjugates the pair
`(N₀∘(μ+L·), N₀∘(μ+L(M·)))` — i.e. `N(μ, LLᵀ)` against the rank-one-bumped
`N(μ, LLᵀ + (Lu)(Lu)ᵀ)` when `M Mᵀ = I + u uᵀ` — to the standard pair `(N₀, N₀∘M)` of
`MultivariateGaussianSpread.lean`, and the source laws are absolutely continuous. -/
private lemma spread_whitening {m : ℕ} (μ : E m) (L : (E m) ≃L[ℝ] (E m)) (u : E m)
    (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2) :
    ∃ e : (E m) ≃ᵐ (E m),
      ((stdGaussian (E m)).map (fun x => μ + L x)).map e = stdGaussian (E m)
      ∧ ((stdGaussian (E m)).map (fun x => μ + L (M x))).map e = (stdGaussian (E m)).map M
      ∧ (stdGaussian (E m)).map (fun x => μ + L x)
          ≪ (stdGaussian (E m)).map (fun x => μ + L (M x)) := by
  classical
  set f : (E m) → (E m) := fun x => μ + L x with hf
  have hf_meas : Measurable f := by fun_prop
  have hM_meas : Measurable (M : (E m) → (E m)) := M.continuous.measurable
  have hfM_meas : Measurable (fun x : E m => μ + L (M x)) := by fun_prop
  set e : (E m) ≃ᵐ (E m) :=
    ((Homeomorph.subRight μ).trans L.symm.toHomeomorph).toMeasurableEquiv with he
  have he_apply : ∀ y : E m, e y = L.symm (y - μ) := fun y => rfl
  have hef : (e ∘ f) = id := by
    funext x
    simp only [Function.comp_apply, he_apply, hf, id_eq]
    rw [show μ + L x - μ = L x from by abel, ContinuousLinearEquiv.symm_apply_apply]
  have hefM : (e ∘ fun x => μ + L (M x)) = (M : (E m) → (E m)) := by
    funext x
    simp only [Function.comp_apply, he_apply]
    rw [show μ + L (M x) - μ = L (M x) from by abel, ContinuousLinearEquiv.symm_apply_apply]
  have hfMcomp : (fun x : E m => μ + L (M x)) = f ∘ (M : (E m) → (E m)) := rfl
  refine ⟨e, ?_, ?_, ?_⟩
  · rw [Measure.map_map e.measurable hf_meas, hef, Measure.map_id]
  · rw [Measure.map_map e.measurable hfM_meas, hefM]
  · rw [hfMcomp, ← Measure.map_map hf_meas hM_meas]
    exact (stdGaussian_absolutelyContinuous_map_rank1 u M hM).map hf_meas

/-- **(3a) General-covariance rank-one spread χ².** For an invertible square root `L`
(covariance `S = LLᵀ`) and a rank-one bump `M Mᵀ = I + u uᵀ`, the χ² of `N(μ, S)` from
`N(μ, S + (Lu)(Lu)ᵀ)` (realized as the pushforward by `x ↦ μ + L(Mx)`) is the standard-base
closed form `(1+‖u‖²)/√(1+2‖u‖²) − 1`. In matrix terms the perturbation is `w·a aᵀ` for
`a = Lu/√w`; the instantiation at the operator's data law is `operator_spread_misreport_chi2`. -/
theorem chi2_gaussian_spread_rank1 {m : ℕ} (μ : E m) (L : (E m) ≃L[ℝ] (E m)) (u : E m)
    (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2) :
    NearNull.chi2 ((stdGaussian (E m)).map (fun x => μ + L x))
        ((stdGaussian (E m)).map (fun x => μ + L (M x)))
      = (1 + ‖u‖ ^ 2) / Real.sqrt (1 + 2 * ‖u‖ ^ 2) - 1 := by
  classical
  obtain ⟨e, hP, hQ, hac⟩ := spread_whitening μ L u M hM
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => μ + L x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => μ + L (M x))) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  rw [← chi2_map_measurableEquiv _ _ hac e, hP, hQ]
  exact chi2_stdGaussian_rank1 u M hM

/-- **Absolute continuity of the general-covariance spread pair:** `N(μ,S) ≪ N(μ,S + (Lu)(Lu)ᵀ)`. -/
theorem gaussian_spread_rank1_ac {m : ℕ} (μ : E m) (L : (E m) ≃L[ℝ] (E m)) (u : E m)
    (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2) :
    (stdGaussian (E m)).map (fun x => μ + L x)
      ≪ (stdGaussian (E m)).map (fun x => μ + L (M x)) :=
  (spread_whitening μ L u M hM).choose_spec.2.2

/-- **L² integrability of the general-covariance spread density** (the Le Cam side-condition). -/
theorem gaussian_spread_rank1_integrable {m : ℕ} (μ : E m) (L : (E m) ≃L[ℝ] (E m)) (u : E m)
    (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2) :
    Integrable
      (fun y => ((((stdGaussian (E m)).map (fun x => μ + L x)).rnDeriv
        ((stdGaussian (E m)).map (fun x => μ + L (M x))) y).toReal) ^ 2)
      ((stdGaussian (E m)).map (fun x => μ + L (M x))) := by
  classical
  obtain ⟨e, hP, hQ, hac⟩ := spread_whitening μ L u M hM
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => μ + L x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => μ + L (M x))) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  apply integrable_rnDeriv_sq_map _ _ hac e
  rw [hP, hQ]
  exact integrable_rnDeriv_sq_stdGaussian_map_rank1 u M hM

/-- **(3a) payoff: undetectability of a general-covariance rank-one spread.** No test separates
`N(μ, S)` from `N(μ, S + (Lu)(Lu)ᵀ)` better than
`1 − √((1+‖u‖²)/√(1+2‖u‖²) − 1)` in summed error. -/
theorem gaussian_spread_rank1_undetectable {m : ℕ} (μ : E m) (L : (E m) ≃L[ℝ] (E m)) (u : E m)
    (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2)
    {A : Set (E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt ((1 + ‖u‖ ^ 2) / Real.sqrt (1 + 2 * ‖u‖ ^ 2) - 1)
      ≤ ((stdGaussian (E m)).map (fun x => μ + L x)).real A
        + ((stdGaussian (E m)).map (fun x => μ + L (M x))).real Aᶜ := by
  classical
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => μ + L x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => μ + L (M x))) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  have hac := gaussian_spread_rank1_ac μ L u M hM
  have hr2 := integrable_rnDeriv_sub_one_sq _ _ (gaussian_spread_rank1_integrable μ L u M hM)
  have h := NearNull.lecam_two_point_chi2 hac hr2 hA
  rwa [chi2_gaussian_spread_rank1 μ L u M hM] at h

/-! ### (5) (3b) The operator's data law: mean and spread misreports

The data space is `E m` and the model space `EuclideanSpace ℝ (Fin d)`. `Aop` is the forward
operator, `Ly` an invertible square root of the data covariance (`S_y = Ly Lyᵀ`; on paper
`S_y = A Σ⋆ Aᵀ + Γ` and `‖Ly⁻¹(Av)‖² = (Av)ᵀ S_y⁻¹ (Av)` — those identifications are the paper's,
not Lean's; see the module docstring). -/

variable {d m : ℕ}

/-- **(3b) Mean misreport at the operator's data law.** Two truths differing by `w·v` in the mean
induce data marginals `N(Aμ + w·Av, S_y)` and `N(Aμ, S_y)`, whose χ² is `exp(β) − 1` with
`β = w²‖Ly⁻¹(Av)‖²` (on paper `β = w²(Av)ᵀS_y⁻¹(Av)`). -/
theorem operator_mean_misreport_chi2
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) :
    NearNull.chi2
        ((stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
      = Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) - 1 := by
  rw [chi2_stdGaussian_affine (Aop μx + w • Aop v) (Aop μx) Ly,
    show Aop μx + w • Aop v - Aop μx = w • Aop v from by abel, map_smul, norm_smul, mul_pow,
    Real.norm_eq_abs, sq_abs]

/-- **(3b) Mean misreport undetectability.** Every test between the two data marginals of a
`w·v` mean misreport has summed error at least `1 − √(exp(w²‖Ly⁻¹(Av)‖²) − 1)`; as `Av → 0`
the bound tends to `1`. -/
theorem operator_mean_misreport_undetectable
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) {A : Set (E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt (Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) - 1)
      ≤ ((stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)).real A
        + ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real Aᶜ := by
  have h := multivariateGaussian_near_null_undetectable (Aop μx + w • Aop v) (Aop μx) Ly hA
  rwa [show Aop μx + w • Aop v - Aop μx = w • Aop v from by abel, map_smul, norm_smul, mul_pow,
    Real.norm_eq_abs, sq_abs] at h

/-- **(3b) Spread misreport at the operator's data law.** Two truths differing by a spread
misreport `Σ⋆ ↦ Σ⋆ + w v vᵀ` induce data marginals `N(Aμ, S_y)` and `N(Aμ, S_y + w(Av)(Av)ᵀ)`
(the latter realized through the rank-one square root `M` of `I + u uᵀ`, `u = √w·Ly⁻¹(Av)`,
which exists by `exists_rank1_sqrt`), whose χ² is `(1+β)/√(1+2β) − 1` with `β = w‖Ly⁻¹(Av)‖²`
(on paper `β = w(Av)ᵀS_y⁻¹(Av)`). -/
theorem operator_spread_misreport_chi2
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2) :
    NearNull.chi2
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)))
      = (1 + w * ‖Ly.symm (Aop v)‖ ^ 2)
          / Real.sqrt (1 + 2 * (w * ‖Ly.symm (Aop v)‖ ^ 2)) - 1 := by
  have hn : ‖Real.sqrt w • Ly.symm (Aop v)‖ ^ 2 = w * ‖Ly.symm (Aop v)‖ ^ 2 := by
    rw [norm_smul, mul_pow, Real.norm_eq_abs, sq_abs, Real.sq_sqrt hw]
  rw [chi2_gaussian_spread_rank1 (Aop μx) Ly (Real.sqrt w • Ly.symm (Aop v)) M hM, hn]

/-- **(3b) Spread misreport undetectability.** Every test between the two data marginals of a
`w·vvᵀ` spread misreport has summed error at least `1 − √((1+β)/√(1+2β) − 1)`,
`β = w‖Ly⁻¹(Av)‖²`; as `Av → 0` the bound tends to `1`. This is `p:nearnull`(ii) per survey. -/
theorem operator_spread_misreport_undetectable
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    {A : Set (E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt ((1 + w * ‖Ly.symm (Aop v)‖ ^ 2)
          / Real.sqrt (1 + 2 * (w * ‖Ly.symm (Aop v)‖ ^ 2)) - 1)
      ≤ ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real A
        + ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))).real Aᶜ := by
  have hn : ‖Real.sqrt w • Ly.symm (Aop v)‖ ^ 2 = w * ‖Ly.symm (Aop v)‖ ^ 2 := by
    rw [norm_smul, mul_pow, Real.norm_eq_abs, sq_abs, Real.sq_sqrt hw]
  have h := gaussian_spread_rank1_undetectable (Aop μx) Ly
    (Real.sqrt w • Ly.symm (Aop v)) M hM hA
  rwa [hn] at h

/-! ### (6) The `Av = 0` endpoint: `c:nonident` recovered -/

/-- **Endpoint (mean): at `Av = 0` the two data marginals coincide** and every test's summed
error is at least `1` — the exact non-identifiability of `c:nonident`. -/
theorem operator_mean_endpoint
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) (hAv : Aop v = 0) {A : Set (E m)} (hA : MeasurableSet A) :
    1 ≤ ((stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)).real A
        + ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real Aᶜ := by
  have hmapeq : (fun x : E m => (Aop μx + w • Aop v) + Ly x)
      = (fun x : E m => Aop μx + Ly x) := by
    funext x; rw [hAv, smul_zero, add_zero]
  rw [hmapeq]
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  exact lecam_exact rfl hA

/-- **Endpoint (mean, χ² form): at `Av = 0` the χ² vanishes.** -/
theorem operator_mean_endpoint_chi2
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) (hAv : Aop v = 0) :
    NearNull.chi2
        ((stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)) = 0 := by
  rw [operator_mean_misreport_chi2 Aop μx v w Ly, hAv]
  simp

/-- **Endpoint (spread): at `Av = 0` the two data marginals coincide** and every test's summed
error is at least `1` — `c:nonident` again, now for the spread misreport of `p:nearnull`(ii). -/
theorem operator_spread_endpoint
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    (hAv : Aop v = 0) {A : Set (E m)} (hA : MeasurableSet A) :
    1 ≤ ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real A
        + ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))).real Aᶜ := by
  classical
  have hu : (Real.sqrt w • Ly.symm (Aop v)) = 0 := by rw [hAv, map_zero, smul_zero]
  have hiso : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ = ‖ξ‖ := by
    intro ξ
    have hsq : ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 := by
      rw [hM ξ, hu, inner_zero_left]; ring
    nlinarith [norm_nonneg (ContinuousLinearMap.adjoint M ξ), norm_nonneg ξ, hsq]
  have hMstd : (stdGaussian (E m)).map M = stdGaussian (E m) :=
    stdGaussian_map_eq_self_of_isometry M hiso
  have hf_meas : Measurable (fun x : E m => Aop μx + Ly x) := by fun_prop
  have hM_meas : Measurable (M : (E m) → (E m)) := M.continuous.measurable
  have hQeq : (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly x) := by
    rw [show (fun x : E m => Aop μx + Ly (M x))
        = (fun x : E m => Aop μx + Ly x) ∘ (M : (E m) → (E m)) from rfl,
      ← Measure.map_map hf_meas hM_meas, hMstd]
  rw [hQeq]
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  exact lecam_exact rfl hA

/-- **Endpoint (spread, χ² form): at `Av = 0` the χ² vanishes.** -/
theorem operator_spread_endpoint_chi2
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    (hAv : Aop v = 0) :
    NearNull.chi2
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))) = 0 := by
  rw [operator_spread_misreport_chi2 Aop μx v hw Ly M hM, hAv]
  simp [Real.sqrt_one]

/-! ### (7) (3c) `n` iid surveys: tensorized χ², the Le Cam bound, and the crossover -/

/-- **(3c) iid tensorization: `χ²(Pⁿ ‖ Qⁿ) = (1+χ²)ⁿ − 1`** for `n` independent surveys. -/
theorem chi2_iid {Ω : Type u} [MeasurableSpace Ω] (P Q : Measure Ω)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q] (n : ℕ) (hPQ : P ≪ Q)
    (hInt : Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q) :
    NearNull.chi2 (Measure.pi fun _ : Fin n => P) (Measure.pi fun _ : Fin n => Q)
      = (NearNull.chi2 P Q + 1) ^ n - 1 := by
  rw [chi2_pi (fun _ : Fin n => P) (fun _ : Fin n => Q) (fun _ => hPQ) (fun _ => hInt)]
  simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin]

/-- **(3c) `n`-survey Le Cam bound.** Every test on the `n`-survey record has summed type-I and
type-II error at least `1 − √((1+χ²)ⁿ − 1)`: independent surveys multiply the per-survey
`1 + χ²`, and the Le Cam two-point χ² step converts the product into a testing floor. -/
theorem nsurvey_lecam {Ω : Type u} [MeasurableSpace Ω] (P Q : Measure Ω)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q] (n : ℕ) (hPQ : P ≪ Q)
    (hInt : Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q)
    {A : Set (Fin n → Ω)} (hA : MeasurableSet A) :
    1 - Real.sqrt ((NearNull.chi2 P Q + 1) ^ n - 1)
      ≤ (Measure.pi fun _ : Fin n => P).real A + (Measure.pi fun _ : Fin n => Q).real Aᶜ := by
  have hac : Measure.pi (fun _ : Fin n => P) ≪ Measure.pi (fun _ : Fin n => Q) :=
    pi_ac n (fun _ => Ω) (fun _ => P) (fun _ => Q) (fun _ => hPQ)
  have hsq : Integrable
      (fun x => (((Measure.pi fun _ : Fin n => P).rnDeriv (Measure.pi fun _ : Fin n => Q))
        x).toReal ^ 2) (Measure.pi fun _ : Fin n => Q) :=
    integrable_rnDeriv_sq_pi n (fun _ => Ω) (fun _ => P) (fun _ => Q) (fun _ => hPQ)
      (fun _ => hInt)
  have hr2 := integrable_rnDeriv_sub_one_sq _ _ hsq
  have h := NearNull.lecam_two_point_chi2 hac hr2 hA
  rwa [chi2_iid P Q n hPQ hInt] at h

/-- **(3c) the crossover, machine form.** Below the crossover — whenever `(1+χ²)ⁿ < 5/4` — every
test on `n` surveys has summed error strictly above `1/2`: the two truths cannot be separated
better than a coin flip leaves room for. (In the paper's scaling `1+χ²` is `1 + O(‖Av‖⁴)`, so the
admissible `n` grows as the inverse fourth power of `‖Av‖` — `eq:nearnull-cross`.) -/
theorem nsurvey_crossover {Ω : Type u} [MeasurableSpace Ω] (P Q : Measure Ω)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q] (n : ℕ) (hPQ : P ≪ Q)
    (hInt : Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q)
    (hn : (NearNull.chi2 P Q + 1) ^ n < 5 / 4)
    {A : Set (Fin n → Ω)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n => P).real A + (Measure.pi fun _ : Fin n => Q).real Aᶜ := by
  have h := nsurvey_lecam P Q n hPQ hInt hA
  have h1 : (1 : ℝ) ≤ NearNull.chi2 P Q + 1 := by linarith [chi2_nonneg P Q]
  have h0 : (0 : ℝ) ≤ (NearNull.chi2 P Q + 1) ^ n - 1 := by
    have := one_le_pow₀ h1 (n := n)
    linarith
  have hlt : (NearNull.chi2 P Q + 1) ^ n - 1 < 1 / 4 := by linarith
  have hs : Real.sqrt ((NearNull.chi2 P Q + 1) ^ n - 1) < 1 / 2 := by
    have hstep : Real.sqrt ((NearNull.chi2 P Q + 1) ^ n - 1) < Real.sqrt (1 / 4) :=
      Real.sqrt_lt_sqrt h0 hlt
    have hq : Real.sqrt (1 / 4 : ℝ) = 1 / 2 := by
      rw [show (1 / 4 : ℝ) = (1 / 2) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
    linarith [hstep, hq.le, hq.ge]
  linarith

/-! ### (8) The operator's `n`-survey statements: `p:nearnull`(ii) assembled -/

/-- **`n`-survey mean misreport at the operator's data law.** Every test on `n` iid surveys
between the two mean-misreport marginals has summed error at least
`1 − √(exp(β)ⁿ − 1)`, `β = w²‖Ly⁻¹(Av)‖²`. -/
theorem operator_mean_nsurvey_undetectable
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) (n : ℕ)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt ((Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2)) ^ n - 1)
      ≤ (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real Aᶜ := by
  haveI : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  have h := nsurvey_lecam _ _ n
    (gaussian_mean_shift_ac (Aop μx + w • Aop v) (Aop μx) Ly)
    (gaussian_mean_shift_integrable (Aop μx + w • Aop v) (Aop μx) Ly) hA
  rwa [show NearNull.chi2
        ((stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)) + 1
      = Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) from by
    rw [operator_mean_misreport_chi2 Aop μx v w Ly]; ring] at h

/-- **`n`-survey spread misreport at the operator's data law** — the machine form of
`p:nearnull`(ii)'s per-survey-to-`n`-survey assembly: summed error at least
`1 − √(((1+β)/√(1+2β))ⁿ − 1)`, `β = w‖Ly⁻¹(Av)‖²`. -/
theorem operator_spread_nsurvey_undetectable
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    (n : ℕ) {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt (((1 + w * ‖Ly.symm (Aop v)‖ ^ 2)
          / Real.sqrt (1 + 2 * (w * ‖Ly.symm (Aop v)‖ ^ 2))) ^ n - 1)
      ≤ (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))).real Aᶜ := by
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  have h := nsurvey_lecam _ _ n
    (gaussian_spread_rank1_ac (Aop μx) Ly (Real.sqrt w • Ly.symm (Aop v)) M hM)
    (gaussian_spread_rank1_integrable (Aop μx) Ly (Real.sqrt w • Ly.symm (Aop v)) M hM) hA
  rwa [show NearNull.chi2
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))) + 1
      = (1 + w * ‖Ly.symm (Aop v)‖ ^ 2)
          / Real.sqrt (1 + 2 * (w * ‖Ly.symm (Aop v)‖ ^ 2)) from by
    rw [operator_spread_misreport_chi2 Aop μx v hw Ly M hM]; ring] at h

/-- **Operator-level crossover (mean).** If `exp(β)ⁿ < 5/4`, `β = w²‖Ly⁻¹(Av)‖²`, then every
test on `n` surveys of the mean misreport has summed error above `1/2`. -/
theorem operator_mean_crossover
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) (n : ℕ)
    (hn : (Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2)) ^ n < 5 / 4)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real Aᶜ := by
  have h := operator_mean_nsurvey_undetectable Aop μx v w Ly n hA
  have h1 : (1 : ℝ) ≤ Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) := by
    rw [show (1 : ℝ) = Real.exp 0 from (Real.exp_zero).symm]
    exact Real.exp_le_exp.mpr (by positivity)
  have h0 : (0 : ℝ) ≤ (Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2)) ^ n - 1 := by
    have := one_le_pow₀ h1 (n := n)
    linarith
  have hs : Real.sqrt ((Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2)) ^ n - 1) < 1 / 2 := by
    have hstep : Real.sqrt ((Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2)) ^ n - 1)
        < Real.sqrt (1 / 4) := Real.sqrt_lt_sqrt h0 (by linarith)
    have hq : Real.sqrt (1 / 4 : ℝ) = 1 / 2 := by
      rw [show (1 / 4 : ℝ) = (1 / 2) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
    linarith
  linarith

/-- **Operator-level crossover (spread) — the paper's `n < n⋆` in machine form.** If
`((1+β)/√(1+2β))ⁿ < 5/4`, `β = w‖Ly⁻¹(Av)‖²`, then every test on `n` surveys of the spread
misreport has summed type-I plus type-II error strictly above `1/2`. Since
`(1+β)/√(1+2β) = 1 + O(β²)` as `β → 0` and `β ≤ w‖Av‖²/λ_min(Γ)` on paper, the admissible `n`
grows as the inverse fourth power of `‖Av‖` — `eq:nearnull-cross` of `p:nearnull`(ii). -/
theorem operator_spread_crossover
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    (n : ℕ)
    (hn : ((1 + w * ‖Ly.symm (Aop v)‖ ^ 2)
          / Real.sqrt (1 + 2 * (w * ‖Ly.symm (Aop v)‖ ^ 2))) ^ n < 5 / 4)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))).real Aᶜ := by
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI : IsProbabilityMeasure ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  have hchi : NearNull.chi2
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))) + 1
      = (1 + w * ‖Ly.symm (Aop v)‖ ^ 2)
          / Real.sqrt (1 + 2 * (w * ‖Ly.symm (Aop v)‖ ^ 2)) := by
    rw [operator_spread_misreport_chi2 Aop μx v hw Ly M hM]; ring
  refine nsurvey_crossover _ _ n
    (gaussian_spread_rank1_ac (Aop μx) Ly (Real.sqrt w • Ly.symm (Aop v)) M hM)
    (gaussian_spread_rank1_integrable (Aop μx) Ly (Real.sqrt w • Ly.symm (Aop v)) M hM)
    ?_ hA
  rw [hchi]
  exact hn

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.chi2_nonneg
#print axioms NearNull.gaussian_mean_shift_ac
#print axioms NearNull.gaussian_mean_shift_integrable
#print axioms NearNull.exists_rank1_sqrt
#print axioms NearNull.chi2_gaussian_spread_rank1
#print axioms NearNull.gaussian_spread_rank1_ac
#print axioms NearNull.gaussian_spread_rank1_integrable
#print axioms NearNull.gaussian_spread_rank1_undetectable
#print axioms NearNull.operator_mean_misreport_chi2
#print axioms NearNull.operator_mean_misreport_undetectable
#print axioms NearNull.operator_spread_misreport_chi2
#print axioms NearNull.operator_spread_misreport_undetectable
#print axioms NearNull.operator_mean_endpoint
#print axioms NearNull.operator_mean_endpoint_chi2
#print axioms NearNull.operator_spread_endpoint
#print axioms NearNull.operator_spread_endpoint_chi2
#print axioms NearNull.chi2_iid
#print axioms NearNull.nsurvey_lecam
#print axioms NearNull.nsurvey_crossover
#print axioms NearNull.operator_mean_nsurvey_undetectable
#print axioms NearNull.operator_spread_nsurvey_undetectable
#print axioms NearNull.operator_mean_crossover
#print axioms NearNull.operator_spread_crossover
