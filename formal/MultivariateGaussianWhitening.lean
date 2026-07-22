import Mathlib
import ChiSquaredTV
import ChiSquaredInvariance
import ChiSquaredPi
import GaussianChiSquared
import MultivariateGaussianChiSquared

/-!
# General-covariance multivariate mean-shift χ² and near-null undetectability

Two `m`-dimensional Gaussians with a **common covariance** `LLᵀ` (`L` an invertible linear
"square root", `L : E ≃L[ℝ] E`, `E := EuclideanSpace ℝ (Fin n)`) and means `μ₁`, `μ₂` are the pushforwards
of the standard Gaussian by the affine maps `x ↦ μᵢ + L x`. This file computes their χ² in closed form
and feeds it through the Le Cam engine to obtain the **near-null undetectability** of a mean shift that
is small in the Mahalanobis metric `‖L⁻¹(μ₁−μ₂)‖² = (μ₁−μ₂)ᵀ(LLᵀ)⁻¹(μ₁−μ₂)`.

The route is an **affine whitening**. The measurable equivalence `e : y ↦ L⁻¹(y − μ₂)` conjugates the two
affine maps to the identity-covariance mean-shift picture (`e ∘ (μ₁ + L ·) = (· + δ)` with
`δ := L⁻¹(μ₁ − μ₂)`, and `e ∘ (μ₂ + L ·) = id`). Since χ² is invariant under a bimeasurable change of
variables (`NearNull.chi2_map_measurableEquiv`), the general χ² collapses to the isotropic one
(`NearNull.chi2_stdGaussian_add`, `MultivariateGaussianChiSquared.lean`), giving `exp(‖δ‖²) − 1`.

## Results
* `NearNull.stdGaussian_add_absolutelyContinuous` — `N₀(·+δ) ≪ N₀`, the shift a.c. helper.
* `NearNull.chi2_stdGaussian_affine` — `χ²(N₀(μ₁+L·) ‖ N₀(μ₂+L·)) = exp(‖L⁻¹(μ₁−μ₂)‖²) − 1`.
* `NearNull.multivariateGaussian_near_null_undetectable` — the payoff:
  `1 − √(exp(‖L⁻¹(μ₁−μ₂)‖²) − 1) ≤ P₁.real A + P₂.real Aᶜ` for every test `A`. As `μ₁ → μ₂` the left side
  `→ 1`, so no test distinguishes two Gaussians whose means differ by a near-null (Mahalanobis) amount.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each of the three results lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure ProbabilityTheory
open scoped NNReal

universe u

namespace NearNull

/-! ### Reusable L²-integrability transport (local re-proofs of `ChiSquaredPi` private lemmas) -/

/-- **Finiteness of the squared 1-D Gaussian density.** Copied from `MultivariateGaussianChiSquared.lean`
(private there): from the near-null χ² finiteness `integrable_chi2_gaussianReal` and
`Measure.integrable_toReal_rnDeriv`, via `r² = (r−1)² + 2r − 1`. -/
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

/-- Transport of the squared-density integrability across a measurable equivalence `e` (from the pushed
level down to the base level). Copied from `ChiSquaredPi.lean` (private there). -/
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

/-- **L² integrability of the density tensorizes.** The squared density of a finite product law is
integrable against the product base. Induction on `n` mirroring `ChiSquaredPi.chi2_pi_aux` (its
integrability branch), carried through `MeasurableEquiv.piFinSuccAbove` by `integrable_rnDeriv_sq_prod`
(head × tail) and `integrable_rnDeriv_sq_map`. -/
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

/-! ### (1) Absolute continuity of the isotropic mean shift -/

/-- **`N₀(·+δ) ≪ N₀`.** The standard Gaussian on `EuclideanSpace ℝ (Fin n)` shifted by `δ` is absolutely
continuous w.r.t. the standard Gaussian. Reduction to a product of shifted 1-D Gaussians via `toLp`,
coordinatewise a.c. by `pi_ac`, and `AbsolutelyContinuous.map`. -/
theorem stdGaussian_add_absolutelyContinuous {n : ℕ} (δ : EuclideanSpace ℝ (Fin n)) :
    (stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)
      ≪ stdGaussian (EuclideanSpace ℝ (Fin n)) := by
  classical
  have hv1 : (1 : ℝ≥0) ≠ 0 := one_ne_zero
  set e : (Fin n → ℝ) ≃ᵐ EuclideanSpace ℝ (Fin n) := MeasurableEquiv.toLp 2 (Fin n → ℝ) with he
  set δ' : Fin n → ℝ := WithLp.ofLp δ with hδ'
  have hac_i : ∀ i : Fin n, gaussianReal (δ' i) 1 ≪ gaussianReal (0 : ℝ) 1 := fun i =>
    (gaussianReal_absolutelyContinuous (δ' i) hv1).trans (gaussianReal_absolutelyContinuous' 0 hv1)
  have hstd : stdGaussian (EuclideanSpace ℝ (Fin n))
      = (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map e := by
    rw [he, MeasurableEquiv.coe_toLp]; exact map_pi_eq_stdGaussian.symm
  have hconj : ((· + δ) ∘ e) = (e ∘ (· + δ')) := by
    funext x
    simp only [Function.comp_apply, he, MeasurableEquiv.coe_toLp, hδ', WithLp.toLp_add,
      WithLp.toLp_ofLp]
  have hshift : (stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)
      = ((Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map (· + δ')).map e := by
    rw [hstd, Measure.map_map (by fun_prop) e.measurable,
        Measure.map_map e.measurable (by fun_prop), hconj]
  have hpimap : (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map (· + δ')
      = Measure.pi (fun i => gaussianReal (δ' i) 1) := by
    have hfun : ((· + δ') : (Fin n → ℝ) → (Fin n → ℝ)) = (fun x i => x i + δ' i) := rfl
    rw [hfun, pi_map_pi (fun i => (measurable_add_const (δ' i)).aemeasurable)]
    congr 1
    funext i
    rw [gaussianReal_map_add_const, zero_add]
  have hac : Measure.pi (fun i => gaussianReal (δ' i) 1)
      ≪ Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1) :=
    pi_ac n (fun _ => ℝ) (fun i => gaussianReal (δ' i) 1)
      (fun _ => gaussianReal (0 : ℝ) 1) hac_i
  rw [hshift, hpimap, hstd]
  exact hac.map e.measurable

/-! ### L² integrability of the isotropic mean-shift density (std level) -/

/-- **L² integrability of `d(N₀(·+δ))/d(N₀)`.** Obtained by transporting the product-level L²
integrability (`integrable_rnDeriv_sq_pi` fed by the 1-D `integrable_rnDeriv_sq_gaussianReal`) up to the
Euclidean level through `toLp`. -/
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
  -- bridge std level ↔ pi level (as in `chi2_stdGaussian_add`)
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
  -- transport pi → std (up) as the down-lemma applied to `e₀.symm`
  have key : Integrable
      (fun y => ((((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ)).map e₀.symm).rnDeriv
        ((stdGaussian (EuclideanSpace ℝ (Fin n))).map e₀.symm) y).toReal ^ 2)
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map e₀.symm) := by
    rw [hM₁map, hM₂map]; exact hpi
  exact integrable_rnDeriv_sq_map ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ))
    (stdGaussian (EuclideanSpace ℝ (Fin n))) hac_std e₀.symm key

/-! ### The affine whitening equivalence -/

/-- **Affine whitening.** The equivalence `e : y ↦ L⁻¹(y − μ₂)` conjugates the two affine pushforwards
to the isotropic mean-shift picture: `((N₀ ∘ (μ₁+L·)).map e = N₀(·+δ)`, `(N₀ ∘ (μ₂+L·)).map e = N₀`
(`δ := L⁻¹(μ₁−μ₂)`), and the source laws are absolutely continuous. Both `chi2_stdGaussian_affine` and
the undetectability corollary read off this single reduction. -/
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

/-! ### (2) General-covariance mean-shift χ² -/

/-- **General-covariance multivariate mean-shift χ².** For a common covariance square root
`L : E ≃L[ℝ] E` and means `μ₁`, `μ₂`,
`χ²(N₀(μ₁+L·) ‖ N₀(μ₂+L·)) = exp(‖L⁻¹(μ₁−μ₂)‖²) − 1`. The exponent is the squared Mahalanobis mean gap
`(μ₁−μ₂)ᵀ(LLᵀ)⁻¹(μ₁−μ₂)`. Proved by affine whitening + χ² change-of-variables invariance, reducing to the
isotropic `chi2_stdGaussian_add`. -/
theorem chi2_stdGaussian_affine {n : ℕ} (μ₁ μ₂ : EuclideanSpace ℝ (Fin n))
    (L : EuclideanSpace ℝ (Fin n) ≃L[ℝ] EuclideanSpace ℝ (Fin n)) :
    NearNull.chi2 (stdGaussian (EuclideanSpace ℝ (Fin n)) |>.map (fun x => μ₁ + L x))
                 (stdGaussian (EuclideanSpace ℝ (Fin n)) |>.map (fun x => μ₂ + L x))
      = Real.exp (‖L.symm (μ₁ - μ₂)‖ ^ 2) - 1 := by
  classical
  obtain ⟨e, hM₁, hM₂, hac⟩ := affine_whitening μ₁ μ₂ L
  haveI hPp : IsProbabilityMeasure
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₁ + L x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI hQp : IsProbabilityMeasure
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  rw [← chi2_map_measurableEquiv _ _ hac e, hM₁, hM₂]
  exact chi2_stdGaussian_add (L.symm (μ₁ - μ₂))

/-! ### (3) The near-null undetectability payoff -/

/-- **Near-null undetectability of a Mahalanobis mean shift.** Two `m`-dimensional Gaussians with common
covariance `LLᵀ` and means `μ₁`, `μ₂` cannot be told apart when their means are close in the Mahalanobis
metric: for every test `A`,
`1 − √(exp(‖L⁻¹(μ₁−μ₂)‖²) − 1) ≤ P₁.real A + P₂.real Aᶜ`. As `μ₁ → μ₂` the left side `→ 1`. Instantiates the
Le Cam two-point χ² bound at the general χ² of `chi2_stdGaussian_affine`, with the finiteness side
condition transported to the Euclidean level. -/
theorem multivariateGaussian_near_null_undetectable {n : ℕ}
    (μ₁ μ₂ : EuclideanSpace ℝ (Fin n))
    (L : EuclideanSpace ℝ (Fin n) ≃L[ℝ] EuclideanSpace ℝ (Fin n))
    {A : Set (EuclideanSpace ℝ (Fin n))} (hA : MeasurableSet A) :
    1 - Real.sqrt (Real.exp (‖L.symm (μ₁ - μ₂)‖ ^ 2) - 1)
      ≤ (stdGaussian _ |>.map (fun x => μ₁ + L x)).real A
        + (stdGaussian _ |>.map (fun x => μ₂ + L x)).real Aᶜ := by
  classical
  obtain ⟨e, hM₁, hM₂, hac⟩ := affine_whitening μ₁ μ₂ L
  set P := (stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₁ + L x) with hP
  set Q := (stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x) with hQ
  set δ : EuclideanSpace ℝ (Fin n) := L.symm (μ₁ - μ₂) with hδ
  haveI hPp : IsProbabilityMeasure P := by
    rw [hP]; exact isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI hQp : IsProbabilityMeasure Q := by
    rw [hQ]; exact isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  -- L² integrability of `d P / d Q`, by transporting the std-level fact through the whitening `e`
  have hsq : Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q := by
    apply integrable_rnDeriv_sq_map P Q hac e
    rw [hM₁, hM₂]
    exact integrable_rnDeriv_sq_stdGaussian_add δ
  -- `(r − 1)²` integrable, from `r²` and `r` integrable
  have hr2 : Integrable (fun x => ((P.rnDeriv Q x).toReal - 1) ^ 2) Q := by
    have h2 : Integrable (fun x => (P.rnDeriv Q x).toReal) Q := Measure.integrable_toReal_rnDeriv
    refine ((hsq.sub (h2.const_mul 2)).add (integrable_const (1 : ℝ))).congr ?_
    filter_upwards with x
    simp only [Pi.add_apply, Pi.sub_apply]
    ring
  have hchi : NearNull.chi2 P Q = Real.exp (‖δ‖ ^ 2) - 1 := by
    rw [hP, hQ, hδ]; exact chi2_stdGaussian_affine μ₁ μ₂ L
  have h := NearNull.lecam_two_point_chi2 hac hr2 hA
  rw [hchi] at h
  exact h

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.stdGaussian_add_absolutelyContinuous
#print axioms NearNull.chi2_stdGaussian_affine
#print axioms NearNull.multivariateGaussian_near_null_undetectable
