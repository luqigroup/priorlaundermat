import Mathlib
import ChiSquaredTV
import ChiSquaredInvariance
import ChiSquaredPi
import GaussianChiSquaredVar
import MultivariateGaussianScale
import MultivariateGaussianWhitening

/-!
# Rank-one covariance spread χ² and near-null undetectability

The standard Gaussian on `E n := EuclideanSpace ℝ (Fin (n+1))` against its pushforward by a linear
"square root" `L` whose covariance is a **rank-one bump of the identity**, `L Lᵀ = I + u uᵀ`, has χ²
`(1 + ‖u‖²)/√(1 + 2‖u‖²) − 1`. This is the multivariate spread-misreport χ² on a single blind
direction `u`: the perturbed law inflates the variance only along `u`.

The route is a **whitening rotation**. Because the `stdGaussian` pushforward depends on `L` only through
the covariance quadratic form `ξ ↦ ‖L.adjoint ξ‖` (`stdGaussian_map_eq_of_adjoint`, proved from the
characteristic-function uniqueness `Measure.ext_of_charFunDual` + `charFunDual_stdGaussian`), an
orthonormal rotation `R` carrying `u` to `‖u‖ · e₀` reduces the rank-one bump to the single-coordinate
diagonal scaling of `MultivariateGaussianScale.lean` (`chi2_stdGaussian_scale_single`). χ² invariance
under the bimeasurable `R` (`NearNull.chi2_map_measurableEquiv`) then transfers the closed form.

## Results
* `NearNull.stdGaussian_map_eq_of_adjoint` — the covariance-quadratic-form uniqueness of the pushforward.
* `NearNull.chi2_stdGaussian_rank1` — `χ²(N₀ ‖ N₀∘L) = (1+‖u‖²)/√(1+2‖u‖²) − 1` for `L Lᵀ = I + u uᵀ`.
* `NearNull.chi2_stdGaussian_rank1_undetectable` — the Le Cam payoff: for every test `A`,
  `1 − √((1+‖u‖²)/√(1+2‖u‖²) − 1) ≤ N₀.real A + (N₀∘L).real Aᶜ`. As `u → 0` the left side `→ 1`.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure ProbabilityTheory
open scoped NNReal RealInnerProductSpace

universe u'

namespace NearNull

/-- The Euclidean carrier `E n = EuclideanSpace ℝ (Fin (n+1))` (dimension `n+1 ≥ 1`). -/
abbrev E (n : ℕ) := EuclideanSpace ℝ (Fin (n + 1))

/-- The first standard basis vector `e₀ = single 0 1`. -/
noncomputable def e0 (n : ℕ) : E n := EuclideanSpace.single (0 : Fin (n + 1)) (1 : ℝ)

lemma inner_e0_left {n : ℕ} (x : E n) : ⟪e0 n, x⟫ = x 0 := by
  simp [e0, EuclideanSpace.inner_single_left]

lemma inner_e0_right {n : ℕ} (x : E n) : ⟪x, e0 n⟫ = x 0 := by
  simp [e0, EuclideanSpace.inner_single_right]

lemma norm_e0 {n : ℕ} : ‖e0 n‖ = 1 := by simp [e0]

lemma e0_apply {n : ℕ} (i : Fin (n + 1)) : (e0 n) i = if i = 0 then (1 : ℝ) else 0 := by
  simp only [e0, PiLp.single_apply]

/-! ### (1) Covariance-quadratic-form uniqueness of the `stdGaussian` pushforward -/

/-- **The `stdGaussian` pushforward depends on `A` only through the covariance quadratic form**
`ξ ↦ ‖A.adjoint ξ‖`. If two continuous linear maps `A`, `B` have `‖A.adjoint ξ‖ = ‖B.adjoint ξ‖` for
all `ξ` (i.e. `A Aᵀ = B Bᵀ`), their pushforwards of the standard Gaussian coincide. Proved by the
characteristic-function uniqueness `Measure.ext_of_charFunDual`: `charFunDual (N₀.map A) L`
`= exp(−‖L ∘ A‖²/2)`, and `‖L ∘ A‖ = ‖A.adjoint (riesz L)‖` via the Riesz identification. -/
theorem stdGaussian_map_eq_of_adjoint {n : ℕ}
    (A B : (E n) →L[ℝ] (E n))
    (h : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint A ξ‖ = ‖ContinuousLinearMap.adjoint B ξ‖) :
    (stdGaussian (E n)).map A = (stdGaussian (E n)).map B := by
  apply Measure.ext_of_charFunDual
  ext L
  rw [charFunDual_map, charFunDual_map, charFunDual_stdGaussian, charFunDual_stdGaussian]
  set v := (InnerProductSpace.toDual ℝ (E n)).symm L with hv
  have hL : L = InnerProductSpace.toDual ℝ (E n) v := by rw [hv]; simp
  have hbridge : ∀ (M : (E n) →L[ℝ] (E n)),
      ‖L.comp M‖ = ‖ContinuousLinearMap.adjoint M v‖ := by
    intro M
    have hcomp : L.comp M = InnerProductSpace.toDual ℝ (E n) (ContinuousLinearMap.adjoint M v) := by
      ext w
      simp only [ContinuousLinearMap.comp_apply, InnerProductSpace.toDual_apply_apply]
      rw [hL, InnerProductSpace.toDual_apply_apply, ContinuousLinearMap.adjoint_inner_left]
    rw [hcomp, LinearIsometryEquiv.norm_map]
  rw [hbridge A, hbridge B, h v]

/-! ### (2) The diagonal square-root map `Scp` and its adjoint quadratic form -/

/-- The diagonal "spread" square-root scaling coordinate `0` by `s` and fixing the rest:
`Scp s x = x + (s−1)·x₀·e₀`. Packaged as a continuous linear map so `stdGaussian.map (Scp s)` has a
computable characteristic function. -/
noncomputable def Scp (n : ℕ) (s : ℝ) : (E n) →L[ℝ] (E n) :=
  ContinuousLinearMap.id ℝ (E n) + (s - 1) • ((innerSL ℝ (e0 n)).smulRight (e0 n))

lemma Scp_apply {n : ℕ} (s : ℝ) (x : E n) : Scp n s x = x + ((s - 1) * (x 0)) • (e0 n) := by
  simp only [Scp, _root_.add_apply, ContinuousLinearMap.id_apply, _root_.smul_apply,
    ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, inner_e0_left]
  rw [smul_smul]

/-- `Scp s` is self-adjoint (it is a real diagonal matrix). -/
lemma Scp_selfadjoint {n : ℕ} (s : ℝ) :
    ContinuousLinearMap.adjoint (Scp n s) = Scp n s := by
  have h : (Scp n s) = ContinuousLinearMap.adjoint (Scp n s) := by
    rw [ContinuousLinearMap.eq_adjoint_iff]
    intro x y
    simp only [Scp, _root_.add_apply, ContinuousLinearMap.id_apply, _root_.smul_apply,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply]
    simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right,
      inner_e0_left, inner_e0_right]
    ring
  exact h.symm

/-- The covariance quadratic form of `Scp s`: `‖Scp(s).adjoint ξ‖² = ‖ξ‖² + (s²−1)·ξ₀²`. -/
lemma Scp_adjoint_norm_sq {n : ℕ} (s : ℝ) (ξ : E n) :
    ‖ContinuousLinearMap.adjoint (Scp n s) ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (s ^ 2 - 1) * (ξ 0) ^ 2 := by
  rw [Scp_selfadjoint, Scp_apply, norm_add_sq_real, norm_smul, norm_e0, mul_one,
    real_inner_smul_right, inner_e0_right, Real.norm_eq_abs, sq_abs]
  ring

/-- The update-form scaling map used in `chi2_stdGaussian_scale_single` equals `Scp s`. -/
lemma Scp_eq_update {n : ℕ} (s : ℝ) :
    (fun x : E n => Scp n s x)
      = (fun x : E n =>
          (WithLp.toLp 2 (Function.update (WithLp.ofLp x) 0 (s * WithLp.ofLp x 0)) : E n)) := by
  funext x
  rw [Scp_apply]
  ext i
  rcases eq_or_ne i 0 with hi | hi
  · subst hi
    simp only [PiLp.add_apply, PiLp.smul_apply, smul_eq_mul, e0_apply, if_true,
      Function.update_self]
    ring
  · simp only [PiLp.add_apply, PiLp.smul_apply, smul_eq_mul, e0_apply, if_neg hi, mul_zero,
      add_zero, Function.update_of_ne hi]

/-- `Scp s` in the coordinatewise-scaling form used by `MultivariateGaussianScale.lean`. -/
lemma Scp_eq_diag {n : ℕ} (s : ℝ) :
    (fun x : E n => Scp n s x)
      = (fun x : E n =>
          (WithLp.toLp 2 (fun i => (Function.update (fun _ => (1 : ℝ)) 0 s i) * WithLp.ofLp x i) :
            E n)) := by
  funext x
  rw [Scp_apply]
  ext i
  rcases eq_or_ne i 0 with hi | hi
  · subst hi
    simp only [PiLp.add_apply, PiLp.smul_apply, smul_eq_mul, e0_apply, if_true,
      Function.update_self]
    ring
  · simp only [PiLp.add_apply, PiLp.smul_apply, smul_eq_mul, e0_apply, if_neg hi, mul_zero,
      add_zero, Function.update_of_ne hi, one_mul]

/-! ### (3) Reusable L²-integrability transport (local re-proofs of the `ChiSquaredPi` privates) -/

/-- **Finiteness of the squared 1-D Gaussian density** (spread-mismatch case). Copied from
`MultivariateGaussianScale.lean` (private there). -/
private lemma integrable_rnDeriv_sq_gaussianReal_var (m : ℝ) {v₁ v₂ : ℝ≥0}
    (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0) (hlt : (v₁ : ℝ) < 2 * v₂) :
    Integrable
      (fun x => (((gaussianReal m v₁).rnDeriv (gaussianReal m v₂) x).toReal) ^ 2)
      (gaussianReal m v₂) := by
  have h1 := integrable_chi2_gaussianReal_var m hv₁ hv₂ hlt
  have h2 : Integrable
      (fun x => ((gaussianReal m v₁).rnDeriv (gaussianReal m v₂) x).toReal)
      (gaussianReal m v₂) := Measure.integrable_toReal_rnDeriv
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
    ∀ (n : ℕ) (X : Fin n → Type u') [∀ i, MeasurableSpace (X i)]
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

/-! ### (4) The diagonal `Scp`-scaled Gaussian: pi-decomposition, a.c., L² integrability -/

/-- **The `Scp`-scaled standard Gaussian as a product of scaled 1-D Gaussians** (via `toLp`). -/
lemma stdGaussian_map_Scp {n : ℕ} (s : ℝ) :
    (stdGaussian (E n)).map (Scp n s)
      = (Measure.pi (fun i => gaussianReal (0 : ℝ)
          (⟨(Function.update (fun _ => (1 : ℝ)) 0 s i) ^ 2, sq_nonneg _⟩ : ℝ≥0))).map
          (MeasurableEquiv.toLp 2 (Fin (n + 1) → ℝ)) := by
  classical
  set c : Fin (n + 1) → ℝ := Function.update (fun _ => (1 : ℝ)) 0 s with hc_def
  set sN : Fin (n + 1) → ℝ≥0 := fun i => ⟨(c i) ^ 2, sq_nonneg (c i)⟩ with hsN
  set e : (Fin (n + 1) → ℝ) ≃ᵐ E n := MeasurableEquiv.toLp 2 (Fin (n + 1) → ℝ) with he
  set sc : (Fin (n + 1) → ℝ) → (Fin (n + 1) → ℝ) := fun y i => c i * y i with hsc
  have hsc_meas : Measurable sc := by rw [hsc]; fun_prop
  have hstd : stdGaussian (E n)
      = (Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1)).map e := by
    rw [he, MeasurableEquiv.coe_toLp]; exact map_pi_eq_stdGaussian.symm
  have hScp_meas : Measurable (fun x : E n => Scp n s x) := (Scp n s).continuous.measurable
  have hconj : ((fun x : E n => Scp n s x) ∘ e) = (e ∘ sc) := by
    rw [Scp_eq_diag]
    funext y
    simp only [Function.comp_apply, he, MeasurableEquiv.coe_toLp, hsc]
    rw [← hc_def]
  have hscale : (stdGaussian (E n)).map (fun x : E n => Scp n s x)
      = ((Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1)).map sc).map e := by
    rw [hstd, Measure.map_map hScp_meas e.measurable, Measure.map_map e.measurable hsc_meas, hconj]
  have hpimap : (Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1)).map sc
      = Measure.pi (fun i => gaussianReal (0 : ℝ) (sN i)) := by
    rw [hsc, pi_map_pi (fun i => (measurable_const_mul (c i)).aemeasurable)]
    congr 1
    funext i
    rw [gaussianReal_map_const_mul (c i), mul_zero, mul_one]
    rfl
  rw [show (stdGaussian (E n)).map (Scp n s) = (stdGaussian (E n)).map (fun x : E n => Scp n s x)
      from rfl, hscale, hpimap]

/-- **`N₀ ≪ N₀ ∘ Scp s`** (diagonal spread a.c.), for `1 < 2s²`. -/
lemma stdGaussian_absolutelyContinuous_map_Scp {n : ℕ} (s : ℝ) (hs : (1 : ℝ) < 2 * s ^ 2) :
    stdGaussian (E n) ≪ (stdGaussian (E n)).map (Scp n s) := by
  classical
  have hv1 : (1 : ℝ≥0) ≠ 0 := one_ne_zero
  set c : Fin (n + 1) → ℝ := Function.update (fun _ => (1 : ℝ)) 0 s with hc_def
  set sN : Fin (n + 1) → ℝ≥0 := fun i => ⟨(c i) ^ 2, sq_nonneg (c i)⟩ with hsN
  set e : (Fin (n + 1) → ℝ) ≃ᵐ E n := MeasurableEquiv.toLp 2 (Fin (n + 1) → ℝ) with he
  have hscoe : ∀ i, ((sN i : ℝ)) = (c i) ^ 2 := fun i => rfl
  have hc0 : c 0 = s := by rw [hc_def, Function.update_self]
  have hci : ∀ b, b ≠ 0 → c b = 1 := fun b hb => by rw [hc_def, Function.update_of_ne hb]
  have hspos : ∀ i, (0 : ℝ) < (c i) ^ 2 := by
    intro i
    rcases eq_or_ne i 0 with hi | hi
    · subst hi; rw [hc0]; nlinarith [hs]
    · rw [hci i hi]; norm_num
  have hsne : ∀ i, sN i ≠ 0 := fun i => (NNReal.coe_pos.mp (by rw [hscoe i]; exact hspos i)).ne'
  have hac_i : ∀ i, gaussianReal (0 : ℝ) 1 ≪ gaussianReal (0 : ℝ) (sN i) := fun i =>
    (gaussianReal_absolutelyContinuous 0 hv1).trans (gaussianReal_absolutelyContinuous' 0 (hsne i))
  have hac : Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1)
      ≪ Measure.pi (fun i => gaussianReal (0 : ℝ) (sN i)) :=
    NearNull.pi_ac (n + 1) (fun _ => ℝ) (fun _ => gaussianReal (0 : ℝ) 1)
      (fun i => gaussianReal (0 : ℝ) (sN i)) hac_i
  have hstd : stdGaussian (E n)
      = (Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1)).map e := by
    rw [he, MeasurableEquiv.coe_toLp]; exact map_pi_eq_stdGaussian.symm
  rw [stdGaussian_map_Scp s, hstd]
  exact hac.map e.measurable

/-- **L² integrability of `d N₀ / d (N₀ ∘ Scp s)`** at the diagonal spread level, for `1 < 2s²`. -/
lemma integrable_rnDeriv_sq_stdGaussian_Scp {n : ℕ} (s : ℝ) (hs : (1 : ℝ) < 2 * s ^ 2) :
    Integrable
      (fun x => ((stdGaussian (E n)).rnDeriv ((stdGaussian (E n)).map (Scp n s)) x).toReal ^ 2)
      ((stdGaussian (E n)).map (Scp n s)) := by
  classical
  have hv1 : (1 : ℝ≥0) ≠ 0 := one_ne_zero
  set c : Fin (n + 1) → ℝ := Function.update (fun _ => (1 : ℝ)) 0 s with hc_def
  set sN : Fin (n + 1) → ℝ≥0 := fun i => ⟨(c i) ^ 2, sq_nonneg (c i)⟩ with hsN
  set e : (Fin (n + 1) → ℝ) ≃ᵐ E n := MeasurableEquiv.toLp 2 (Fin (n + 1) → ℝ) with he
  have hscoe : ∀ i, ((sN i : ℝ)) = (c i) ^ 2 := fun i => rfl
  have hc0 : c 0 = s := by rw [hc_def, Function.update_self]
  have hci : ∀ b, b ≠ 0 → c b = 1 := fun b hb => by rw [hc_def, Function.update_of_ne hb]
  have hspos : ∀ i, (0 : ℝ) < (c i) ^ 2 := by
    intro i
    rcases eq_or_ne i 0 with hi | hi
    · subst hi; rw [hc0]; nlinarith [hs]
    · rw [hci i hi]; norm_num
  have hsne : ∀ i, sN i ≠ 0 := fun i => (NNReal.coe_pos.mp (by rw [hscoe i]; exact hspos i)).ne'
  have hlt' : ∀ i, ((1 : ℝ≥0) : ℝ) < 2 * ((sN i : ℝ)) := by
    intro i
    rw [NNReal.coe_one, hscoe i]
    rcases eq_or_ne i 0 with hi | hi
    · subst hi; rw [hc0]; exact hs
    · rw [hci i hi]; norm_num
  have hac_i : ∀ i, gaussianReal (0 : ℝ) 1 ≪ gaussianReal (0 : ℝ) (sN i) := fun i =>
    (gaussianReal_absolutelyContinuous 0 hv1).trans (gaussianReal_absolutelyContinuous' 0 (hsne i))
  have hInt_i : ∀ i, Integrable
      (fun x => (((gaussianReal (0 : ℝ) 1).rnDeriv (gaussianReal (0 : ℝ) (sN i)) x).toReal) ^ 2)
      (gaussianReal (0 : ℝ) (sN i)) := fun i =>
    integrable_rnDeriv_sq_gaussianReal_var 0 hv1 (hsne i) (hlt' i)
  have hpi : Integrable
      (fun x => ((Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1)).rnDeriv
        (Measure.pi (fun i => gaussianReal (0 : ℝ) (sN i))) x).toReal ^ 2)
      (Measure.pi (fun i => gaussianReal (0 : ℝ) (sN i))) :=
    integrable_rnDeriv_sq_pi (n + 1) (fun _ => ℝ) (fun _ => gaussianReal (0 : ℝ) 1)
      (fun i => gaussianReal (0 : ℝ) (sN i)) hac_i hInt_i
  have hstd : stdGaussian (E n)
      = (Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1)).map e := by
    rw [he, MeasurableEquiv.coe_toLp]; exact map_pi_eq_stdGaussian.symm
  have hmapScp : (stdGaussian (E n)).map (Scp n s)
      = (Measure.pi (fun i => gaussianReal (0 : ℝ) (sN i))).map e := stdGaussian_map_Scp s
  have hac_std : stdGaussian (E n) ≪ (stdGaussian (E n)).map (Scp n s) :=
    stdGaussian_absolutelyContinuous_map_Scp s hs
  haveI hprobScp : IsProbabilityMeasure ((stdGaussian (E n)).map (Scp n s)) :=
    isProbabilityMeasure_map (Scp n s).continuous.measurable.aemeasurable
  have hM₂map : (stdGaussian (E n)).map e.symm
      = Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1) := by
    rw [hstd, e.map_symm_map]
  have hM₁map : ((stdGaussian (E n)).map (Scp n s)).map e.symm
      = Measure.pi (fun i => gaussianReal (0 : ℝ) (sN i)) := by
    rw [hmapScp, e.map_symm_map]
  have key : Integrable
      (fun y => (((stdGaussian (E n)).map e.symm).rnDeriv
        (((stdGaussian (E n)).map (Scp n s)).map e.symm) y).toReal ^ 2)
      (((stdGaussian (E n)).map (Scp n s)).map e.symm) := by
    rw [hM₁map, hM₂map]; exact hpi
  exact integrable_rnDeriv_sq_map (stdGaussian (E n)) ((stdGaussian (E n)).map (Scp n s))
    hac_std e.symm key

/-! ### (5) The whitening rotation carrying `u` to the axis `‖u‖·e₀` -/

/-- **Whitening rotation.** For `u ≠ 0` there is an orthonormal `R : E n ≃ₗᵢ[ℝ] E n` sending `u` to
`‖u‖·e₀`. Obtained by extending the unit vector `u/‖u‖` to an orthonormal basis (`b 0 = u/‖u‖`) and
taking `R = b.repr`, which maps `b 0` to `single 0 1 = e₀`. -/
lemma exists_rotation {n : ℕ} (u : E n) (hu : u ≠ 0) :
    ∃ R : (E n) ≃ₗᵢ[ℝ] (E n), R u = (‖u‖ : ℝ) • (e0 n) := by
  classical
  have hcard : Module.finrank ℝ (E n) = Fintype.card (Fin (n + 1)) := by simp [E]
  have hunit : ‖(‖u‖⁻¹ : ℝ) • u‖ = 1 := by
    rw [norm_smul, norm_inv, Real.norm_eq_abs, abs_of_nonneg (norm_nonneg u),
      inv_mul_cancel₀ (norm_ne_zero_iff.mpr hu)]
  have horth : Orthonormal ℝ (({0} : Set (Fin (n + 1))).restrict (fun _ => (‖u‖⁻¹ : ℝ) • u)) := by
    rw [orthonormal_iff_ite]
    intro i j
    fin_cases i; fin_cases j
    simp only [Set.restrict_apply]
    rw [real_inner_self_eq_norm_sq, hunit]
    simp
  obtain ⟨b, hb⟩ := Orthonormal.exists_orthonormalBasis_extension_of_card_eq hcard horth
  have hb0 : b 0 = (‖u‖⁻¹ : ℝ) • u := hb 0 rfl
  refine ⟨b.repr, ?_⟩
  have hrepr : b.repr (b 0) = EuclideanSpace.single (0 : Fin (n + 1)) (1 : ℝ) := b.repr_self 0
  rw [hb0, map_smul] at hrepr
  have hfin := congrArg (fun z => (‖u‖ : ℝ) • z) hrepr
  simp only [smul_smul] at hfin
  rw [mul_inv_cancel₀ (norm_ne_zero_iff.mpr hu), one_smul] at hfin
  exact hfin

/-- `χ²(μ‖μ) = 0`. -/
private lemma chi2_self {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [SigmaFinite μ] :
    NearNull.chi2 μ μ = 0 := by
  unfold NearNull.chi2
  rw [integral_congr_ae (g := fun _ => (0 : ℝ))]
  · simp
  · filter_upwards [Measure.rnDeriv_self μ] with x hx
    rw [hx]; norm_num

/-- **`N₀∘L = N₀` when `L` is an isometry** (`‖L.adjoint ξ‖ = ‖ξ‖`, i.e. `L Lᵀ = I`). The `u = 0`
degenerate case of the rank-one bump. -/
lemma stdGaussian_map_eq_self_of_isometry {n : ℕ} (L : (E n) →L[ℝ] (E n))
    (h : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖ = ‖ξ‖) :
    (stdGaussian (E n)).map L = stdGaussian (E n) := by
  have hLnorm : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖
      = ‖ContinuousLinearMap.adjoint (ContinuousLinearMap.id ℝ (E n)) ξ‖ := by
    intro ξ
    rw [ContinuousLinearMap.adjoint_id, ContinuousLinearMap.id_apply, h ξ]
  rw [stdGaussian_map_eq_of_adjoint L (ContinuousLinearMap.id ℝ (E n)) hLnorm]
  simp [Measure.map_id]

/-! ### (6) The whitening reduction, χ², a.c., and undetectability -/

/-- **The whitening reduction.** For `u ≠ 0` with `L Lᵀ = I + u uᵀ`, a rotation `eR = R` (as a
bimeasurable equivalence) carries `N₀∘L` onto the single-coordinate diagonal spread `N₀∘(Scp s)` with
`s = √(1+‖u‖²)`, while fixing `N₀`. The four consequences below (χ², a.c., L² integrability, Le Cam)
all read off this one reduction. -/
lemma rank1_reduction {n : ℕ} (u : E n) (L : (E n) →L[ℝ] (E n)) (hu : u ≠ 0)
    (hL : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2) :
    ∃ (s : ℝ) (eR : (E n) ≃ᵐ (E n)),
      (1 : ℝ) < 2 * s ^ 2 ∧ s ^ 2 = 1 + ‖u‖ ^ 2
      ∧ ((stdGaussian (E n)).map L).map eR = (stdGaussian (E n)).map (Scp n s)
      ∧ (stdGaussian (E n)).map eR = stdGaussian (E n)
      ∧ (stdGaussian (E n)).map eR.symm = stdGaussian (E n) := by
  classical
  obtain ⟨R, hRu⟩ := exists_rotation u hu
  set Rc := R.toContinuousLinearMap with hRc
  set s := Real.sqrt (1 + ‖u‖ ^ 2) with hs_def
  have hs2 : s ^ 2 = 1 + ‖u‖ ^ 2 := Real.sq_sqrt (by positivity)
  have hscond : (1 : ℝ) < 2 * s ^ 2 := by rw [hs2]; nlinarith [norm_nonneg u]
  set eR : (E n) ≃ᵐ (E n) := R.toHomeomorph.toMeasurableEquiv with heR
  have hRadj_symm : ContinuousLinearMap.adjoint Rc = R.symm.toContinuousLinearMap := by
    have h : R.symm.toContinuousLinearMap = ContinuousLinearMap.adjoint Rc := by
      rw [ContinuousLinearMap.eq_adjoint_iff]
      intro x y
      rw [show (R.symm.toContinuousLinearMap) x = R.symm x from rfl,
        show Rc y = R y from rfl, ← R.inner_map_map (R.symm x) y,
        LinearIsometryEquiv.apply_symm_apply]
    exact h.symm
  have hcov : ∀ ξ : E n,
      ‖ContinuousLinearMap.adjoint (Rc ∘L L) ξ‖ = ‖ContinuousLinearMap.adjoint (Scp n s) ξ‖ := by
    intro ξ
    have hRLsq : ‖ContinuousLinearMap.adjoint (Rc ∘L L) ξ‖ ^ 2
        = ‖ξ‖ ^ 2 + ‖u‖ ^ 2 * (ξ 0) ^ 2 := by
      have hnorm : ‖ContinuousLinearMap.adjoint Rc ξ‖ = ‖ξ‖ := by
        rw [hRadj_symm]; exact R.symm.norm_map ξ
      have hinner : (inner ℝ u (ContinuousLinearMap.adjoint Rc ξ)) = ‖u‖ * (ξ 0) := by
        rw [ContinuousLinearMap.adjoint_inner_right, show Rc u = R u from rfl, hRu,
          real_inner_smul_left, inner_e0_left]
      rw [ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_apply, hL, hnorm, hinner]
      ring
    have hScpsq : ‖ContinuousLinearMap.adjoint (Scp n s) ξ‖ ^ 2
        = ‖ξ‖ ^ 2 + ‖u‖ ^ 2 * (ξ 0) ^ 2 := by
      rw [Scp_adjoint_norm_sq, hs2]; ring
    have hsqeq : ‖ContinuousLinearMap.adjoint (Rc ∘L L) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Scp n s) ξ‖ ^ 2 := by rw [hRLsq, hScpsq]
    nlinarith [norm_nonneg (ContinuousLinearMap.adjoint (Rc ∘L L) ξ),
      norm_nonneg (ContinuousLinearMap.adjoint (Scp n s) ξ), hsqeq]
  have huniq : (stdGaussian (E n)).map (Rc ∘L L) = (stdGaussian (E n)).map (Scp n s) :=
    stdGaussian_map_eq_of_adjoint _ _ hcov
  have hmapR : (stdGaussian (E n)).map eR = stdGaussian (E n) := by
    rw [show ((stdGaussian (E n)).map eR) = (stdGaussian (E n)).map (⇑R) from rfl]
    exact stdGaussian_map R
  have hmapRsymm : (stdGaussian (E n)).map eR.symm = stdGaussian (E n) := by
    rw [show ((stdGaussian (E n)).map eR.symm) = (stdGaussian (E n)).map (⇑R.symm) from rfl]
    exact stdGaussian_map R.symm
  have hLmapR : ((stdGaussian (E n)).map L).map eR = (stdGaussian (E n)).map (Scp n s) := by
    rw [Measure.map_map eR.measurable (L.continuous.measurable),
      show (⇑eR ∘ ⇑L) = ⇑(Rc ∘L L) from rfl, huniq]
  exact ⟨s, eR, hscond, hs2, hLmapR, hmapR, hmapRsymm⟩

/-- **Absolute continuity `N₀ ≪ N₀∘L`** for a rank-one covariance bump `L Lᵀ = I + u uᵀ`. -/
lemma stdGaussian_absolutelyContinuous_map_rank1 {n : ℕ} (u : E n) (L : (E n) →L[ℝ] (E n))
    (hL : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2) :
    stdGaussian (E n) ≪ (stdGaussian (E n)).map L := by
  classical
  rcases eq_or_ne u 0 with hu | hu
  · have hiso : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖ = ‖ξ‖ := by
      intro ξ
      have hsq : ‖ContinuousLinearMap.adjoint L ξ‖ ^ 2 = ‖ξ‖ ^ 2 := by rw [hL ξ, hu]; simp
      nlinarith [norm_nonneg (ContinuousLinearMap.adjoint L ξ), norm_nonneg ξ, hsq]
    rw [stdGaussian_map_eq_self_of_isometry L hiso]
  · obtain ⟨s, eR, hscond, _, hLmapR, _, hmapRsymm⟩ := rank1_reduction u L hu hL
    have hac_LR : stdGaussian (E n) ≪ ((stdGaussian (E n)).map L).map eR := by
      rw [hLmapR]; exact stdGaussian_absolutelyContinuous_map_Scp s hscond
    have h := hac_LR.map eR.symm.measurable
    rw [hmapRsymm, Measure.map_map eR.symm.measurable eR.measurable,
      show (⇑eR.symm ∘ ⇑eR) = id from by funext z; simp, Measure.map_id] at h
    exact h

/-- **Rank-one covariance spread χ².** If `L Lᵀ = I + u uᵀ` (in quadratic-form terms
`‖L.adjoint ξ‖² = ‖ξ‖² + ⟪u,ξ⟫²`), then
`χ²(N₀ ‖ N₀∘L) = (1+‖u‖²)/√(1+2‖u‖²) − 1`. Whitening rotates `u` onto the axis `e₀`, collapsing the
rank-one bump to the single-coordinate diagonal spread `chi2_stdGaussian_scale_single`. -/
theorem chi2_stdGaussian_rank1 {n : ℕ} (u : E n) (L : (E n) →L[ℝ] (E n))
    (hL : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2) :
    NearNull.chi2 (stdGaussian (E n)) ((stdGaussian (E n)).map L)
      = (1 + ‖u‖ ^ 2) / Real.sqrt (1 + 2 * ‖u‖ ^ 2) - 1 := by
  classical
  rcases eq_or_ne u 0 with hu | hu
  · -- `u = 0`: `L` is an isometry, so `N₀∘L = N₀` and χ² = 0.
    have hiso : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖ = ‖ξ‖ := by
      intro ξ
      have hsq : ‖ContinuousLinearMap.adjoint L ξ‖ ^ 2 = ‖ξ‖ ^ 2 := by rw [hL ξ, hu]; simp
      nlinarith [norm_nonneg (ContinuousLinearMap.adjoint L ξ), norm_nonneg ξ, hsq]
    rw [stdGaussian_map_eq_self_of_isometry L hiso, chi2_self, hu]
    simp
  · -- `u ≠ 0`: whiten.
    obtain ⟨s, eR, hscond, hs2, hLmapR, hmapR, _⟩ := rank1_reduction u L hu hL
    haveI hprobL : IsProbabilityMeasure ((stdGaussian (E n)).map L) :=
      isProbabilityMeasure_map L.continuous.measurable.aemeasurable
    have hac_orig : stdGaussian (E n) ≪ (stdGaussian (E n)).map L :=
      stdGaussian_absolutelyContinuous_map_rank1 u L hL
    have hchi : NearNull.chi2 (stdGaussian (E n)) ((stdGaussian (E n)).map L)
        = NearNull.chi2 (stdGaussian (E n)) ((stdGaussian (E n)).map (Scp n s)) := by
      rw [← chi2_map_measurableEquiv (stdGaussian (E n)) ((stdGaussian (E n)).map L) hac_orig eR,
        hmapR, hLmapR]
    rw [hchi, show ((stdGaussian (E n)).map (Scp n s))
          = (stdGaussian (E n)).map (fun x : E n =>
              (WithLp.toLp 2 (Function.update (WithLp.ofLp x) 0 (s * WithLp.ofLp x 0)) : E n))
        from by rw [← Scp_eq_update],
      chi2_stdGaussian_scale_single s hscond, hs2,
      show 2 * (1 + ‖u‖ ^ 2) - 1 = 1 + 2 * ‖u‖ ^ 2 from by ring]

/-- **L² integrability of `d N₀ / d(N₀∘L)`** for a rank-one covariance bump (the Le Cam side-condition). -/
lemma integrable_rnDeriv_sq_stdGaussian_map_rank1 {n : ℕ} (u : E n) (L : (E n) →L[ℝ] (E n))
    (hL : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2) :
    Integrable
      (fun x => ((stdGaussian (E n)).rnDeriv ((stdGaussian (E n)).map L) x).toReal ^ 2)
      ((stdGaussian (E n)).map L) := by
  classical
  haveI hprobL : IsProbabilityMeasure ((stdGaussian (E n)).map L) :=
    isProbabilityMeasure_map L.continuous.measurable.aemeasurable
  rcases eq_or_ne u 0 with hu | hu
  · have hiso : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖ = ‖ξ‖ := by
      intro ξ
      have hsq : ‖ContinuousLinearMap.adjoint L ξ‖ ^ 2 = ‖ξ‖ ^ 2 := by rw [hL ξ, hu]; simp
      nlinarith [norm_nonneg (ContinuousLinearMap.adjoint L ξ), norm_nonneg ξ, hsq]
    rw [stdGaussian_map_eq_self_of_isometry L hiso]
    apply (integrable_const (1 : ℝ)).congr
    filter_upwards [Measure.rnDeriv_self (stdGaussian (E n))] with x hx
    rw [hx]; norm_num
  · obtain ⟨s, eR, hscond, _, hLmapR, hmapR, _⟩ := rank1_reduction u L hu hL
    have hac_orig : stdGaussian (E n) ≪ (stdGaussian (E n)).map L :=
      stdGaussian_absolutelyContinuous_map_rank1 u L hL
    have key : Integrable
        (fun y => (((stdGaussian (E n)).map eR).rnDeriv (((stdGaussian (E n)).map L).map eR) y).toReal
          ^ 2) (((stdGaussian (E n)).map L).map eR) := by
      rw [hmapR, hLmapR]; exact integrable_rnDeriv_sq_stdGaussian_Scp s hscond
    exact integrable_rnDeriv_sq_map (stdGaussian (E n)) ((stdGaussian (E n)).map L) hac_orig eR key

/-- **Near-null undetectability of a rank-one covariance spread.** For a rank-one bump
`L Lᵀ = I + u uᵀ`, no test separates `N₀` from `N₀∘L`: for every test `A`,
`1 − √((1+‖u‖²)/√(1+2‖u‖²) − 1) ≤ N₀.real A + (N₀∘L).real Aᶜ`. As `u → 0` the χ² `→ 0` and the left side
`→ 1`. Instantiates the Le Cam two-point χ² bound at `chi2_stdGaussian_rank1`. -/
theorem chi2_stdGaussian_rank1_undetectable {n : ℕ} (u : E n) (L : (E n) →L[ℝ] (E n))
    (hL : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint L ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2)
    {A : Set (E n)} (hA : MeasurableSet A) :
    1 - Real.sqrt ((1 + ‖u‖ ^ 2) / Real.sqrt (1 + 2 * ‖u‖ ^ 2) - 1)
      ≤ (stdGaussian (E n)).real A + ((stdGaussian (E n)).map L).real Aᶜ := by
  classical
  set P := stdGaussian (E n) with hP
  set Q := (stdGaussian (E n)).map L with hQ
  haveI hQp : IsProbabilityMeasure Q := by
    rw [hQ]; exact isProbabilityMeasure_map L.continuous.measurable.aemeasurable
  have hac : P ≪ Q := stdGaussian_absolutelyContinuous_map_rank1 u L hL
  have hsq : Integrable (fun x => ((P.rnDeriv Q x).toReal) ^ 2) Q :=
    integrable_rnDeriv_sq_stdGaussian_map_rank1 u L hL
  have hr2 : Integrable (fun x => ((P.rnDeriv Q x).toReal - 1) ^ 2) Q := by
    have h2 : Integrable (fun x => (P.rnDeriv Q x).toReal) Q := Measure.integrable_toReal_rnDeriv
    refine ((hsq.sub (h2.const_mul 2)).add (integrable_const (1 : ℝ))).congr ?_
    filter_upwards with x
    simp only [Pi.add_apply, Pi.sub_apply]
    ring
  have hchi : NearNull.chi2 P Q = (1 + ‖u‖ ^ 2) / Real.sqrt (1 + 2 * ‖u‖ ^ 2) - 1 := by
    rw [hP, hQ]; exact chi2_stdGaussian_rank1 u L hL
  have h := NearNull.lecam_two_point_chi2 hac hr2 hA
  rw [hchi] at h
  exact h

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.stdGaussian_map_eq_of_adjoint
#print axioms NearNull.chi2_stdGaussian_rank1
#print axioms NearNull.chi2_stdGaussian_rank1_undetectable
