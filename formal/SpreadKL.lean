import Mathlib
import PinskerAndNonlinear
import MultivariateGaussianSpread
import DataCovarianceSqrt

/-!
# The spread misreport's exact per-survey KL — closing `eq:nearnull-kl`'s `=` half

`PinskerAndNonlinear.lean` proved the paper's crossover plumbing (Pinsker, KL tensorization,
`½(t − log(1+t)) ≤ t²/4`, `nearnull_spread_crossover`) but left the **spread** misreport's exact
per-survey Gaussian KL as the flagged hypothesis `hKL`. This module discharges it:

**(1) The 1-D closed form** (`eq:nearnull-kl`'s `=`, one coordinate). For `v₁, v₂ ≠ 0`,
`KL(N(m,v₁) ‖ N(m,v₂)) = ½(v₁/v₂ − 1 − log(v₁/v₂))`; at `v₁ = (1+t)v₂` this is the paper's
`½(t − log(1+t))` **with the bumped/wider law as the first argument**, matching the display.
Route: the log-likelihood-ratio integral against `gaussianReal`, mirroring
`PinskerAndNonlinear.klDiv_gaussianReal` (the mean case); the llr is affine in `(x−m)²` and its
integral is the Gaussian second moment `variance_id_gaussianReal`.

**(2) Transport + tensorization.** The multivariate rank-one spread pair — `stdGaussian` vs its
pushforward by `M` with `M Mᵀ = I + u uᵀ` — reduces by the whitening rotation of
`MultivariateGaussianSpread.rank1_reduction` (the `ext_of_charFunDual` pattern) and
`PinskerAndNonlinear.klDiv_map_measurableEquiv` + `klDiv_pi` to the 1-D case with `t = ‖u‖²` in
one coordinate and `KL = 0` in the rest: `klDiv_stdGaussian_rank1`. Conjugating by the affine
whitening `y ↦ Ly⁻¹(y − Aμx)` gives the **operator spread pair** of
`NearNullOperator`/`GaussianDataLaw` (`N(Aμx, S_y)` vs the `Σ⋆ + w v vᵀ`-bumped law) the exact
per-survey KL `½(β − log(1+β))`, `β = w‖Ly⁻¹(Av)‖² = w (Av)ᵀ S_y⁻¹ (Av)` in the proved precision
(`klDiv_operator_spread`, `klDiv_operator_spread_precision`), and `klDiv_pi` gives the `n`-survey
product KL `n·½(β − log(1+β))` (`klDiv_operator_spread_pi`).

**(3) CAPSTONE — `eq:nearnull-cross` verbatim for the spread misreport.** Composing with
`half_sub_log_le_quarter_sq` (`KL ≤ t²/4`) and `summed_error_gt_half_of_klDiv_lt_half` (Pinsker +
Le Cam): for the operator spread pair over `n` iid surveys, `n·β² < 2` — i.e. `n < n⋆ = 2/β²` —
forces every test's summed type-I + type-II error above `1/2` (`operator_spread_kl_crossover`,
verbatim-`n < 2/t²` form `operator_spread_kl_crossover_verbatim`). Finally
`model_spread_kl_crossover_closed` states the whole chain **on the physical record laws**
`y = A x + ε` with the square root constructed by `DataCovarianceSqrt` and the bumped truth root
by `exists_bump_sqrt`: **no flagged hypotheses remain** — `hKL` is a theorem, and the paper's
`n < 2/t²` holds verbatim for the spread misreport with `t = w·blindGap = w (Av)ᵀ S_y⁻¹ (Av)`.

## Explicit hypotheses (flagged)
None beyond the scenario. The operator-level statements take the same satisfiable inputs as the
wave-2 modules (`hM : M Mᵀ = I + u uᵀ`, satisfiable by `exists_rank1_sqrt`); the model-law
capstone constructs everything (`Lx`, `Lε` invertible — the paper's `Σ⋆ ≻ 0`, `Γ ≻ 0`).

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each public result lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure ProbabilityTheory InformationTheory Real
open scoped ENNReal NNReal RealInnerProductSpace

namespace NearNull

/-! ### (1) The 1-D Gaussian spread KL closed form -/

/-- The Radon–Nikodym density of two same-mean Gaussians is a.e. the pdf ratio (chain rule
through `volume`; the variance-mismatch analogue of the mean-case lemma in
`PinskerAndNonlinear.lean`). -/
private lemma rnDeriv_gaussian_var_toReal_ae (m : ℝ) {v₁ v₂ : ℝ≥0} (hv₁ : v₁ ≠ 0)
    (hv₂ : v₂ ≠ 0) :
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
  have h1v := rnDeriv_gaussianReal m v₁
  have h2v := rnDeriv_gaussianReal m v₂
  filter_upwards [hchain, h1v, h2v] with x hx h1 h2
  rw [Pi.mul_apply, h1, h2] at hx
  have hpos2 : (0 : ℝ) < gaussianPDFReal m v₂ x := gaussianPDFReal_pos m v₂ x hv₂
  have hxR := congrArg ENNReal.toReal hx
  simp only [ENNReal.toReal_mul, toReal_gaussianPDF] at hxR
  rw [eq_div_iff hpos2.ne']
  exact hxR

/-- The log of the same-mean Gaussian pdf ratio: affine in `(x−m)²`. -/
private lemma log_gaussianPDFReal_div_var (m : ℝ) {v₁ v₂ : ℝ≥0} (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0)
    (x : ℝ) :
    Real.log (gaussianPDFReal m v₁ x / gaussianPDFReal m v₂ x)
      = (1 / (2 * (v₂ : ℝ)) - 1 / (2 * (v₁ : ℝ))) * (x - m) ^ 2
        - Real.log ((v₁ : ℝ) / (v₂ : ℝ)) / 2 := by
  have hv₁' : (0 : ℝ) < v₁ := by exact_mod_cast pos_of_ne_zero hv₁
  have hv₂' : (0 : ℝ) < v₂ := by exact_mod_cast pos_of_ne_zero hv₂
  have hs₁ : (0 : ℝ) < Real.sqrt (2 * π * v₁) := Real.sqrt_pos.mpr (by positivity)
  have hs₂ : (0 : ℝ) < Real.sqrt (2 * π * v₂) := Real.sqrt_pos.mpr (by positivity)
  have hp₁ : (0 : ℝ) < gaussianPDFReal m v₁ x := gaussianPDFReal_pos m v₁ x hv₁
  have hp₂ : (0 : ℝ) < gaussianPDFReal m v₂ x := gaussianPDFReal_pos m v₂ x hv₂
  rw [Real.log_div hp₁.ne' hp₂.ne']
  simp only [gaussianPDFReal]
  rw [Real.log_mul (inv_pos.mpr hs₁).ne' (Real.exp_pos _).ne',
    Real.log_mul (inv_pos.mpr hs₂).ne' (Real.exp_pos _).ne',
    Real.log_exp, Real.log_exp, Real.log_inv, Real.log_inv,
    Real.log_sqrt (by positivity), Real.log_sqrt (by positivity),
    Real.log_mul (by positivity : (0 : ℝ) < 2 * π).ne' hv₁'.ne',
    Real.log_mul (by positivity : (0 : ℝ) < 2 * π).ne' hv₂'.ne',
    Real.log_div hv₁'.ne' hv₂'.ne']
  field_simp
  ring

/-- `(x−m)²` is Gaussian-integrable (from `MemLp id 2`). -/
private lemma integrable_sq_sub_gaussian (m : ℝ) (v : ℝ≥0) :
    Integrable (fun x : ℝ => (x - m) ^ 2) (gaussianReal m v) := by
  have hid : MemLp id 2 (gaussianReal m v) := by
    have h := memLp_id_gaussianReal (μ := m) (v := v) 2
    exact_mod_cast h
  have hsub : MemLp (fun x : ℝ => x - m) 2 (gaussianReal m v) := hid.sub (memLp_const m)
  exact hsub.integrable_sq

/-- The Gaussian second central moment: `∫ (x−m)² dN(m,v) = v` (from
`variance_id_gaussianReal`). -/
private lemma integral_sq_sub_gaussian (m : ℝ) (v : ℝ≥0) :
    ∫ x, (x - m) ^ 2 ∂(gaussianReal m v) = (v : ℝ) := by
  have h := variance_id_gaussianReal (μ := m) (v := v)
  rw [variance_eq_integral aemeasurable_id] at h
  simp only [id_eq, integral_id_gaussianReal] at h
  exact h

/-- The llr integral: `∫ llr dN(m,v₁) = ½(v₁/v₂ − 1 − log(v₁/v₂))`. -/
private lemma integral_llr_var (m : ℝ) {v₁ v₂ : ℝ≥0} (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0) :
    ∫ x, ((1 / (2 * (v₂ : ℝ)) - 1 / (2 * (v₁ : ℝ))) * (x - m) ^ 2
        - Real.log ((v₁ : ℝ) / (v₂ : ℝ)) / 2) ∂(gaussianReal m v₁)
      = ((v₁ : ℝ) / (v₂ : ℝ) - 1 - Real.log ((v₁ : ℝ) / (v₂ : ℝ))) / 2 := by
  have hv₁' : (0 : ℝ) < v₁ := by exact_mod_cast pos_of_ne_zero hv₁
  have hv₂' : (0 : ℝ) < v₂ := by exact_mod_cast pos_of_ne_zero hv₂
  rw [integral_sub ((integrable_sq_sub_gaussian m v₁).const_mul _) (integrable_const _),
    integral_const_mul, integral_sq_sub_gaussian, integral_const]
  simp only [probReal_univ, one_smul]
  set ℓ := Real.log ((v₁ : ℝ) / (v₂ : ℝ)) with hℓ
  field_simp

/-- **The 1-D Gaussian spread KL closed form**: for `v₁, v₂ ≠ 0`,
`KL(N(m,v₁) ‖ N(m,v₂)) = ½(v₁/v₂ − 1 − log(v₁/v₂))`. -/
theorem klDiv_gaussianReal_var (m : ℝ) {v₁ v₂ : ℝ≥0} (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0) :
    klDiv (gaussianReal m v₁) (gaussianReal m v₂)
      = ENNReal.ofReal (((v₁ : ℝ) / (v₂ : ℝ) - 1 - Real.log ((v₁ : ℝ) / (v₂ : ℝ))) / 2) := by
  have hac : gaussianReal m v₁ ≪ gaussianReal m v₂ :=
    (gaussianReal_absolutelyContinuous m hv₁).trans
      (gaussianReal_absolutelyContinuous' m hv₂)
  have hrnP : (fun x => ((gaussianReal m v₁).rnDeriv (gaussianReal m v₂) x).toReal)
      =ᵐ[gaussianReal m v₁]
      (fun x => gaussianPDFReal m v₁ x / gaussianPDFReal m v₂ x) :=
    hac.ae_eq (rnDeriv_gaussian_var_toReal_ae m hv₁ hv₂)
  have hllr : llr (gaussianReal m v₁) (gaussianReal m v₂) =ᵐ[gaussianReal m v₁]
      fun x => (1 / (2 * (v₂ : ℝ)) - 1 / (2 * (v₁ : ℝ))) * (x - m) ^ 2
        - Real.log ((v₁ : ℝ) / (v₂ : ℝ)) / 2 := by
    filter_upwards [hrnP] with x hx
    simp only [llr_def]
    rw [hx, log_gaussianPDFReal_div_var m hv₁ hv₂ x]
  have hLint : Integrable (fun x : ℝ =>
      (1 / (2 * (v₂ : ℝ)) - 1 / (2 * (v₁ : ℝ))) * (x - m) ^ 2
        - Real.log ((v₁ : ℝ) / (v₂ : ℝ)) / 2) (gaussianReal m v₁) :=
    ((integrable_sq_sub_gaussian m v₁).const_mul _).sub (integrable_const _)
  have hint : Integrable (llr (gaussianReal m v₁) (gaussianReal m v₂)) (gaussianReal m v₁) :=
    hLint.congr hllr.symm
  rw [klDiv_of_ac_of_integrable hac hint]
  have hI : ∫ x, llr (gaussianReal m v₁) (gaussianReal m v₂) x ∂(gaussianReal m v₁)
      = ((v₁ : ℝ) / (v₂ : ℝ) - 1 - Real.log ((v₁ : ℝ) / (v₂ : ℝ))) / 2 := by
    rw [integral_congr_ae hllr]
    exact integral_llr_var m hv₁ hv₂
  rw [hI, probReal_univ, probReal_univ]
  norm_num

/-- **The paper's display, verbatim direction** (`eq:nearnull-kl`, one coordinate): with the
bumped/wider law FIRST, `KL(N(m,(1+t)v) ‖ N(m,v)) = ½(t − log(1+t))` for `t > −1`, `v ≠ 0`. -/
theorem klDiv_gaussianReal_spread (m : ℝ) {t : ℝ} (ht : -1 < t) {v v' : ℝ≥0} (hv : v ≠ 0)
    (hv' : (v' : ℝ) = (1 + t) * (v : ℝ)) :
    klDiv (gaussianReal m v') (gaussianReal m v)
      = ENNReal.ofReal ((t - Real.log (1 + t)) / 2) := by
  have h1t : (0 : ℝ) < 1 + t := by linarith
  have hvpos : (0 : ℝ) < v := by exact_mod_cast pos_of_ne_zero hv
  have hv'pos : (0 : ℝ) < v' := by rw [hv']; positivity
  have hv'ne : v' ≠ 0 := by
    have h : (v' : ℝ) ≠ 0 := hv'pos.ne'
    exact_mod_cast h
  rw [klDiv_gaussianReal_var m hv'ne hv]
  congr 1
  rw [hv', mul_div_cancel_right₀ _ hvpos.ne']
  ring

/-- The `toReal` reading of the display: `KL.toReal = ½(t − log(1+t))` (the KL is finite and the
closed form is nonnegative since `log(1+t) ≤ t`). -/
theorem toReal_klDiv_gaussianReal_spread (m : ℝ) {t : ℝ} (ht : -1 < t) {v v' : ℝ≥0}
    (hv : v ≠ 0) (hv' : (v' : ℝ) = (1 + t) * (v : ℝ)) :
    (klDiv (gaussianReal m v') (gaussianReal m v)).toReal = (t - Real.log (1 + t)) / 2 := by
  have hlog : Real.log (1 + t) ≤ t := by
    have h := Real.log_le_sub_one_of_pos (show (0 : ℝ) < 1 + t by linarith)
    linarith
  rw [klDiv_gaussianReal_spread m ht hv hv', ENNReal.toReal_ofReal (by linarith)]

/-! ### (2) The multivariate rank-one spread KL: rotation + tensorization -/

/-- **The rank-one spread KL on the standard Gaussian.** For `M Mᵀ = I + u uᵀ` (variationally
`‖Mᵀξ‖² = ‖ξ‖² + ⟪u,ξ⟫²`), with the bumped law FIRST:
`KL(N₀∘M ‖ N₀) = ½(‖u‖² − log(1+‖u‖²))`. The whitening rotation of
`MultivariateGaussianSpread.rank1_reduction` carries `u` to an axis; `klDiv_map_measurableEquiv`
and `klDiv_pi` reduce to the 1-D closed form in coordinate `0` and `KL = 0` in the rest. -/
theorem klDiv_stdGaussian_rank1 {n : ℕ} (u : E n) (M : (E n) →L[ℝ] (E n))
    (hM : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2) :
    klDiv ((stdGaussian (E n)).map M) (stdGaussian (E n))
      = ENNReal.ofReal ((‖u‖ ^ 2 - Real.log (1 + ‖u‖ ^ 2)) / 2) := by
  classical
  rcases eq_or_ne u 0 with hu | hu
  · -- `u = 0`: `M` is an isometry, both laws coincide, both sides vanish.
    have hiso : ∀ ξ : E n, ‖ContinuousLinearMap.adjoint M ξ‖ = ‖ξ‖ := by
      intro ξ
      have hsq : ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 := by rw [hM ξ, hu]; simp
      nlinarith [norm_nonneg (ContinuousLinearMap.adjoint M ξ), norm_nonneg ξ, hsq]
    rw [stdGaussian_map_eq_self_of_isometry M hiso, klDiv_self, hu]
    simp
  · obtain ⟨s, eR, hscond, hs2, hLmapR, hmapR, _⟩ := rank1_reduction u M hu hM
    haveI hprobM : IsProbabilityMeasure ((stdGaussian (E n)).map M) :=
      isProbabilityMeasure_map M.continuous.measurable.aemeasurable
    -- rotate: KL(N₀∘M ‖ N₀) = KL(N₀∘Scp(s) ‖ N₀)
    have hswap : klDiv ((stdGaussian (E n)).map M) (stdGaussian (E n))
        = klDiv ((stdGaussian (E n)).map (Scp n s)) (stdGaussian (E n)) := by
      rw [← klDiv_map_measurableEquiv ((stdGaussian (E n)).map M) (stdGaussian (E n)) eR,
        hLmapR, hmapR]
    -- decompose both sides as pi products through `toLp`
    have hstd : stdGaussian (E n)
        = (Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1)).map
            (MeasurableEquiv.toLp 2 (Fin (n + 1) → ℝ)) := by
      rw [MeasurableEquiv.coe_toLp]
      exact map_pi_eq_stdGaussian.symm
    have hpiKL : klDiv ((stdGaussian (E n)).map (Scp n s)) (stdGaussian (E n))
        = klDiv
            (Measure.pi (fun i => gaussianReal (0 : ℝ)
              (⟨(Function.update (fun _ => (1 : ℝ)) 0 s i) ^ 2, sq_nonneg _⟩ : ℝ≥0)))
            (Measure.pi (fun _ : Fin (n + 1) => gaussianReal (0 : ℝ) 1)) := by
      rw [stdGaussian_map_Scp s, hstd, klDiv_map_measurableEquiv]
    have hklpi := klDiv_pi (n + 1) (fun _ => ℝ)
      (fun i => gaussianReal (0 : ℝ)
        (⟨(Function.update (fun _ => (1 : ℝ)) 0 s i) ^ 2, sq_nonneg _⟩ : ℝ≥0))
      (fun _ => gaussianReal (0 : ℝ) 1)
    -- coordinatewise: the bumped coordinate `0` carries the closed form, the rest vanish
    have hterm : ∀ i : Fin (n + 1),
        klDiv (gaussianReal (0 : ℝ)
            (⟨(Function.update (fun _ => (1 : ℝ)) 0 s i) ^ 2, sq_nonneg _⟩ : ℝ≥0))
          (gaussianReal (0 : ℝ) 1)
          = if i = 0 then ENNReal.ofReal ((‖u‖ ^ 2 - Real.log (1 + ‖u‖ ^ 2)) / 2) else 0 := by
      intro i
      rcases eq_or_ne i 0 with hi | hi
      · subst hi
        rw [if_pos rfl]
        have ht' : (-1 : ℝ) < ‖u‖ ^ 2 := lt_of_lt_of_le (by norm_num) (sq_nonneg _)
        have hv' : ((⟨(Function.update (fun _ : Fin (n + 1) => (1 : ℝ)) 0 s 0) ^ 2,
            sq_nonneg _⟩ : ℝ≥0) : ℝ) = (1 + ‖u‖ ^ 2) * ((1 : ℝ≥0) : ℝ) := by
          simp only [Function.update_self, NNReal.coe_one, mul_one]
          exact hs2
        exact klDiv_gaussianReal_spread 0 ht' one_ne_zero hv'
      · rw [if_neg hi]
        have hval : (⟨(Function.update (fun _ : Fin (n + 1) => (1 : ℝ)) 0 s i) ^ 2,
            sq_nonneg _⟩ : ℝ≥0) = 1 := by
          simp [Function.update_of_ne hi]
        rw [hval]
        exact klDiv_self _
    rw [hswap, hpiKL, hklpi]
    simp_rw [hterm]
    simp

/-- **The operator spread pair's exact per-survey KL** (`eq:nearnull-kl`'s `=`, at the data law).
For the pair of `NearNullOperator`/`GaussianDataLaw` — base `N(Aμx, S_y) = N₀∘(x ↦ Aμx + Ly x)`
vs the `Σ⋆ + w v vᵀ`-bumped law `N₀∘(x ↦ Aμx + Ly (M x))`, `M Mᵀ = I + u uᵀ`,
`u = √w·Ly⁻¹(Av)` — with the bumped law FIRST:
`KL = ½(β − log(1+β))`, `β = w‖Ly⁻¹(Av)‖²`. -/
theorem klDiv_operator_spread {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2) :
    klDiv ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
      = ENNReal.ofReal ((w * ‖Ly.symm (Aop v)‖ ^ 2
          - Real.log (1 + w * ‖Ly.symm (Aop v)‖ ^ 2)) / 2) := by
  classical
  set e : (E m) ≃ᵐ (E m) :=
    ((Homeomorph.subRight (Aop μx)).trans Ly.symm.toHomeomorph).toMeasurableEquiv with he
  have he_apply : ∀ y : E m, e y = Ly.symm (y - Aop μx) := fun y => rfl
  have hef : (⇑e ∘ fun x : E m => Aop μx + Ly x) = id := by
    funext x
    simp only [Function.comp_apply, he_apply, id_eq]
    rw [show Aop μx + Ly x - Aop μx = Ly x from by abel, ContinuousLinearEquiv.symm_apply_apply]
  have hefM : (⇑e ∘ fun x : E m => Aop μx + Ly (M x)) = (M : (E m) → (E m)) := by
    funext x
    simp only [Function.comp_apply, he_apply]
    rw [show Aop μx + Ly (M x) - Aop μx = Ly (M x) from by abel,
      ContinuousLinearEquiv.symm_apply_apply]
  haveI h₁ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => Aop μx + Ly (M x))) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI h₂ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  have hklinv := klDiv_map_measurableEquiv
    ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)))
    ((stdGaussian (E m)).map (fun x => Aop μx + Ly x)) e
  rw [← hklinv,
    Measure.map_map e.measurable
      (show Measurable (fun x : E m => Aop μx + Ly (M x)) by fun_prop),
    Measure.map_map e.measurable
      (show Measurable (fun x : E m => Aop μx + Ly x) by fun_prop),
    hef, hefM, Measure.map_id]
  have hnorm : ‖Real.sqrt w • Ly.symm (Aop v)‖ ^ 2 = w * ‖Ly.symm (Aop v)‖ ^ 2 := by
    rw [norm_smul, mul_pow, Real.norm_eq_abs, sq_abs, Real.sq_sqrt hw]
  rw [klDiv_stdGaussian_rank1 (Real.sqrt w • Ly.symm (Aop v)) M hM, hnorm]

/-- The per-survey KL in the manuscript's precision variable — `eq:nearnull-kl` verbatim:
`KL = ½(t − log(1+t))`, `t = w·(Av)ᵀ S_y⁻¹ (Av)` with `S_y⁻¹` the proved two-sided inverse of
`Ly Lyᵀ`. -/
theorem klDiv_operator_spread_precision {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2) :
    klDiv ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
      = ENNReal.ofReal ((w * inner ℝ (Aop v) (dataPrecision Ly (Aop v))
          - Real.log (1 + w * inner ℝ (Aop v) (dataPrecision Ly (Aop v)))) / 2) := by
  rw [klDiv_operator_spread Aop μx v hw Ly M hM, norm_symm_sq_eq_inner_dataPrecision]

/-- **The `n`-survey operator spread KL**: iid surveys sum the per-survey divergences,
`KL = n·½(β − log(1+β))`. -/
theorem klDiv_operator_spread_pi {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2) (n : ℕ) :
    klDiv (Measure.pi fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)))
        (Measure.pi fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx + Ly x))
      = ENNReal.ofReal ((n : ℝ) * ((w * ‖Ly.symm (Aop v)‖ ^ 2
          - Real.log (1 + w * ‖Ly.symm (Aop v)‖ ^ 2)) / 2)) := by
  haveI h₁ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => Aop μx + Ly (M x))) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI h₂ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  rw [klDiv_pi n (fun _ => E m)
    (fun _ => (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)))
    (fun _ => (stdGaussian (E m)).map (fun x => Aop μx + Ly x))]
  simp only [klDiv_operator_spread Aop μx v hw Ly M hM]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    ← ENNReal.ofReal_natCast n, ← ENNReal.ofReal_mul (Nat.cast_nonneg n)]

/-! ### (3) CAPSTONE: `eq:nearnull-cross` verbatim for the spread misreport -/

/-- **The spread crossover with the exact KL — `eq:nearnull-cross`, no flagged hypotheses.**
For the operator spread pair over `n` iid surveys, `n·β² < 2` (i.e. `n < n⋆ = 2/β²`,
`β = w‖Ly⁻¹(Av)‖² = w (Av)ᵀS_y⁻¹(Av)`) forces every test's summed type-I + type-II error
strictly above `1/2`. Chain: exact per-survey `KL = ½(β − log(1+β))` (this module) `≤ β²/4`
(`half_sub_log_le_quarter_sq`), tensorized, fed to Pinsker + Le Cam
(`summed_error_gt_half_of_klDiv_lt_half`). -/
theorem operator_spread_kl_crossover {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    (n : ℕ) (hn : (n : ℝ) * (w * ‖Ly.symm (Aop v)‖ ^ 2) ^ 2 < 2)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real Aᶜ := by
  haveI h₁ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => Aop μx + Ly (M x))) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI h₂ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  have hkl := klDiv_operator_spread_pi Aop μx v hw Ly M hM n
  set β := w * ‖Ly.symm (Aop v)‖ ^ 2 with hβdef
  have hβ0 : 0 ≤ β := mul_nonneg hw (sq_nonneg _)
  have hlog : Real.log (1 + β) ≤ β := by
    have h := Real.log_le_sub_one_of_pos (show (0 : ℝ) < 1 + β by linarith)
    linarith
  apply summed_error_gt_half_of_klDiv_lt_half
  · rw [hkl]; exact ENNReal.ofReal_ne_top
  · rw [hkl, ENNReal.toReal_ofReal (mul_nonneg (Nat.cast_nonneg n)
      (by linarith : (0 : ℝ) ≤ (β - Real.log (1 + β)) / 2))]
    have hq := half_sub_log_le_quarter_sq hβ0
    have h2 : (n : ℝ) * ((β - Real.log (1 + β)) / 2) ≤ (n : ℝ) * (β ^ 2 / 4) :=
      mul_le_mul_of_nonneg_left hq (Nat.cast_nonneg n)
    have h3 : (n : ℝ) * (β ^ 2 / 4) < 1 / 2 := by nlinarith
    linarith
  · exact hA

/-- The crossover in the paper's verbatim form `n < n⋆ = 2/t²` (for `t > 0`). -/
theorem operator_spread_kl_crossover_verbatim {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    (ht : 0 < w * ‖Ly.symm (Aop v)‖ ^ 2)
    (n : ℕ) (hn : (n : ℝ) < 2 / (w * ‖Ly.symm (Aop v)‖ ^ 2) ^ 2)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real Aᶜ := by
  apply operator_spread_kl_crossover Aop μx v hw Ly M hM n ?_ hA
  have ht2 : (0 : ℝ) < (w * ‖Ly.symm (Aop v)‖ ^ 2) ^ 2 := by positivity
  calc (n : ℝ) * (w * ‖Ly.symm (Aop v)‖ ^ 2) ^ 2
      < (2 / (w * ‖Ly.symm (Aop v)‖ ^ 2) ^ 2) * (w * ‖Ly.symm (Aop v)‖ ^ 2) ^ 2 :=
        mul_lt_mul_of_pos_right hn ht2
    _ = 2 := div_mul_cancel₀ 2 ht2.ne'

/-- **CAPSTONE on the physical record laws, fully closed.** For the record `y = A x + ε` of a
Gaussian truth with invertible root `Lx` under independent invertible Gaussian noise `Lε`, there
exists a root `Lx'` of the bumped truth covariance `Σ⋆ + w v vᵀ` such that, with
`β = w·blindGap = w (Av)ᵀ S_y⁻¹ (Av)` in the constructed data covariance (`DataCovarianceSqrt`):
(i) the per-survey KL of the two record laws (bumped FIRST) is exactly `½(β − log(1+β))` —
`eq:nearnull-kl`'s `=`, no flagged hypotheses; (ii) `n` iid surveys have KL `n·½(β − log(1+β))`;
(iii) whenever `n·β² < 2` — the paper's `n < n⋆ = 2/t²`, `eq:nearnull-cross` verbatim — every
test on `n` surveys has summed type-I + type-II error strictly above `1/2`. -/
theorem model_spread_kl_crossover_closed {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w)
    (Lx : EuclideanSpace ℝ (Fin d) ≃L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) :
    ∃ Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d),
      (∀ η : EuclideanSpace ℝ (Fin d),
        ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
          = ‖ContinuousLinearMap.adjoint
              (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) η‖ ^ 2
            + w * (inner ℝ v η) ^ 2) ∧
      klDiv (modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m)))
          (modelLaw Aop μx (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
            (↑Lε : (E m) →L[ℝ] (E m)))
        = ENNReal.ofReal ((w * blindGap Aop (↑Lx) Lε v
            - Real.log (1 + w * blindGap Aop (↑Lx) Lε v)) / 2) ∧
      (∀ n : ℕ,
        klDiv (Measure.pi fun _ : Fin n => modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m)))
            (Measure.pi fun _ : Fin n =>
              modelLaw Aop μx (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
                (↑Lε : (E m) →L[ℝ] (E m)))
          = ENNReal.ofReal ((n : ℝ) * ((w * blindGap Aop (↑Lx) Lε v
              - Real.log (1 + w * blindGap Aop (↑Lx) Lε v)) / 2))) ∧
      (∀ n : ℕ, (n : ℝ) * (w * blindGap Aop (↑Lx) Lε v) ^ 2 < 2 →
        ∀ {A : Set (Fin n → E m)}, MeasurableSet A →
        1 / 2 < (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))).real A
            + (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx
                  (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
                  (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ) := by
  obtain ⟨Lx', hbump⟩ := exists_bump_sqrt Lx v hw
  obtain ⟨M, hM⟩ := exists_rank1_sqrt
    (Real.sqrt w •
      (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
        EuclideanSpace ℝ (Fin d))) Lε).symm (Aop v))
  have hcov := dataCovSqrt_spec
    (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))) Lε
  have hβ := blindGap_eq_norm_sq Aop
    (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε v
  have h₁ : modelLaw Aop μx (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
        (↑Lε : (E m) →L[ℝ] (E m))
      = (stdGaussian (E m)).map (fun x => Aop μx
          + (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
              EuclideanSpace ℝ (Fin d))) Lε) x) :=
    modelLaw_repr Aop μx (↑Lx) (↑Lε)
      (↑(dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
        EuclideanSpace ℝ (Fin d))) Lε)) hcov
  have h₂ : modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))
      = (stdGaussian (E m)).map (fun x => Aop μx
          + (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
              EuclideanSpace ℝ (Fin d))) Lε) (M x)) :=
    modelLaw_repr Aop μx Lx' (↑Lε)
      ((↑(dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
        EuclideanSpace ℝ (Fin d))) Lε) : (E m) →L[ℝ] (E m)) ∘L M)
      (bump_cov_transfer Aop v hw (↑Lx) Lx' (↑Lε)
        (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
          EuclideanSpace ℝ (Fin d))) Lε) M hcov hbump hM)
  refine ⟨Lx', hbump, ?_, ?_, ?_⟩
  · rw [h₁, h₂, hβ]
    exact klDiv_operator_spread Aop μx v hw
      (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
        EuclideanSpace ℝ (Fin d))) Lε) M hM
  · intro n
    rw [show (fun _ : Fin n => modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m)))
        = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx
            + (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
                EuclideanSpace ℝ (Fin d))) Lε) (M x))) from
      funext fun _ => h₂,
      show (fun _ : Fin n =>
          modelLaw Aop μx (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
            (↑Lε : (E m) →L[ℝ] (E m)))
        = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx
            + (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
                EuclideanSpace ℝ (Fin d))) Lε) x)) from
      funext fun _ => h₁, hβ]
    exact klDiv_operator_spread_pi Aop μx v hw
      (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
        EuclideanSpace ℝ (Fin d))) Lε) M hM n
  · intro n hn A hA
    rw [hβ] at hn
    rw [show (fun _ : Fin n => modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m)))
        = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx
            + (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
                EuclideanSpace ℝ (Fin d))) Lε) (M x))) from
      funext fun _ => h₂,
      show (fun _ : Fin n =>
          modelLaw Aop μx (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
            (↑Lε : (E m) →L[ℝ] (E m)))
        = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx
            + (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
                EuclideanSpace ℝ (Fin d))) Lε) x)) from
      funext fun _ => h₁]
    exact operator_spread_kl_crossover Aop μx v hw
      (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
        EuclideanSpace ℝ (Fin d))) Lε) M hM n hn hA

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.klDiv_gaussianReal_var
#print axioms NearNull.klDiv_gaussianReal_spread
#print axioms NearNull.toReal_klDiv_gaussianReal_spread
#print axioms NearNull.klDiv_stdGaussian_rank1
#print axioms NearNull.klDiv_operator_spread
#print axioms NearNull.klDiv_operator_spread_precision
#print axioms NearNull.klDiv_operator_spread_pi
#print axioms NearNull.operator_spread_kl_crossover
#print axioms NearNull.operator_spread_kl_crossover_verbatim
#print axioms NearNull.model_spread_kl_crossover_closed
