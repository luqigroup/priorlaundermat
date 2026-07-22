import Mathlib
import PinskerAndNonlinear
import MultivariateGaussianSpread
import DataCovarianceSqrt

/-!
# The mean misreport's exact per-survey KL — the Pinsker/KL route at the OPERATOR record law

`SpreadKL.lean` closed the Pinsker/KL route for the **spread** misreport at the operator record
law; the mean misreport's KL route stopped at the 1-D/whitened statements of
`PinskerAndNonlinear.lean` (`klDiv_gaussianReal`, `klDiv_pi_gaussian_mean_shift`), never reaching
the physical record laws. This module closes that gap:

**(1) The isotropic shift KL and its affine transport.** `klDiv_stdGaussian_shift` lifts the 1-D
mean-shift closed form to `KL(N₀(·+δ) ‖ N₀) = ‖δ‖²/2` on `EuclideanSpace ℝ (Fin n)` — transport
through `toLp` (`map_pi_eq_stdGaussian`, `Measure.pi_map_pi`) + `klDiv_map_measurableEquiv` +
`klDiv_pi_gaussian_mean_shift`, the same whitening pattern as
`MultivariateGaussianWhitening.chi2_stdGaussian_add`. `klDiv_stdGaussian_affine` conjugates by the
affine whitening `y ↦ L⁻¹(y − μ₂)` (the `Homeomorph.subRight` + `L.symm` equivalence, as in
`SpreadKL.klDiv_operator_spread`): for a common invertible root `L`,
`KL(N(μ₁, LLᵀ) ‖ N(μ₂, LLᵀ)) = ‖L⁻¹(μ₁−μ₂)‖²/2`.

**(2) The operator mean-misreport pair.** The record pair of `NearNullOperator`/`GaussianDataLaw`
for a `w·v` mean misreport — `N(Aμx + w·Av, S_y)` vs `N(Aμx, S_y)`, the SAME covariance root
`Ly` — has exact per-survey `KL = β/2` with `β = w²‖Ly⁻¹(Av)‖² = w²·(Av)ᵀS_y⁻¹(Av)`
(`klDiv_operator_mean`, precision form `klDiv_operator_mean_precision` via
`norm_symm_sq_eq_inner_dataPrecision`); `klDiv_pi` tensorizes `n` iid surveys to `KL = n·β/2`
(`klDiv_operator_mean_pi`).

**(3) The crossover, and the honest constant.** Feeding the exact `n`-survey KL to Pinsker + Le
Cam (`summed_error_gt_half_of_klDiv_lt_half`): `n·β < 1` — i.e. `n < n⋆ = 1/β` — forces every
test's summed type-I + type-II error strictly above `1/2` (`operator_mean_kl_crossover`, verbatim
`n < 1/β` form `operator_mean_kl_crossover_verbatim`). **Constant note (as flagged in
`PinskerAndNonlinear.lean`):** the per-survey mean KL is exactly `β/2`, so Pinsker crosses at
`n·β/2 < 1/2 ⟺ n·β < 1`. The spread case's `n < 2/t²` does NOT transfer: `n·β < 2` only bounds
the summed error below by `1 − √(1/2) < 1/2`, and the claim "`n·β < 2 ⟹` error `> 1/2`" is in
fact FALSE (1-D, `n = 1`, `β` near `2`: the likelihood-ratio test at the midpoint has summed
error `2Φ(−√β/2) < 1/2`). `eq:nearnull-cross`'s mean case therefore carries `n⋆ = 1/β`.

**(4) CAPSTONE on the physical record laws** — hypothesis-free (`Lx` an ARBITRARY continuous
linear truth root, `Lε` invertible; no bump root, no existential — the mean pair shares one
covariance): with `β = w²·blindGap = w²·(Av)ᵀS_y⁻¹(Av)` in the constructed data covariance
(`DataCovarianceSqrt`), (i) `model_mean_kl_closed` — the per-survey KL of the two record laws
`y = A x + ε` is exactly `β/2`; (ii) `model_mean_kl_pi_closed` — `n` iid surveys have KL `n·β/2`;
(iii) `model_mean_kl_crossover_closed` — `n·β < 1` forces every test's summed error above `1/2`.
On the blind subspace (`Av = 0`) the KL is `0` for every `n`: permanent undetectability.

## Explicit hypotheses (flagged)
None beyond the scenario. `w` is an arbitrary real (the exponent is `w² ≥ 0`); the capstones take
`Lx` an arbitrary continuous linear map and `Lε` invertible (the paper's `Γ ≻ 0`).

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each public result lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure ProbabilityTheory InformationTheory Real
open scoped ENNReal NNReal RealInnerProductSpace

namespace NearNull

/-! ### (1) The isotropic mean-shift KL and its affine transport -/

/-- **The isotropic multivariate mean-shift KL**: `KL(N₀(·+δ) ‖ N₀) = ‖δ‖²/2` on
`EuclideanSpace ℝ (Fin n)`. Transport through `toLp` to the product of shifted 1-D Gaussians
(the `chi2_stdGaussian_add` pattern), then `klDiv_map_measurableEquiv` +
`klDiv_pi_gaussian_mean_shift`. -/
theorem klDiv_stdGaussian_shift {n : ℕ} (δ : EuclideanSpace ℝ (Fin n)) :
    klDiv ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ))
        (stdGaussian (EuclideanSpace ℝ (Fin n)))
      = ENNReal.ofReal (‖δ‖ ^ 2 / 2) := by
  classical
  set e : (Fin n → ℝ) ≃ᵐ EuclideanSpace ℝ (Fin n) := MeasurableEquiv.toLp 2 (Fin n → ℝ) with he
  set δ' : Fin n → ℝ := WithLp.ofLp δ with hδ'
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
  have hkl : klDiv ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + δ))
      (stdGaussian (EuclideanSpace ℝ (Fin n)))
      = klDiv (Measure.pi (fun i => gaussianReal (δ' i) 1))
          (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)) := by
    rw [hshift, hpimap, hstd]
    exact klDiv_map_measurableEquiv _ _ e
  have h := klDiv_pi_gaussian_mean_shift δ' (fun _ => (0 : ℝ))
  simp only [sub_zero] at h
  have hnorm : (∑ i : Fin n, (δ' i) ^ 2) = ‖δ‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq]
  rw [hkl, h, hnorm]

/-- **The general-covariance mean-shift KL** (the KL analogue of `chi2_stdGaussian_affine`): for
a common invertible covariance root `L` and means `μ₁, μ₂`,
`KL(N₀(μ₁+L·) ‖ N₀(μ₂+L·)) = ‖L⁻¹(μ₁−μ₂)‖²/2` — the squared Mahalanobis mean gap over `2`.
Conjugation by the affine whitening `y ↦ L⁻¹(y − μ₂)` + `klDiv_map_measurableEquiv`. -/
theorem klDiv_stdGaussian_affine {n : ℕ} (μ₁ μ₂ : EuclideanSpace ℝ (Fin n))
    (L : EuclideanSpace ℝ (Fin n) ≃L[ℝ] EuclideanSpace ℝ (Fin n)) :
    klDiv ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₁ + L x))
        ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x))
      = ENNReal.ofReal (‖L.symm (μ₁ - μ₂)‖ ^ 2 / 2) := by
  classical
  set e : EuclideanSpace ℝ (Fin n) ≃ᵐ EuclideanSpace ℝ (Fin n) :=
    ((Homeomorph.subRight μ₂).trans L.symm.toHomeomorph).toMeasurableEquiv with he
  have he_apply : ∀ y : EuclideanSpace ℝ (Fin n), e y = L.symm (y - μ₂) := fun y => rfl
  have hef₁ : (⇑e ∘ fun x : EuclideanSpace ℝ (Fin n) => μ₁ + L x)
      = (· + L.symm (μ₁ - μ₂)) := by
    funext x
    simp only [Function.comp_apply, he_apply]
    rw [show μ₁ + L x - μ₂ = L x + (μ₁ - μ₂) from by abel, map_add,
      ContinuousLinearEquiv.symm_apply_apply]
  have hef₂ : (⇑e ∘ fun x : EuclideanSpace ℝ (Fin n) => μ₂ + L x) = id := by
    funext x
    simp only [Function.comp_apply, he_apply, id_eq]
    rw [show μ₂ + L x - μ₂ = L x from by abel, ContinuousLinearEquiv.symm_apply_apply]
  haveI h₁ : IsProbabilityMeasure
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map
        (fun x : EuclideanSpace ℝ (Fin n) => μ₁ + L x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI h₂ : IsProbabilityMeasure
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map
        (fun x : EuclideanSpace ℝ (Fin n) => μ₂ + L x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  rw [← klDiv_map_measurableEquiv
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₁ + L x))
      ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => μ₂ + L x)) e,
    Measure.map_map e.measurable
      (show Measurable (fun x : EuclideanSpace ℝ (Fin n) => μ₁ + L x) by fun_prop),
    Measure.map_map e.measurable
      (show Measurable (fun x : EuclideanSpace ℝ (Fin n) => μ₂ + L x) by fun_prop),
    hef₁, hef₂, Measure.map_id]
  exact klDiv_stdGaussian_shift (L.symm (μ₁ - μ₂))

/-! ### (2) The operator mean-misreport pair's exact per-survey KL -/

/-- **The operator mean-misreport pair's exact per-survey KL** (`eq:nearnull-kl`'s mean case at
the data law). The record pair of `NearNullOperator`/`GaussianDataLaw` — `N(Aμx + w·Av, S_y)` vs
`N(Aμx, S_y)`, a mean shift `w·Av` through the SAME covariance root `Ly` — has
`KL = β/2`, `β = w²‖Ly⁻¹(Av)‖²`. -/
theorem klDiv_operator_mean {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) :
    klDiv ((stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
      = ENNReal.ofReal ((w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) / 2) := by
  rw [klDiv_stdGaussian_affine (Aop μx + w • Aop v) (Aop μx) Ly,
    show Aop μx + w • Aop v - Aop μx = w • Aop v from by abel, map_smul, norm_smul, mul_pow,
    Real.norm_eq_abs, sq_abs]

/-- The per-survey mean KL in the manuscript's precision variable:
`KL = β/2`, `β = w²·(Av)ᵀ S_y⁻¹ (Av)` with `S_y⁻¹` the proved two-sided inverse of `Ly Lyᵀ`. -/
theorem klDiv_operator_mean_precision {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) :
    klDiv ((stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
      = ENNReal.ofReal ((w ^ 2 * inner ℝ (Aop v) (dataPrecision Ly (Aop v))) / 2) := by
  rw [klDiv_operator_mean Aop μx v w Ly, norm_symm_sq_eq_inner_dataPrecision]

/-- **The `n`-survey operator mean KL**: iid surveys sum the per-survey divergences,
`KL = n·β/2`. -/
theorem klDiv_operator_mean_pi {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) (n : ℕ) :
    klDiv (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x))
        (Measure.pi fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx + Ly x))
      = ENNReal.ofReal ((n : ℝ) * ((w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) / 2)) := by
  haveI h₁ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => (Aop μx + w • Aop v) + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI h₂ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  rw [klDiv_pi n (fun _ => E m)
    (fun _ => (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x))
    (fun _ => (stdGaussian (E m)).map (fun x => Aop μx + Ly x))]
  simp only [klDiv_operator_mean Aop μx v w Ly]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    ← ENNReal.ofReal_natCast n, ← ENNReal.ofReal_mul (Nat.cast_nonneg n)]

/-! ### (3) The mean crossover via Pinsker, with the honest constant `n⋆ = 1/β` -/

/-- **The mean crossover with the exact KL** (`eq:nearnull-cross`, mean case). For the operator
mean-misreport pair over `n` iid surveys, `n·β < 1` (i.e. `n < n⋆ = 1/β`,
`β = w²‖Ly⁻¹(Av)‖² = w²·(Av)ᵀS_y⁻¹(Av)`) forces every test's summed type-I + type-II error
strictly above `1/2`. Chain: exact `n`-survey `KL = n·β/2 < 1/2`, fed to Pinsker + Le Cam
(`summed_error_gt_half_of_klDiv_lt_half`). The constant is `1/β`, NOT the spread case's `2/β²`:
see the module docstring (the `n·β < 2` variant is false). -/
theorem operator_mean_kl_crossover {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m))
    (n : ℕ) (hn : (n : ℝ) * (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) < 1)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real Aᶜ := by
  haveI h₁ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => (Aop μx + w • Aop v) + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI h₂ : IsProbabilityMeasure
      ((stdGaussian (E m)).map (fun x : E m => Aop μx + Ly x)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  have hkl := klDiv_operator_mean_pi Aop μx v w Ly n
  have hβ0 : (0 : ℝ) ≤ w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2 := by positivity
  set β := w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2 with hβdef
  apply summed_error_gt_half_of_klDiv_lt_half
  · rw [hkl]; exact ENNReal.ofReal_ne_top
  · rw [hkl, ENNReal.toReal_ofReal
      (mul_nonneg (Nat.cast_nonneg n) (div_nonneg hβ0 (by norm_num)))]
    have heq : (n : ℝ) * (β / 2) = ((n : ℝ) * β) / 2 := by ring
    rw [heq]
    linarith
  · exact hA

/-- The crossover in the verbatim form `n < n⋆ = 1/β` (for `β > 0`). -/
theorem operator_mean_kl_crossover_verbatim {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m))
    (ht : 0 < w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2)
    (n : ℕ) (hn : (n : ℝ) < 1 / (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2))
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real Aᶜ := by
  apply operator_mean_kl_crossover Aop μx v w Ly n ?_ hA
  calc (n : ℝ) * (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2)
      < (1 / (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2)) * (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) :=
        mul_lt_mul_of_pos_right hn ht
    _ = 1 := one_div_mul_cancel ht.ne'

/-! ### (4) CAPSTONE: the mean-misreport KL and crossover on the physical record laws -/

/-- **CAPSTONE (per-survey KL, hypothesis-free).** For the records `y = A x + ε` of two Gaussian
truths differing by `w·v` in the mean — SAME truth root `Lx` (an arbitrary continuous linear
map), same independent invertible Gaussian noise `Lε` — the per-survey KL is exactly `β/2`,
`β = w²·blindGap = w²·(Av)ᵀ S_y⁻¹ (Av)` in the constructed data covariance
(`DataCovarianceSqrt`). No square-root or covariance-splitting hypothesis. -/
theorem model_mean_kl_closed {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) :
    klDiv (modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m)))
        (modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m)))
      = ENNReal.ofReal ((w ^ 2 * blindGap Aop Lx Lε v) / 2) := by
  have h₂ : modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))
      = (stdGaussian (E m)).map (fun x => Aop μx + dataCovSqrt (Aop ∘L Lx) Lε x) :=
    modelLaw_repr_closed Aop μx Lx Lε
  have h₁ : modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m))
      = (stdGaussian (E m)).map
          (fun x => (Aop μx + w • Aop v) + dataCovSqrt (Aop ∘L Lx) Lε x) := by
    rw [modelLaw_repr_closed Aop (μx + w • v) Lx Lε]
    congr 1
    funext z
    simp [map_add, map_smul]
  rw [h₁, h₂, blindGap_eq_norm_sq]
  exact klDiv_operator_mean Aop μx v w (dataCovSqrt (Aop ∘L Lx) Lε)

/-- **CAPSTONE (`n`-survey KL, hypothesis-free):** `n` iid surveys have KL `n·β/2`,
`β = w²·blindGap`. On the blind subspace (`Av = 0`) this is `0` for every `n`. -/
theorem model_mean_kl_pi_closed {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) (n : ℕ) :
    klDiv (Measure.pi fun _ : Fin n =>
            modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m)))
        (Measure.pi fun _ : Fin n => modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m)))
      = ENNReal.ofReal ((n : ℝ) * ((w ^ 2 * blindGap Aop Lx Lε v) / 2)) := by
  have h₂ : modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))
      = (stdGaussian (E m)).map (fun x => Aop μx + dataCovSqrt (Aop ∘L Lx) Lε x) :=
    modelLaw_repr_closed Aop μx Lx Lε
  have h₁ : modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m))
      = (stdGaussian (E m)).map
          (fun x => (Aop μx + w • Aop v) + dataCovSqrt (Aop ∘L Lx) Lε x) := by
    rw [modelLaw_repr_closed Aop (μx + w • v) Lx Lε]
    congr 1
    funext z
    simp [map_add, map_smul]
  rw [show (fun _ : Fin n => modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m)))
      = (fun _ : Fin n => (stdGaussian (E m)).map
          (fun x => (Aop μx + w • Aop v) + dataCovSqrt (Aop ∘L Lx) Lε x)) from
    funext fun _ => h₁,
    show (fun _ : Fin n => modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m)))
      = (fun _ : Fin n => (stdGaussian (E m)).map
          (fun x => Aop μx + dataCovSqrt (Aop ∘L Lx) Lε x)) from
    funext fun _ => h₂, blindGap_eq_norm_sq]
  exact klDiv_operator_mean_pi Aop μx v w (dataCovSqrt (Aop ∘L Lx) Lε) n

/-- **CAPSTONE (crossover, hypothesis-free)** — `eq:nearnull-cross`'s mean case on the physical
record laws: with `β = w²·blindGap = w²·(Av)ᵀ S_y⁻¹ (Av)`, whenever `n·β < 1` — i.e.
`n < n⋆ = 1/β` — every test on `n` iid surveys has summed type-I + type-II error strictly above
`1/2`. No hypotheses beyond the scenario (`Lx` arbitrary, `Lε` invertible). -/
theorem model_mean_kl_crossover_closed {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) (n : ℕ)
    (hn : (n : ℝ) * (w ^ 2 * blindGap Aop Lx Lε v) < 1)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m))).real A
        + (Measure.pi fun _ : Fin n =>
            modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ := by
  rw [blindGap_eq_norm_sq] at hn
  have h₂ : modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))
      = (stdGaussian (E m)).map (fun x => Aop μx + dataCovSqrt (Aop ∘L Lx) Lε x) :=
    modelLaw_repr_closed Aop μx Lx Lε
  have h₁ : modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m))
      = (stdGaussian (E m)).map
          (fun x => (Aop μx + w • Aop v) + dataCovSqrt (Aop ∘L Lx) Lε x) := by
    rw [modelLaw_repr_closed Aop (μx + w • v) Lx Lε]
    congr 1
    funext z
    simp [map_add, map_smul]
  rw [show (fun _ : Fin n => modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m)))
      = (fun _ : Fin n => (stdGaussian (E m)).map
          (fun x => (Aop μx + w • Aop v) + dataCovSqrt (Aop ∘L Lx) Lε x)) from
    funext fun _ => h₁,
    show (fun _ : Fin n => modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m)))
      = (fun _ : Fin n => (stdGaussian (E m)).map
          (fun x => Aop μx + dataCovSqrt (Aop ∘L Lx) Lε x)) from
    funext fun _ => h₂]
  exact operator_mean_kl_crossover Aop μx v w (dataCovSqrt (Aop ∘L Lx) Lε) n hn hA

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.klDiv_stdGaussian_shift
#print axioms NearNull.klDiv_stdGaussian_affine
#print axioms NearNull.klDiv_operator_mean
#print axioms NearNull.klDiv_operator_mean_precision
#print axioms NearNull.klDiv_operator_mean_pi
#print axioms NearNull.operator_mean_kl_crossover
#print axioms NearNull.operator_mean_kl_crossover_verbatim
#print axioms NearNull.model_mean_kl_closed
#print axioms NearNull.model_mean_kl_pi_closed
#print axioms NearNull.model_mean_kl_crossover_closed
