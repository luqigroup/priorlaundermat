import Mathlib
import NearNullOperator
import GaussianDataLaw

/-!
# The invertible square root of the data covariance — and the fully-closed operator undetectability

`GaussianDataLaw.lean` proved `p:nearnull`(ii) end to end on the physical record `y = A x + ε`, but
every `model_*` theorem there carries the flagged hypothesis

  `hcov : ∀ ξ, ‖Lyᵀξ‖² = ‖(A Lx)ᵀξ‖² + ‖Lεᵀξ‖²`

— "`Ly` is an invertible square root of the data covariance `S_y = A Σ⋆ Aᵀ + Γ`" — whose
*existence* was left to the paper's spectral-theorem step. This module **builds that dependency**
and then restates the undetectability with **no square-root hypothesis at all**.

**(1) The construction** (matrix route, as suggested; the pinned mathlib has no
`Matrix.PosSemidef.sqrt`, but it has the continuous-functional-calculus square root `CFC.sqrt`
on `Matrix n n ℝ` under the scoped `MatrixOrder` instances, plus the star-algebra equivalence
`Matrix.toEuclideanCLM` — exactly the toolkit `Probability/Distributions/Gaussian/Multivariate.lean`
uses). Given `T : ℝᵈ →L ℝ^m` and an **invertible** noise root `Lε : (E m) ≃L (E m)`:
* `covOp T Lε := T Tᵀ + Lε Lεᵀ` is the data covariance as a CLM; `inner_covOp_self` gives its
  quadratic form `⟪ξ, S_y ξ⟫ = ‖Tᵀξ‖² + ‖Lεᵀξ‖²`, and `covOp_isSelfAdjoint` its symmetry.
* `covMatrix := toEuclideanCLM⁻¹ (covOp)` is its standard-basis matrix; `covMatrix_posSemidef`
  proves `PosSemidef` from the quadratic form through `Matrix.inner_toEuclideanCLM`.
* `sqrtOp := toEuclideanCLM (CFC.sqrt covMatrix)` transports the CFC square root back:
  `sqrtOp_isSelfAdjoint` (CFC roots are nonneg hence self-adjoint, and star-alg equivalences
  preserve `star`), `sqrtOp_comp_self` (`sqrtOp ∘ sqrtOp = covOp`, from
  `CFC.sqrt_mul_sqrt_self` + `map_mul`).
* **Invertibility is where `Lε`'s invertibility enters:** `sqrtOp ξ = 0` forces
  `0 = ⟪ξ, covOp ξ⟫ ≥ ‖Lεᵀξ‖²`, and `Lεᵀ` is injective (two-sided inverse `(Lε⁻¹)ᵀ`), so `ξ = 0`
  (`sqrtOp_injective`); injective ⟹ bijective in finite dimension
  (`LinearMap.injective_iff_surjective`), giving the **continuous linear equivalence**
  `dataCovSqrt T Lε : (E m) ≃L[ℝ] (E m)` with `coe_dataCovSqrt : ↑(dataCovSqrt) = sqrtOp`.
* The delivered facts: `dataCovSqrt_isSelfAdjoint`, `dataCovariance_dataCovSqrt`
  (`Ly Lyᵀ = S_y` as operators), `dataCovSqrt_spec` (**exactly the `hcov` hypothesis** of
  `GaussianDataLaw`), and the existential summary `exists_invertible_dataCov_sqrt`.

**(2) The capstone.** With the square root constructed, the exponent of every bound becomes the
manuscript's `(Av)ᵀ S_y⁻¹ (Av)` in the *proved* precision `S_y⁻¹ = dataPrecision (dataCovSqrt …)`
(the unique two-sided inverse of `S_y`, by `dataPrecision_unique`): `blindGap A Lx Lε v` names it,
`blindGap_eq_norm_sq` identifies it with the whitened norm, and each wave-2 `model_*` theorem
reappears **with no `hcov`/`Ly` input**:
* `modelLaw_repr_closed` — the record `y = A x + ε` *is* `N(Aμx, S_y)` (as a pushforward), full stop.
* mean misreport (`β = w²·blindGap`): `model_mean_misreport_chi2_closed`,
  `model_mean_undetectable_closed`, `model_mean_nsurvey_undetectable_closed`,
  `model_mean_crossover_closed` (certified constant `1/5`).
* spread misreport (`β = w·blindGap`): `model_spread_undetectable_closed` packages, for **an
  existentially constructed** bumped truth root `Lx'` (via `exists_bump_sqrt`; this is the one
  place the *signal* root `Lx` must be invertible), the χ² closed form, the per-survey and
  `n`-survey Le Cam bounds, and the certified crossover (`n·β² < 2/5 ⟹` error `> 1/2`).
* the blind endpoint `Av = 0` (`v ∈ ker A`): `model_mean_blind_undetectable` /
  `model_spread_blind_undetectable` — summed type-I + type-II error `≥ 1` for every test, on the
  physical record laws, **no hypotheses beyond the scenario itself**.

## Route report
The tasked primary route (matrix transport through `Matrix.PosDef`-style sqrt machinery) worked
with one substitution: the pinned mathlib's square root of a PSD matrix is `CFC.sqrt` (the
`Matrix.PosSemidef.sqrt` of newer/older mathlibs does not exist at this pin), and invertibility of
the root is obtained *operator-theoretically* (kernel argument through the quadratic form +
finite dimension) rather than via `det`-positivity of a `PosDef` sqrt — no spectral theorem and no
eigendecomposition needed anywhere.

## Explicit hypotheses (flagged)
None beyond the scenario: `Lε` invertible (the paper's `Γ ≻ 0`) throughout, and `Lx` invertible
only where the *bumped truth* covariance root is constructed (as in wave 2's `exists_bump_sqrt`).
The former `hcov` is now a theorem (`dataCovSqrt_spec`), not a hypothesis.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on every public result lists only
`propext`, `Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure ProbabilityTheory
open scoped NNReal RealInnerProductSpace MatrixOrder

namespace NearNull

variable {d m : ℕ}

/-! ### (1a) The data covariance `S_y = T Tᵀ + Lε Lεᵀ` as an operator -/

/-- The data covariance `S_y = T Tᵀ + Lε Lεᵀ` as a continuous linear map on the data space. -/
noncomputable def covOp (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) →L[ℝ] (E m)) : (E m) →L[ℝ] (E m) :=
  T ∘L ContinuousLinearMap.adjoint T + Lε ∘L ContinuousLinearMap.adjoint Lε

/-- The quadratic form of the data covariance: `⟪ξ, S_y ξ⟫ = ‖Tᵀξ‖² + ‖Lεᵀξ‖²`. -/
theorem inner_covOp_self (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) →L[ℝ] (E m)) (ξ : E m) :
    inner ℝ ξ (covOp T Lε ξ)
      = ‖ContinuousLinearMap.adjoint T ξ‖ ^ 2
        + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2 := by
  simp only [covOp, _root_.add_apply, ContinuousLinearMap.comp_apply,
    inner_add_right]
  rw [← ContinuousLinearMap.adjoint_inner_left T (ContinuousLinearMap.adjoint T ξ) ξ,
    ← ContinuousLinearMap.adjoint_inner_left Lε (ContinuousLinearMap.adjoint Lε ξ) ξ,
    real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq]

/-- The data covariance is self-adjoint. -/
theorem covOp_isSelfAdjoint (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) →L[ℝ] (E m)) : IsSelfAdjoint (covOp T Lε) := by
  rw [IsSelfAdjoint, ContinuousLinearMap.star_eq_adjoint, covOp, map_add,
    ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.adjoint_adjoint,
    ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.adjoint_adjoint]

/-! ### (1b) Matrix transport and the CFC square root -/

/-- The matrix of the data covariance in the standard basis of `E m`. -/
noncomputable def covMatrix (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) →L[ℝ] (E m)) : Matrix (Fin (m + 1)) (Fin (m + 1)) ℝ :=
  (Matrix.toEuclideanCLM (𝕜 := ℝ)).symm (covOp T Lε)

/-- Transporting the matrix back gives the operator. -/
theorem toEuclideanCLM_covMatrix (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) →L[ℝ] (E m)) :
    Matrix.toEuclideanCLM (𝕜 := ℝ) (covMatrix T Lε) = covOp T Lε :=
  (Matrix.toEuclideanCLM (𝕜 := ℝ)).apply_symm_apply _

/-- The data covariance matrix is positive semidefinite (Hermitian by self-adjointness of
`covOp`; the quadratic form is the sum of two squares by `inner_covOp_self`). -/
theorem covMatrix_posSemidef (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) →L[ℝ] (E m)) : (covMatrix T Lε).PosSemidef := by
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg
    (Matrix.isHermitian_iff_isSelfAdjoint.mpr
      ((covOp_isSelfAdjoint T Lε).map (Matrix.toEuclideanCLM (𝕜 := ℝ)).symm)) fun x => ?_
  have h := Matrix.inner_toEuclideanCLM (covMatrix T Lε)
    (WithLp.toLp 2 x) (WithLp.toLp 2 x)
  rw [toEuclideanCLM_covMatrix, inner_covOp_self] at h
  have h0 : (0 : ℝ) ≤ ‖ContinuousLinearMap.adjoint T (WithLp.toLp 2 x)‖ ^ 2
      + ‖ContinuousLinearMap.adjoint Lε (WithLp.toLp 2 x)‖ ^ 2 := by positivity
  rw [h] at h0
  simpa [WithLp.ofLp_toLp] using h0

/-- The square root of the data covariance, transported back from the continuous-functional-
calculus square root of its matrix: `sqrtOp = toEuclideanCLM (CFC.sqrt covMatrix)`. -/
noncomputable def sqrtOp (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) →L[ℝ] (E m)) : (E m) →L[ℝ] (E m) :=
  Matrix.toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt (covMatrix T Lε))

/-- The square root is self-adjoint (CFC square roots are nonnegative, hence self-adjoint, and
the star-algebra equivalence preserves `star`). -/
theorem sqrtOp_isSelfAdjoint (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) →L[ℝ] (E m)) : IsSelfAdjoint (sqrtOp T Lε) :=
  (CFC.sqrt_nonneg (covMatrix T Lε)).isSelfAdjoint.map (Matrix.toEuclideanCLM (𝕜 := ℝ))

/-- **The square-root property:** `sqrtOp ∘ sqrtOp = covOp`. -/
theorem sqrtOp_comp_self (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) →L[ℝ] (E m)) : sqrtOp T Lε ∘L sqrtOp T Lε = covOp T Lε := by
  have h : sqrtOp T Lε * sqrtOp T Lε
      = Matrix.toEuclideanCLM (𝕜 := ℝ)
          (CFC.sqrt (covMatrix T Lε) * CFC.sqrt (covMatrix T Lε)) :=
    (map_mul _ _ _).symm
  rw [← ContinuousLinearMap.mul_def, h,
    CFC.sqrt_mul_sqrt_self (covMatrix T Lε) (covMatrix_posSemidef T Lε).nonneg,
    toEuclideanCLM_covMatrix]

/-- `(Lε⁻¹)ᵀ ∘ Lεᵀ = id` pointwise: the adjoint of an invertible map is injective. -/
private lemma adjoint_symm_apply_adjoint (Lε : (E m) ≃L[ℝ] (E m)) (ξ : E m) :
    ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))
      (ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) ξ) = ξ := by
  have hcomp : (↑Lε : (E m) →L[ℝ] (E m)) ∘L (↑Lε.symm : (E m) →L[ℝ] (E m))
      = ContinuousLinearMap.id ℝ (E m) := by ext x; simp
  rw [← ContinuousLinearMap.comp_apply, ← ContinuousLinearMap.adjoint_comp, hcomp,
    ContinuousLinearMap.adjoint_id, ContinuousLinearMap.id_apply]

/-- **Injectivity of the square root** — this is where invertibility of the noise root enters:
`sqrtOp ξ = 0` forces `⟪ξ, S_y ξ⟫ = 0`, hence `‖Lεᵀξ‖² = 0`, hence `ξ = 0` since `Lεᵀ` has the
two-sided inverse `(Lε⁻¹)ᵀ`. -/
theorem sqrtOp_injective (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) ≃L[ℝ] (E m)) :
    Function.Injective (sqrtOp T (↑Lε : (E m) →L[ℝ] (E m))) := by
  intro ξ η hξη
  have hz : sqrtOp T (↑Lε : (E m) →L[ℝ] (E m)) (ξ - η) = 0 := by
    rw [map_sub, hξη, sub_self]
  have hcov : covOp T (↑Lε : (E m) →L[ℝ] (E m)) (ξ - η) = 0 := by
    rw [← sqrtOp_comp_self T (↑Lε : (E m) →L[ℝ] (E m)),
      ContinuousLinearMap.comp_apply, hz, map_zero]
  have hsum : ‖ContinuousLinearMap.adjoint T (ξ - η)‖ ^ 2
      + ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) (ξ - η)‖ ^ 2 = 0 := by
    rw [← inner_covOp_self, hcov, inner_zero_right]
  have hsq : ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) (ξ - η)‖ ^ 2 = 0 := by
    have h1 : (0 : ℝ) ≤ ‖ContinuousLinearMap.adjoint T (ξ - η)‖ ^ 2 := sq_nonneg _
    have h2 : (0 : ℝ)
        ≤ ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) (ξ - η)‖ ^ 2 :=
      sq_nonneg _
    linarith
  have hker : ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) (ξ - η) = 0 :=
    norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp hsq)
  have hz2 : ξ - η = 0 := by
    have h := congrArg (ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))) hker
    rwa [adjoint_symm_apply_adjoint, map_zero] at h
  exact sub_eq_zero.mp hz2

/-! ### (1c) The invertible square root as a continuous linear equivalence -/

/-- **The invertible square root of the data covariance** `S_y = T Tᵀ + Lε Lεᵀ`, as a continuous
linear equivalence: the injective `sqrtOp` is bijective in finite dimension. This is the object
whose existence `GaussianDataLaw.lean` hypothesized (`hcov`). -/
noncomputable def dataCovSqrt (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) ≃L[ℝ] (E m)) : (E m) ≃L[ℝ] (E m) :=
  (LinearEquiv.ofInjectiveEndo
    ((sqrtOp T (↑Lε : (E m) →L[ℝ] (E m)) : (E m) →L[ℝ] (E m)) : (E m) →ₗ[ℝ] (E m))
    (sqrtOp_injective T Lε)).toContinuousLinearEquiv

/-- The equivalence is the square-root operator. -/
theorem coe_dataCovSqrt (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) ≃L[ℝ] (E m)) :
    (↑(dataCovSqrt T Lε) : (E m) →L[ℝ] (E m))
      = sqrtOp T (↑Lε : (E m) →L[ℝ] (E m)) := by
  ext ξ; rfl

/-- The constructed square root is self-adjoint: `Lyᵀ = Ly`. -/
theorem dataCovSqrt_isSelfAdjoint (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) ≃L[ℝ] (E m)) :
    IsSelfAdjoint (↑(dataCovSqrt T Lε) : (E m) →L[ℝ] (E m)) := by
  rw [coe_dataCovSqrt]
  exact sqrtOp_isSelfAdjoint T (↑Lε : (E m) →L[ℝ] (E m))

/-- **`Ly Lyᵀ = S_y` as operators:** the `dataCovariance` of the constructed root (in the sense of
`GaussianDataLaw.dataCovariance`) is exactly `T Tᵀ + Lε Lεᵀ`. -/
theorem dataCovariance_dataCovSqrt (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) ≃L[ℝ] (E m)) :
    dataCovariance (dataCovSqrt T Lε) = covOp T (↑Lε : (E m) →L[ℝ] (E m)) := by
  rw [dataCovariance, coe_dataCovSqrt,
    (sqrtOp_isSelfAdjoint T (↑Lε : (E m) →L[ℝ] (E m))).adjoint_eq,
    sqrtOp_comp_self]

/-- **The former `hcov` hypothesis, now a theorem:** the constructed `Ly` satisfies
`‖Lyᵀξ‖² = ‖Tᵀξ‖² + ‖Lεᵀξ‖²` for all `ξ` — the exact variational covariance-splitting input of
every `model_*` theorem in `GaussianDataLaw.lean`. -/
theorem dataCovSqrt_spec (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) ≃L[ℝ] (E m)) (ξ : E m) :
    ‖ContinuousLinearMap.adjoint (↑(dataCovSqrt T Lε) : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
      = ‖ContinuousLinearMap.adjoint T ξ‖ ^ 2
        + ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) ξ‖ ^ 2 := by
  rw [← inner_dataCovariance_self (dataCovSqrt T Lε) ξ, dataCovariance_dataCovSqrt,
    inner_covOp_self]

/-- **Existential summary (task deliverable 1):** for every `T` and every invertible noise root
`Lε` there is an invertible, self-adjoint `Ly` with `Ly Lyᵀ = T Tᵀ + Lε Lεᵀ`, in both the
operator and the variational form. -/
theorem exists_invertible_dataCov_sqrt (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lε : (E m) ≃L[ℝ] (E m)) :
    ∃ Ly : (E m) ≃L[ℝ] (E m),
      IsSelfAdjoint (↑Ly : (E m) →L[ℝ] (E m)) ∧
      dataCovariance Ly = covOp T (↑Lε : (E m) →L[ℝ] (E m)) ∧
      ∀ ξ : E m, ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint T ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) ξ‖ ^ 2 :=
  ⟨dataCovSqrt T Lε, dataCovSqrt_isSelfAdjoint T Lε, dataCovariance_dataCovSqrt T Lε,
    dataCovSqrt_spec T Lε⟩

/-! ### (2a) The undetectability exponent `(Av)ᵀ S_y⁻¹ (Av)` -/

/-- **The blind gap** `(Av)ᵀ S_y⁻¹ (Av)`: the manuscript's undetectability exponent, in the proved
precision `S_y⁻¹ = dataPrecision (dataCovSqrt (A Lx) Lε)` (the unique two-sided inverse of
`S_y = (A Lx)(A Lx)ᵀ + Lε Lεᵀ`, by `dataCovariance_dataCovSqrt` + `dataPrecision_unique`).
It vanishes exactly on the blind subspace (`blindGap_eq_zero_of_blind`). -/
noncomputable def blindGap (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) (v : EuclideanSpace ℝ (Fin d)) : ℝ :=
  inner ℝ (Aop v) (dataPrecision (dataCovSqrt (Aop ∘L Lx) Lε) (Aop v))

/-- The blind gap is the whitened norm `‖Ly⁻¹(Av)‖²` of the wave-2 closed forms. -/
theorem blindGap_eq_norm_sq (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) (v : EuclideanSpace ℝ (Fin d)) :
    blindGap Aop Lx Lε v = ‖(dataCovSqrt (Aop ∘L Lx) Lε).symm (Aop v)‖ ^ 2 :=
  (norm_symm_sq_eq_inner_dataPrecision (dataCovSqrt (Aop ∘L Lx) Lε) (Aop v)).symm

/-- The blind gap is nonnegative. -/
theorem blindGap_nonneg (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) (v : EuclideanSpace ℝ (Fin d)) :
    0 ≤ blindGap Aop Lx Lε v := by
  rw [blindGap_eq_norm_sq]; positivity

/-- On the blind subspace (`Av = 0`, e.g. `v ∈ ker A`) the blind gap vanishes. -/
theorem blindGap_eq_zero_of_blind (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) {v : EuclideanSpace ℝ (Fin d)} (hAv : Aop v = 0) :
    blindGap Aop Lx Lε v = 0 := by
  rw [blindGap_eq_norm_sq, hAv, map_zero, norm_zero]
  norm_num

/-! ### (2b) CAPSTONE: the operator undetectability with no square-root hypothesis -/

/-- **CAPSTONE (representation).** The record `y = A x + ε` of a Gaussian truth
`x ∼ N(μx, Lx Lxᵀ)` under independent invertible Gaussian noise `ε ∼ N(0, Lε Lεᵀ)` **is** the
Gaussian `N(Aμx, S_y)`, exhibited as a pushforward through the constructed square root — no
covariance-splitting hypothesis. -/
theorem modelLaw_repr_closed (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx : EuclideanSpace ℝ (Fin d))
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) :
    modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))
      = (stdGaussian (E m)).map (fun z => Aop μx + dataCovSqrt (Aop ∘L Lx) Lε z) :=
  modelLaw_repr Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))
    (↑(dataCovSqrt (Aop ∘L Lx) Lε) : (E m) →L[ℝ] (E m))
    (dataCovSqrt_spec (Aop ∘L Lx) Lε)

/-- **CAPSTONE (mean misreport, χ² closed form, hypothesis-free):**
`χ² = exp(w²·(Av)ᵀS_y⁻¹(Av)) − 1` on the record laws. -/
theorem model_mean_misreport_chi2_closed (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) :
    NearNull.chi2 (modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m)))
        (modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m)))
      = Real.exp (w ^ 2 * blindGap Aop Lx Lε v) - 1 := by
  rw [blindGap_eq_norm_sq]
  exact model_mean_misreport_chi2 Aop μx v w Lx (↑Lε : (E m) →L[ℝ] (E m))
    (dataCovSqrt (Aop ∘L Lx) Lε) (dataCovSqrt_spec (Aop ∘L Lx) Lε)

/-- **CAPSTONE (mean misreport undetectability, hypothesis-free):** every test between the record
laws of two truths differing by `w·v` in the mean has summed type-I + type-II error at least
`1 − √(exp(w²·(Av)ᵀS_y⁻¹(Av)) − 1)`. -/
theorem model_mean_undetectable_closed (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) {A : Set (E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt (Real.exp (w ^ 2 * blindGap Aop Lx Lε v) - 1)
      ≤ (modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m))).real A
        + (modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ := by
  have h := model_mean_misreport_undetectable Aop μx v w Lx (↑Lε : (E m) →L[ℝ] (E m))
    (dataCovSqrt (Aop ∘L Lx) Lε) (dataCovSqrt_spec (Aop ∘L Lx) Lε) hA
  rwa [← blindGap_eq_norm_sq Aop Lx Lε v] at h

/-- **CAPSTONE (mean misreport, `n` surveys, hypothesis-free):** summed error at least
`1 − √(exp(β)ⁿ − 1)`, `β = w²·(Av)ᵀS_y⁻¹(Av)`. -/
theorem model_mean_nsurvey_undetectable_closed (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) (n : ℕ) {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt ((Real.exp (w ^ 2 * blindGap Aop Lx Lε v)) ^ n - 1)
      ≤ (Measure.pi fun _ : Fin n =>
            modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m))).real A
        + (Measure.pi fun _ : Fin n =>
            modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ := by
  have h := model_mean_nsurvey_undetectable Aop μx v w Lx (↑Lε : (E m) →L[ℝ] (E m))
    (dataCovSqrt (Aop ∘L Lx) Lε) (dataCovSqrt_spec (Aop ∘L Lx) Lε) n hA
  rwa [← blindGap_eq_norm_sq Aop Lx Lε v] at h

/-- **CAPSTONE (mean crossover, certified, hypothesis-free):** with
`β = w²·(Av)ᵀS_y⁻¹(Av)`, whenever `n·β < 1/5` every test on `n` surveys has summed error
strictly above `1/2`. -/
theorem model_mean_crossover_closed (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) (n : ℕ)
    (hn : (n : ℝ) * (w ^ 2 * blindGap Aop Lx Lε v) < 1 / 5)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m))).real A
        + (Measure.pi fun _ : Fin n =>
            modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ := by
  rw [blindGap_eq_norm_sq Aop Lx Lε v] at hn
  exact model_mean_crossover_certified Aop μx v w Lx (↑Lε : (E m) →L[ℝ] (E m))
    (dataCovSqrt (Aop ∘L Lx) Lε) (dataCovSqrt_spec (Aop ∘L Lx) Lε) n hn hA

/-- **CAPSTONE (spread misreport, hypothesis-free package).** For a Gaussian truth with
invertible root `Lx` and invertible noise root `Lε`, there **exists** a root `Lx'` of the bumped
truth covariance `Σ⋆ + w v vᵀ` such that, with `β = w·(Av)ᵀS_y⁻¹(Av)` (`= w·blindGap`), the two
record laws `modelLaw(Lx)` vs `modelLaw(Lx')` satisfy: (i) the χ² closed form
`(1+β)/√(1+2β) − 1`; (ii) the per-survey Le Cam bound `1 − √χ²`; (iii) the `n`-survey bound
`1 − √((1+χ²)ⁿ − 1)`; (iv) the certified crossover — `n·β² < 2/5` forces every test's summed
error above `1/2`. No square-root or covariance-splitting hypothesis remains. -/
theorem model_spread_undetectable_closed (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ} (hw : 0 ≤ w)
    (Lx : EuclideanSpace ℝ (Fin d) ≃L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) :
    ∃ Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d),
      (∀ η : EuclideanSpace ℝ (Fin d),
        ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
          = ‖ContinuousLinearMap.adjoint
              (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) η‖ ^ 2
            + w * (inner ℝ v η) ^ 2) ∧
      NearNull.chi2
          (modelLaw Aop μx (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
            (↑Lε : (E m) →L[ℝ] (E m)))
          (modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m)))
        = (1 + w * blindGap Aop (↑Lx) Lε v)
            / Real.sqrt (1 + 2 * (w * blindGap Aop (↑Lx) Lε v)) - 1 ∧
      (∀ {A : Set (E m)}, MeasurableSet A →
        1 - Real.sqrt ((1 + w * blindGap Aop (↑Lx) Lε v)
              / Real.sqrt (1 + 2 * (w * blindGap Aop (↑Lx) Lε v)) - 1)
          ≤ (modelLaw Aop μx
                (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
                (↑Lε : (E m) →L[ℝ] (E m))).real A
            + (modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ) ∧
      (∀ (n : ℕ) {A : Set (Fin n → E m)}, MeasurableSet A →
        1 - Real.sqrt (((1 + w * blindGap Aop (↑Lx) Lε v)
              / Real.sqrt (1 + 2 * (w * blindGap Aop (↑Lx) Lε v))) ^ n - 1)
          ≤ (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx
                  (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
                  (↑Lε : (E m) →L[ℝ] (E m))).real A
            + (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ) ∧
      (∀ (n : ℕ), (n : ℝ) * (w * blindGap Aop (↑Lx) Lε v) ^ 2 < 2 / 5 →
        ∀ {A : Set (Fin n → E m)}, MeasurableSet A →
        1 / 2 < (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx
                  (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
                  (↑Lε : (E m) →L[ℝ] (E m))).real A
            + (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ) := by
  obtain ⟨Lx', hbump⟩ := exists_bump_sqrt Lx v hw
  obtain ⟨M, hM⟩ := exists_rank1_sqrt
    (Real.sqrt w •
      (dataCovSqrt (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
        EuclideanSpace ℝ (Fin d))) Lε).symm (Aop v))
  have hcov := dataCovSqrt_spec
    (Aop ∘L (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))) Lε
  have hβ := blindGap_eq_norm_sq Aop
    (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε v
  refine ⟨Lx', hbump, ?_, ?_, ?_, ?_⟩
  · rw [hβ]
    exact model_spread_misreport_chi2 Aop μx v hw
      (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lx'
      (↑Lε : (E m) →L[ℝ] (E m)) _ M hcov hbump hM
  · intro A hA
    rw [hβ]
    exact model_spread_misreport_undetectable Aop μx v hw
      (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lx'
      (↑Lε : (E m) →L[ℝ] (E m)) _ M hcov hbump hM hA
  · intro n A hA
    rw [hβ]
    exact model_spread_nsurvey_undetectable Aop μx v hw
      (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lx'
      (↑Lε : (E m) →L[ℝ] (E m)) _ M hcov hbump hM n hA
  · intro n hn A hA
    rw [hβ] at hn
    exact model_spread_crossover_certified Aop μx v hw
      (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lx'
      (↑Lε : (E m) →L[ℝ] (E m)) _ M hcov hbump hM n hn hA

/-! ### (2c) The blind endpoint: full undetectability on `ker A`, no hypotheses -/

/-- **The blind endpoint (mean), fully closed:** if `Av = 0`, then for the physical record laws of
two Gaussian truths differing by `w·v` in the mean, **every** test's summed type-I + type-II error
is at least `1` — undetectability on the blind subspace with no square-root hypothesis. -/
theorem model_mean_blind_undetectable (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) (hAv : Aop v = 0)
    {A : Set (E m)} (hA : MeasurableSet A) :
    1 ≤ (modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m))).real A
        + (modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ := by
  have h := model_mean_undetectable_closed Aop μx v w Lx Lε hA
  rwa [blindGap_eq_zero_of_blind Aop Lx Lε hAv, mul_zero, Real.exp_zero, sub_self,
    Real.sqrt_zero, sub_zero] at h

/-- **The blind endpoint (spread), fully closed:** if `Av = 0`, there exists a root `Lx'` of the
bumped truth covariance `Σ⋆ + w v vᵀ` such that **every** test between the two record laws has
summed error at least `1` — the prior's spread on the blind subspace is unverifiable from data,
with no hypotheses beyond the scenario. -/
theorem model_spread_blind_undetectable (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ} (hw : 0 ≤ w)
    (Lx : EuclideanSpace ℝ (Fin d) ≃L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) (hAv : Aop v = 0) :
    ∃ Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d),
      (∀ η : EuclideanSpace ℝ (Fin d),
        ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
          = ‖ContinuousLinearMap.adjoint
              (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) η‖ ^ 2
            + w * (inner ℝ v η) ^ 2) ∧
      ∀ {A : Set (E m)}, MeasurableSet A →
        1 ≤ (modelLaw Aop μx
              (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
              (↑Lε : (E m) →L[ℝ] (E m))).real A
          + (modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ := by
  obtain ⟨Lx', hbump, _, hper, _, _⟩ := model_spread_undetectable_closed Aop μx v hw Lx Lε
  refine ⟨Lx', hbump, fun {A} hA => ?_⟩
  have h := hper hA
  rwa [blindGap_eq_zero_of_blind Aop
      (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε hAv,
    mul_zero, mul_zero, add_zero, Real.sqrt_one, div_one, sub_self,
    Real.sqrt_zero, sub_zero] at h

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.inner_covOp_self
#print axioms NearNull.covOp_isSelfAdjoint
#print axioms NearNull.toEuclideanCLM_covMatrix
#print axioms NearNull.covMatrix_posSemidef
#print axioms NearNull.sqrtOp_isSelfAdjoint
#print axioms NearNull.sqrtOp_comp_self
#print axioms NearNull.sqrtOp_injective
#print axioms NearNull.coe_dataCovSqrt
#print axioms NearNull.dataCovSqrt_isSelfAdjoint
#print axioms NearNull.dataCovariance_dataCovSqrt
#print axioms NearNull.dataCovSqrt_spec
#print axioms NearNull.exists_invertible_dataCov_sqrt
#print axioms NearNull.blindGap_eq_norm_sq
#print axioms NearNull.blindGap_nonneg
#print axioms NearNull.blindGap_eq_zero_of_blind
#print axioms NearNull.modelLaw_repr_closed
#print axioms NearNull.model_mean_misreport_chi2_closed
#print axioms NearNull.model_mean_undetectable_closed
#print axioms NearNull.model_mean_nsurvey_undetectable_closed
#print axioms NearNull.model_mean_crossover_closed
#print axioms NearNull.model_spread_undetectable_closed
#print axioms NearNull.model_mean_blind_undetectable
#print axioms NearNull.model_spread_blind_undetectable
