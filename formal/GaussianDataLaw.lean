import Mathlib
import ChiSquaredTV
import ChiSquaredInvariance
import ChiSquaredPi
import GaussianChiSquared
import MultivariateGaussianChiSquared
import MultivariateGaussianWhitening
import MultivariateGaussianScale
import MultivariateGaussianSpread
import NearNullOperator

/-!
# The data-law representation for `p:nearnull`(ii): `y = A x + ε` really has the pushforward law

`NearNullOperator.lean` proves the near-null undetectability on Gaussian data laws **given as
pushforwards** `N(Aμ, S_y) = N₀ ∘ (x ↦ Aμ + Ly x)`, and its docstring flags three readings left to
the paper: (a) that the record `y = A x + ε` of a Gaussian truth under independent Gaussian noise
has exactly that marginal with `Ly Lyᵀ = A Σ⋆ Aᵀ + Γ`; (b) that `‖Ly⁻¹(Av)‖² = (Av)ᵀ S_y⁻¹ (Av)`;
(c) the translation of the χ²-route threshold into `eq:nearnull-cross`'s crossover constant. This
module closes all three inside Lean.

**(B1) The convolution/pushforward identity.** `modelLaw A μx Lx Lε` is the *physical* record law:
the pushforward of the product `N₀(ℝᵈ) ⊗ N₀(ℝᵐ)` under `(z, w) ↦ A(μx + Lx z) + Lε w` — i.e.
`y = A x + ε` with truth `x = μx + Lx z ∼ N(μx, Lx Lxᵀ)` and noise `ε = Lε w ∼ N(0, Lε Lεᵀ)`,
independent *by construction* (product measure); `modelLaw_eq_conv` exhibits it as the convolution
`(N₀ ∘ (z ↦ A(μx + Lx z))) ∗ (N₀ ∘ Lε)` of the signal and noise laws. The representation theorem
`signal_noise_law` / `modelLaw_repr` proves `modelLaw A μx Lx Lε = N₀ ∘ (z ↦ Aμx + Ly z)` for
**any** `Ly` whose covariance splits as `Ly Lyᵀ = (A Lx)(A Lx)ᵀ + Lε Lεᵀ`, stated variationally:
`‖Lyᵀ ξ‖² = ‖(A Lx)ᵀ ξ‖² + ‖Lεᵀ ξ‖²` for all `ξ` (hypothesis `hcov`). Proof: characteristic-function
uniqueness (`Measure.ext_of_charFun`) — the charFun of the convolution is the product of the two
Gaussian charFuns (`charFun_conv`), `exp(i⟪c,t⟫)·exp(−‖(A Lx)ᵀt‖²/2)·exp(−‖Lεᵀt‖²/2)`, which `hcov`
matches to `exp(i⟪c,t⟫)·exp(−‖Lyᵀt‖²/2)`. Composing with `NearNullOperator`, the undetectability
and χ² statements then hold **on the model laws themselves**: `model_mean_misreport_chi2`,
`model_spread_misreport_chi2`, `model_*_misreport_undetectable`, `model_*_nsurvey_undetectable`.

**(B2) The β identification.** With `S_y := Ly Lyᵀ` (`dataCovariance`) and
`S_y⁻¹ := (Ly⁻¹)ᵀ Ly⁻¹` (`dataPrecision`), `dataCovariance_comp_dataPrecision` /
`dataPrecision_comp_dataCovariance` prove the two are mutually inverse (and `dataPrecision_unique`
that any two-sided inverse of `S_y` *is* `dataPrecision`), and
`norm_symm_sq_eq_inner_dataPrecision` proves `‖Ly⁻¹ u‖² = ⟪u, S_y⁻¹ u⟫`. So the exponents of
`NearNullOperator` read exactly as the manuscript's `w·(Av)ᵀ S_y⁻¹ (Av)`:
`operator_mean_misreport_chi2_precision` and `operator_spread_misreport_chi2_precision` restate the
closed forms in that variable.

**(B3) The crossover constant.** The pinned mathlib has **no Pinsker inequality and no Gaussian KL
closed form** (checked by grep: `Pinsker` absent; `klDiv` exists only abstractly), so the paper's
KL route to `eq:nearnull-cross` (`n < 2/t²`) is replaced by the certified χ² route in the *same*
variable `t = w (Av)ᵀ S_y⁻¹ (Av)`: `chi2_spread_le_half_sq` proves the comparison
`(1+β)/√(1+2β) − 1 ≤ β²/2` for all `β ≥ 0` (stronger than the tasked `≤ β²` on `[0,1]`), and
`chi2_pow_lt_of_mul_lt` turns it into `n·β² < 2/5 ⟹ ((1+β)/√(1+2β))ⁿ < 5/4` via
`(1+x)ⁿ ≤ exp(nx)` and the certified numeric bound `exp(1/5) < 5/4` (from `exp 1 < (5/4)⁵`).
`operator_spread_crossover_certified` / `model_spread_crossover_certified` then give the paper's
crossover with the explicit machine constant: **if `n·t² < 2/5`, every test on `n` surveys has
summed type-I + type-II error above `1/2`** — the same `1/t²` scaling as `eq:nearnull-cross`, with
certified constant `2/5` in place of the paper's Pinsker constant `2`. (Note: the task's suggested
chain `n·β² < 1/4 ⟹ (1+χ²)ⁿ < 5/4` is false at the threshold — `e^{1/4} ≈ 1.284 > 5/4` — so the
constant `2/5` is used, for which `e^{1/5} < 5/4` is provable.) The mean-misreport analogue
(`n·β < 1/5`, `β = w²(Av)ᵀS_y⁻¹(Av)`) is `operator_mean_crossover_certified` /
`model_mean_crossover_certified`.

## Explicit hypotheses (flagged, not machine-derived)
* `hcov : ∀ ξ, ‖Lyᵀξ‖² = ‖(A∘Lx)ᵀξ‖² + ‖Lεᵀξ‖²` — the *definition* of `Ly` as a square root of the
  data covariance `A Σ⋆ Aᵀ + Γ`. Existence of such an (invertible) `Ly` is the spectral-theorem step
  the paper takes (it holds whenever `Γ ≻ 0`); it is **not** constructed here.
* `hbump : ∀ η, ‖Lx'ᵀη‖² = ‖Lxᵀη‖² + w⟪v,η⟫²` — `Lx'` realizes the misreported truth covariance
  `Σ⋆ + w v vᵀ`. **Satisfiable for every `v, w ≥ 0`** when `Lx` is invertible: `exists_bump_sqrt`
  constructs a witness.
* `hM : M Mᵀ = I + u uᵀ` (with `u = √w·Ly⁻¹(Av)`) — inherited from `NearNullOperator`;
  satisfiable for every direction by `NearNull.exists_rank1_sqrt`.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each public result lists only `propext`,
`Classical.choice`, `Quot.sound`. What remains on paper after this module: only the existence of the
invertible square root `Ly` itself (the hypothesis `hcov` above), and the identification of the
variational statements with the matrix equalities they encode.
-/

open MeasureTheory Measure ProbabilityTheory
open scoped NNReal RealInnerProductSpace

namespace NearNull

/-! ### (1) Characteristic-function helpers -/

/-- The charFun of a pushforward by a continuous linear map is the charFun at the adjoint point:
`charFun (μ ∘ T⁻¹) t = charFun μ (Tᵀ t)`. -/
private lemma charFun_map_clm {d k : ℕ} (μ : Measure (EuclideanSpace ℝ (Fin d)))
    (T : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin k))
    (t : EuclideanSpace ℝ (Fin k)) :
    charFun (μ.map T) t = charFun μ (ContinuousLinearMap.adjoint T t) := by
  rw [charFun_apply, charFun_apply,
    integral_map T.continuous.measurable.aemeasurable
      (Continuous.aestronglyMeasurable (by fun_prop))]
  simp_rw [ContinuousLinearMap.adjoint_inner_right]

/-- The charFun of the affine Gaussian pushforward `N₀ ∘ (z ↦ c + T z)`:
`exp(i⟪c,t⟫)·exp(−‖Tᵀt‖²/2)`. -/
private lemma charFun_stdGaussian_map_affine {d k : ℕ} (c : EuclideanSpace ℝ (Fin k))
    (T : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin k))
    (t : EuclideanSpace ℝ (Fin k)) :
    charFun ((stdGaussian (EuclideanSpace ℝ (Fin d))).map (fun z => c + T z)) t
      = Complex.exp (⟪c, t⟫ * Complex.I)
          * Complex.exp (- ‖ContinuousLinearMap.adjoint T t‖ ^ 2 / 2) := by
  have hcomp : (fun z : EuclideanSpace ℝ (Fin d) => c + T z) = (c + ·) ∘ ⇑T := rfl
  rw [hcomp, ← Measure.map_map (by fun_prop) T.continuous.measurable,
    charFun_map_const_add, charFun_map_clm, charFun_stdGaussian, mul_comm]

/-! ### (2) (B1) The record `y = A x + ε` as a product-measure pushforward -/

/-- **The physical record law** `y = A x + ε`: the pushforward of the *product* measure
`N₀(ℝᵈ) ⊗ N₀(ℝᵐ)` (independence by construction) under `(z, w) ↦ A(μx + Lx z) + Lε w`, i.e. the
truth is `x = μx + Lx z ∼ N(μx, Lx Lxᵀ)` and the noise `ε = Lε w ∼ N(0, Lε Lεᵀ)`, independent. -/
noncomputable def modelLaw {d m : ℕ} (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx : EuclideanSpace ℝ (Fin d))
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) : Measure (E m) :=
  ((stdGaussian (EuclideanSpace ℝ (Fin d))).prod (stdGaussian (E m))).map
    (fun p => Aop (μx + Lx p.1) + Lε p.2)

/-- The signal+noise pushforward of the product measure is the **convolution** of the signal law
and the noise law: `(N₀ ⊗ N₀) ∘ ((z,w) ↦ (c + T z) + Lε w)⁻¹ = (N₀ ∘ (c + T·)) ∗ (N₀ ∘ Lε)`. -/
private lemma signalNoise_eq_conv {d m : ℕ} (c : E m)
    (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (Lε : (E m) →L[ℝ] (E m)) :
    ((stdGaussian (EuclideanSpace ℝ (Fin d))).prod (stdGaussian (E m))).map
        (fun p => c + T p.1 + Lε p.2)
      = ((stdGaussian (EuclideanSpace ℝ (Fin d))).map (fun z => c + T z))
          ∗ ((stdGaussian (E m)).map ⇑Lε) := by
  have h : ((stdGaussian (EuclideanSpace ℝ (Fin d))).map (fun z => c + T z))
        ∗ ((stdGaussian (E m)).map ⇑Lε)
      = (((stdGaussian (EuclideanSpace ℝ (Fin d))).map (fun z => c + T z)).prod
          ((stdGaussian (E m)).map ⇑Lε)).map (fun q : (E m) × (E m) => q.1 + q.2) := rfl
  rw [h, Measure.map_prod_map _ _ (by fun_prop) Lε.continuous.measurable,
    Measure.map_map (by fun_prop) (by fun_prop)]
  rfl

/-- **(B1, primary) The convolution/pushforward identity.** The law of `c + T z + Lε w` under
independent standard Gaussians `z, w` equals the single-Gaussian pushforward `N₀ ∘ (z ↦ c + Ly z)`
for **any** `Ly` whose covariance splits as `Ly Lyᵀ = T Tᵀ + Lε Lεᵀ` (hypothesis `hcov`, in
variational form). Proved by characteristic-function uniqueness: both sides have charFun
`exp(i⟪c,t⟫ − ‖Lyᵀt‖²/2)`. -/
theorem signal_noise_law {d m : ℕ} (c : E m)
    (T : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (Lε Ly : (E m) →L[ℝ] (E m))
    (hcov : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint Ly ξ‖ ^ 2
      = ‖ContinuousLinearMap.adjoint T ξ‖ ^ 2 + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2) :
    ((stdGaussian (EuclideanSpace ℝ (Fin d))).prod (stdGaussian (E m))).map
        (fun p => c + T p.1 + Lε p.2)
      = (stdGaussian (E m)).map (fun z => c + Ly z) := by
  haveI h₁ : IsProbabilityMeasure
      ((stdGaussian (EuclideanSpace ℝ (Fin d))).map (fun z => c + T z)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI h₂ : IsProbabilityMeasure ((stdGaussian (E m)).map ⇑Lε) :=
    isProbabilityMeasure_map Lε.continuous.measurable.aemeasurable
  haveI h₃ : IsProbabilityMeasure
      (((stdGaussian (EuclideanSpace ℝ (Fin d))).prod (stdGaussian (E m))).map
        (fun p => c + T p.1 + Lε p.2)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  haveI h₄ : IsProbabilityMeasure ((stdGaussian (E m)).map (fun z => c + Ly z)) :=
    isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
  apply Measure.ext_of_charFun
  funext t
  conv_rhs => rw [charFun_stdGaussian_map_affine]
  rw [signalNoise_eq_conv c T Lε, charFun_conv, charFun_stdGaussian_map_affine,
    charFun_map_clm, charFun_stdGaussian, mul_assoc, ← Complex.exp_add]
  congr 1
  congr 1
  have hc : ((‖ContinuousLinearMap.adjoint T t‖ : ℂ)) ^ 2
        + ((‖ContinuousLinearMap.adjoint Lε t‖ : ℂ)) ^ 2
      = ((‖ContinuousLinearMap.adjoint Ly t‖ : ℂ)) ^ 2 := by
    exact_mod_cast (hcov t).symm
  linear_combination (-1 / 2 : ℂ) * hc

/-- **The record law is the convolution of signal and noise** — the independence reading of
`modelLaw`, exhibited explicitly: `modelLaw = (N₀ ∘ (z ↦ A(μx + Lx z))) ∗ (N₀ ∘ Lε)`. -/
theorem modelLaw_eq_conv {d m : ℕ} (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx : EuclideanSpace ℝ (Fin d))
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) :
    modelLaw Aop μx Lx Lε
      = ((stdGaussian (EuclideanSpace ℝ (Fin d))).map (fun z => Aop (μx + Lx z)))
          ∗ ((stdGaussian (E m)).map ⇑Lε) := by
  have hfun : (fun p : EuclideanSpace ℝ (Fin d) × E m => Aop (μx + Lx p.1) + Lε p.2)
      = fun p => Aop μx + (Aop ∘L Lx) p.1 + Lε p.2 := by
    funext p; rw [map_add]; rfl
  have hfun2 : (fun z : EuclideanSpace ℝ (Fin d) => Aop (μx + Lx z))
      = fun z => Aop μx + (Aop ∘L Lx) z := by
    funext z; rw [map_add]; rfl
  rw [modelLaw, hfun, hfun2, signalNoise_eq_conv]

/-- **(B1) The data-law representation at the operator.** The record `y = A x + ε` of a Gaussian
truth `x ∼ N(μx, Lx Lxᵀ)` under independent noise `ε ∼ N(0, Lε Lεᵀ)` has exactly the pushforward
law `N₀ ∘ (z ↦ Aμx + Ly z) = N(Aμx, S_y)` that `NearNullOperator.lean` takes as given, for any
square root `Ly` of `S_y = (A Lx)(A Lx)ᵀ + Lε Lεᵀ` (hypothesis `hcov`). This discharges the
docstring caveat of `NearNullOperator.lean`. -/
theorem modelLaw_repr {d m : ℕ} (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (μx : EuclideanSpace ℝ (Fin d))
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε Ly : (E m) →L[ℝ] (E m))
    (hcov : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint Ly ξ‖ ^ 2
      = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
        + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2) :
    modelLaw Aop μx Lx Lε = (stdGaussian (E m)).map (fun z => Aop μx + Ly z) := by
  have hfun : (fun p : EuclideanSpace ℝ (Fin d) × E m => Aop (μx + Lx p.1) + Lε p.2)
      = fun p => Aop μx + (Aop ∘L Lx) p.1 + Lε p.2 := by
    funext p; rw [map_add]; rfl
  rw [modelLaw, hfun]
  exact signal_noise_law (Aop μx) (Aop ∘L Lx) Lε Ly hcov

/-! ### (3) (B2) The precision bridging: `‖Ly⁻¹ u‖² = ⟪u, S_y⁻¹ u⟫` -/

/-- The data covariance `S_y = Ly Lyᵀ` as a continuous linear map. -/
noncomputable def dataCovariance {m : ℕ} (Ly : (E m) ≃L[ℝ] (E m)) : (E m) →L[ℝ] (E m) :=
  (↑Ly : (E m) →L[ℝ] (E m))
    ∘L ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m))

/-- The data precision `S_y⁻¹ = (Ly⁻¹)ᵀ Ly⁻¹` as a continuous linear map. -/
noncomputable def dataPrecision {m : ℕ} (Ly : (E m) ≃L[ℝ] (E m)) : (E m) →L[ℝ] (E m) :=
  ContinuousLinearMap.adjoint (↑Ly.symm : (E m) →L[ℝ] (E m))
    ∘L (↑Ly.symm : (E m) →L[ℝ] (E m))

/-- `Lyᵀ ∘ (Ly⁻¹)ᵀ = 1`: the adjoints of an equivalence and its inverse cancel. -/
private lemma adjoint_adjoint_symm_cancel {m : ℕ} (Ly : (E m) ≃L[ℝ] (E m)) :
    ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m))
        ∘L ContinuousLinearMap.adjoint (↑Ly.symm : (E m) →L[ℝ] (E m))
      = ContinuousLinearMap.id ℝ (E m) := by
  rw [← ContinuousLinearMap.adjoint_comp]
  have h : (↑Ly.symm : (E m) →L[ℝ] (E m)) ∘L (↑Ly : (E m) →L[ℝ] (E m))
      = ContinuousLinearMap.id ℝ (E m) := by
    ext x; simp
  rw [h, ContinuousLinearMap.adjoint_id]

/-- `(Ly⁻¹)ᵀ ∘ Lyᵀ = 1`. -/
private lemma adjoint_symm_adjoint_cancel {m : ℕ} (Ly : (E m) ≃L[ℝ] (E m)) :
    ContinuousLinearMap.adjoint (↑Ly.symm : (E m) →L[ℝ] (E m))
        ∘L ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m))
      = ContinuousLinearMap.id ℝ (E m) := by
  rw [← ContinuousLinearMap.adjoint_comp]
  have h : (↑Ly : (E m) →L[ℝ] (E m)) ∘L (↑Ly.symm : (E m) →L[ℝ] (E m))
      = ContinuousLinearMap.id ℝ (E m) := by
    ext x; simp
  rw [h, ContinuousLinearMap.adjoint_id]

/-- **`S_y ∘ S_y⁻¹ = 1`**: `dataPrecision` is a right inverse of `dataCovariance`. -/
theorem dataCovariance_comp_dataPrecision {m : ℕ} (Ly : (E m) ≃L[ℝ] (E m)) :
    dataCovariance Ly ∘L dataPrecision Ly = ContinuousLinearMap.id ℝ (E m) := by
  rw [dataCovariance, dataPrecision, ContinuousLinearMap.comp_assoc,
    ← ContinuousLinearMap.comp_assoc
      (ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m))),
    adjoint_adjoint_symm_cancel, ContinuousLinearMap.id_comp]
  ext x; simp

/-- **`S_y⁻¹ ∘ S_y = 1`**: `dataPrecision` is a left inverse of `dataCovariance`. -/
theorem dataPrecision_comp_dataCovariance {m : ℕ} (Ly : (E m) ≃L[ℝ] (E m)) :
    dataPrecision Ly ∘L dataCovariance Ly = ContinuousLinearMap.id ℝ (E m) := by
  rw [dataPrecision, dataCovariance, ContinuousLinearMap.comp_assoc,
    ← ContinuousLinearMap.comp_assoc ((↑Ly.symm : (E m) →L[ℝ] (E m))),
    show (↑Ly.symm : (E m) →L[ℝ] (E m)) ∘L (↑Ly : (E m) →L[ℝ] (E m))
        = ContinuousLinearMap.id ℝ (E m) from by ext x; simp,
    ContinuousLinearMap.id_comp, adjoint_symm_adjoint_cancel]

/-- **Uniqueness of the precision:** any left inverse of `S_y` is `dataPrecision Ly` — so
"`S_y⁻¹`" below is unambiguous. -/
theorem dataPrecision_unique {m : ℕ} (Ly : (E m) ≃L[ℝ] (E m)) (B : (E m) →L[ℝ] (E m))
    (h : B ∘L dataCovariance Ly = ContinuousLinearMap.id ℝ (E m)) :
    B = dataPrecision Ly :=
  calc B = B ∘L ContinuousLinearMap.id ℝ (E m) := (ContinuousLinearMap.comp_id B).symm
    _ = B ∘L (dataCovariance Ly ∘L dataPrecision Ly) := by
        rw [dataCovariance_comp_dataPrecision]
    _ = (B ∘L dataCovariance Ly) ∘L dataPrecision Ly :=
        (ContinuousLinearMap.comp_assoc _ _ _).symm
    _ = ContinuousLinearMap.id ℝ (E m) ∘L dataPrecision Ly := by rw [h]
    _ = dataPrecision Ly := ContinuousLinearMap.id_comp _

/-- The covariance quadratic form: `⟪ξ, S_y ξ⟫ = ‖Lyᵀ ξ‖²` — the link between `dataCovariance` and
the variational hypothesis `hcov` of `modelLaw_repr`. -/
theorem inner_dataCovariance_self {m : ℕ} (Ly : (E m) ≃L[ℝ] (E m)) (ξ : E m) :
    inner ℝ ξ (dataCovariance Ly ξ)
      = ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2 := by
  rw [dataCovariance, ContinuousLinearMap.comp_apply,
    ← ContinuousLinearMap.adjoint_inner_left]
  exact real_inner_self_eq_norm_sq _

/-- **(B2) The β identification: `‖Ly⁻¹ u‖² = ⟪u, S_y⁻¹ u⟫`.** The whitened norms in
`NearNullOperator`'s closed forms are exactly the manuscript's quadratic forms in the data
precision. -/
theorem norm_symm_sq_eq_inner_dataPrecision {m : ℕ} (Ly : (E m) ≃L[ℝ] (E m)) (u : E m) :
    ‖Ly.symm u‖ ^ 2 = inner ℝ u (dataPrecision Ly u) := by
  rw [dataPrecision, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.adjoint_inner_right, real_inner_self_eq_norm_sq]
  rfl

/-- **(B2) Mean misreport χ² in precision form:** `χ² = exp(w²·(Av)ᵀS_y⁻¹(Av)) − 1` — the
manuscript's closed form verbatim, with `S_y⁻¹` the proved two-sided inverse of `Ly Lyᵀ`. -/
theorem operator_mean_misreport_chi2_precision {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) :
    NearNull.chi2
        ((stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
      = Real.exp (w ^ 2 * inner ℝ (Aop v) (dataPrecision Ly (Aop v))) - 1 := by
  rw [operator_mean_misreport_chi2 Aop μx v w Ly, norm_symm_sq_eq_inner_dataPrecision]

/-- **(B2) Spread misreport χ² in precision form:** `χ² = (1+β)/√(1+2β) − 1` with
`β = w·(Av)ᵀS_y⁻¹(Av)` — the manuscript's closed form verbatim. -/
theorem operator_spread_misreport_chi2_precision {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2) :
    NearNull.chi2
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly x))
        ((stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)))
      = (1 + w * inner ℝ (Aop v) (dataPrecision Ly (Aop v)))
          / Real.sqrt (1 + 2 * (w * inner ℝ (Aop v) (dataPrecision Ly (Aop v)))) - 1 := by
  rw [operator_spread_misreport_chi2 Aop μx v hw Ly M hM,
    norm_symm_sq_eq_inner_dataPrecision]

/-! ### (4) Covariance bookkeeping for the spread misreport on the model laws -/

/-- **Existence of the rank-one covariance square root, general dimension.** Same construction as
`NearNull.exists_rank1_sqrt` (which is stated on `E m`, dimension `m+1`); reproved on
`EuclideanSpace ℝ (Fin d)` for the truth space. -/
theorem exists_rank1_sqrt_x {d : ℕ} (u : EuclideanSpace ℝ (Fin d)) :
    ∃ M : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d),
      ∀ ξ : EuclideanSpace ℝ (Fin d),
        ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2 = ‖ξ‖ ^ 2 + (inner ℝ u ξ) ^ 2 := by
  classical
  rcases eq_or_ne u 0 with hu | hu
  · refine ⟨ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin d)), fun ξ => ?_⟩
    rw [ContinuousLinearMap.adjoint_id, ContinuousLinearMap.id_apply, hu]
    simp
  · have hnu : (0 : ℝ) < ‖u‖ ^ 2 := by
      have h := norm_pos_iff.mpr hu
      positivity
    set c : ℝ := (Real.sqrt (1 + ‖u‖ ^ 2) - 1) / ‖u‖ ^ 2 with hc
    set M : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d) :=
      ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin d))
        + c • ((innerSL ℝ u).smulRight u) with hMdef
    have hMsa : ContinuousLinearMap.adjoint M = M := by
      have h : M = ContinuousLinearMap.adjoint M := by
        rw [ContinuousLinearMap.eq_adjoint_iff]
        intro x y
        simp only [hMdef, _root_.add_apply, ContinuousLinearMap.id_apply,
          _root_.smul_apply, ContinuousLinearMap.smulRight_apply,
          innerSL_apply_apply]
        simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right]
        rw [real_inner_comm x u]
        ring
      exact h.symm
    have hM_apply : ∀ x, M x = x + (c * inner ℝ u x) • u := by
      intro x
      simp only [hMdef, _root_.add_apply, ContinuousLinearMap.id_apply,
        _root_.smul_apply, ContinuousLinearMap.smulRight_apply,
        innerSL_apply_apply]
      rw [smul_smul]
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

/-- **The bumped truth covariance is realizable** (`hbump` is satisfiable): for invertible `Lx` and
every `v`, `w ≥ 0`, some `Lx'` has `Lx' Lx'ᵀ = Lx Lxᵀ + w v vᵀ` (in quadratic-form terms). Witness:
`Lx' = Lx ∘ Mx` with `Mx Mxᵀ = I + w (Lx⁻¹v)(Lx⁻¹v)ᵀ` from `exists_rank1_sqrt_x`. -/
theorem exists_bump_sqrt {d : ℕ}
    (Lx : EuclideanSpace ℝ (Fin d) ≃L[ℝ] EuclideanSpace ℝ (Fin d))
    (v : EuclideanSpace ℝ (Fin d)) {w : ℝ} (hw : 0 ≤ w) :
    ∃ Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d),
      ∀ η : EuclideanSpace ℝ (Fin d),
        ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
          = ‖ContinuousLinearMap.adjoint (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ]
              EuclideanSpace ℝ (Fin d)) η‖ ^ 2 + w * (inner ℝ v η) ^ 2 := by
  obtain ⟨Mx, hMx⟩ := exists_rank1_sqrt_x (Real.sqrt w • Lx.symm v)
  refine ⟨(↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) ∘L Mx,
    fun η => ?_⟩
  rw [ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_apply, hMx,
    real_inner_smul_left, ContinuousLinearMap.adjoint_inner_right]
  have hsymm : (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
      (Lx.symm v) = v := by simp
  rw [hsymm, mul_pow, Real.sq_sqrt hw]

/-- **Covariance transfer for the spread misreport.** If `Ly Lyᵀ = (A Lx)(A Lx)ᵀ + Lε Lεᵀ`
(`hcov`), the truth covariance is bumped to `Lx' Lx'ᵀ = Lx Lxᵀ + w v vᵀ` (`hbump`), and `M` is the
rank-one root `M Mᵀ = I + w (Ly⁻¹Av)(Ly⁻¹Av)ᵀ` (`hM`), then `Ly ∘ M` is a square root of the bumped
data covariance: `(Ly M)(Ly M)ᵀ = (A Lx')(A Lx')ᵀ + Lε Lεᵀ` — i.e. `S_y + w (Av)(Av)ᵀ`. -/
theorem bump_cov_transfer {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w)
    (Lx Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hcov : ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2)
    (hbump : ∀ η : EuclideanSpace ℝ (Fin d),
      ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
        = ‖ContinuousLinearMap.adjoint Lx η‖ ^ 2 + w * (inner ℝ v η) ^ 2)
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2) :
    ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint ((↑Ly : (E m) →L[ℝ] (E m)) ∘L M) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx') ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2 := by
  intro ξ
  have hinner : inner ℝ (Real.sqrt w • Ly.symm (Aop v))
        (ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ)
      = Real.sqrt w * inner ℝ (Aop v) ξ := by
    rw [real_inner_smul_left, ContinuousLinearMap.adjoint_inner_right]
    have hsymm : (↑Ly : (E m) →L[ℝ] (E m)) (Ly.symm (Aop v)) = Aop v := by simp
    rw [hsymm]
  have hAv : inner ℝ v (ContinuousLinearMap.adjoint Aop ξ) = inner ℝ (Aop v) ξ :=
    ContinuousLinearMap.adjoint_inner_right Aop v ξ
  rw [ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_apply, hM, hcov, hinner,
    show ContinuousLinearMap.adjoint (Aop ∘L Lx') ξ
        = ContinuousLinearMap.adjoint Lx' (ContinuousLinearMap.adjoint Aop ξ) from by
      rw [ContinuousLinearMap.adjoint_comp]; rfl,
    hbump, hAv,
    show ContinuousLinearMap.adjoint Lx (ContinuousLinearMap.adjoint Aop ξ)
        = ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ from by
      rw [ContinuousLinearMap.adjoint_comp]; rfl,
    mul_pow, Real.sq_sqrt hw]
  ring

/-! ### (5) `p:nearnull`(ii) on the physical model laws, end to end -/

/-- **Mean misreport, end to end on the record `y = A x + ε`.** Two Gaussian truths differing by
`w·v` in the mean, same covariance, same independent noise: the χ² of their data laws is
`exp(w²‖Ly⁻¹(Av)‖²) − 1`. -/
theorem model_mean_misreport_chi2 {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) (Ly : (E m) ≃L[ℝ] (E m))
    (hcov : ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2) :
    NearNull.chi2 (modelLaw Aop (μx + w • v) Lx Lε) (modelLaw Aop μx Lx Lε)
      = Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) - 1 := by
  have h₁ : modelLaw Aop (μx + w • v) Lx Lε
      = (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x) := by
    rw [modelLaw_repr Aop (μx + w • v) Lx Lε (↑Ly) hcov]
    congr 1
    funext z
    simp [map_add, map_smul]
  have h₂ : modelLaw Aop μx Lx Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly x) :=
    modelLaw_repr Aop μx Lx Lε (↑Ly) hcov
  rw [h₁, h₂]
  exact operator_mean_misreport_chi2 Aop μx v w Ly

/-- **Mean misreport undetectability, end to end:** every test between the two record laws has
summed type-I + type-II error at least `1 − √(exp(w²‖Ly⁻¹(Av)‖²) − 1)`; at `Av = 0` the bound
is `1`. -/
theorem model_mean_misreport_undetectable {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) (Ly : (E m) ≃L[ℝ] (E m))
    (hcov : ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2)
    {A : Set (E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt (Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) - 1)
      ≤ (modelLaw Aop (μx + w • v) Lx Lε).real A + (modelLaw Aop μx Lx Lε).real Aᶜ := by
  have h₁ : modelLaw Aop (μx + w • v) Lx Lε
      = (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x) := by
    rw [modelLaw_repr Aop (μx + w • v) Lx Lε (↑Ly) hcov]
    congr 1
    funext z
    simp [map_add, map_smul]
  have h₂ : modelLaw Aop μx Lx Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly x) :=
    modelLaw_repr Aop μx Lx Lε (↑Ly) hcov
  rw [h₁, h₂]
  exact operator_mean_misreport_undetectable Aop μx v w Ly hA

/-- **Spread misreport, end to end on the record `y = A x + ε`.** Two Gaussian truths with the same
mean whose covariances differ by `w v vᵀ` (realized by `Lx'`, satisfiable by `exists_bump_sqrt`),
same independent noise: the χ² of their data laws is `(1+β)/√(1+2β) − 1`, `β = w‖Ly⁻¹(Av)‖²`
(`= w (Av)ᵀ S_y⁻¹ (Av)` by `norm_symm_sq_eq_inner_dataPrecision`). -/
theorem model_spread_misreport_chi2 {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w)
    (Lx Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hcov : ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2)
    (hbump : ∀ η : EuclideanSpace ℝ (Fin d),
      ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
        = ‖ContinuousLinearMap.adjoint Lx η‖ ^ 2 + w * (inner ℝ v η) ^ 2)
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2) :
    NearNull.chi2 (modelLaw Aop μx Lx Lε) (modelLaw Aop μx Lx' Lε)
      = (1 + w * ‖Ly.symm (Aop v)‖ ^ 2)
          / Real.sqrt (1 + 2 * (w * ‖Ly.symm (Aop v)‖ ^ 2)) - 1 := by
  have h₁ : modelLaw Aop μx Lx Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly x) :=
    modelLaw_repr Aop μx Lx Lε (↑Ly) hcov
  have h₂ : modelLaw Aop μx Lx' Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)) :=
    modelLaw_repr Aop μx Lx' Lε ((↑Ly : (E m) →L[ℝ] (E m)) ∘L M)
      (bump_cov_transfer Aop v hw Lx Lx' Lε Ly M hcov hbump hM)
  rw [h₁, h₂]
  exact operator_spread_misreport_chi2 Aop μx v hw Ly M hM

/-- **Spread misreport undetectability, end to end** — `p:nearnull`(ii) per survey on the physical
record laws: every test has summed error at least `1 − √((1+β)/√(1+2β) − 1)`,
`β = w‖Ly⁻¹(Av)‖²`; at `Av = 0` the bound is `1`. -/
theorem model_spread_misreport_undetectable {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w)
    (Lx Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hcov : ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2)
    (hbump : ∀ η : EuclideanSpace ℝ (Fin d),
      ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
        = ‖ContinuousLinearMap.adjoint Lx η‖ ^ 2 + w * (inner ℝ v η) ^ 2)
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    {A : Set (E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt ((1 + w * ‖Ly.symm (Aop v)‖ ^ 2)
          / Real.sqrt (1 + 2 * (w * ‖Ly.symm (Aop v)‖ ^ 2)) - 1)
      ≤ (modelLaw Aop μx Lx Lε).real A + (modelLaw Aop μx Lx' Lε).real Aᶜ := by
  have h₁ : modelLaw Aop μx Lx Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly x) :=
    modelLaw_repr Aop μx Lx Lε (↑Ly) hcov
  have h₂ : modelLaw Aop μx Lx' Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)) :=
    modelLaw_repr Aop μx Lx' Lε ((↑Ly : (E m) →L[ℝ] (E m)) ∘L M)
      (bump_cov_transfer Aop v hw Lx Lx' Lε Ly M hcov hbump hM)
  rw [h₁, h₂]
  exact operator_spread_misreport_undetectable Aop μx v hw Ly M hM hA

/-- **`n`-survey mean misreport on the record laws:** summed error at least
`1 − √(exp(β)ⁿ − 1)`, `β = w²‖Ly⁻¹(Av)‖²`. -/
theorem model_mean_nsurvey_undetectable {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) (Ly : (E m) ≃L[ℝ] (E m))
    (hcov : ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2)
    (n : ℕ) {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt ((Real.exp (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2)) ^ n - 1)
      ≤ (Measure.pi fun _ : Fin n => modelLaw Aop (μx + w • v) Lx Lε).real A
        + (Measure.pi fun _ : Fin n => modelLaw Aop μx Lx Lε).real Aᶜ := by
  have h₁ : modelLaw Aop (μx + w • v) Lx Lε
      = (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x) := by
    rw [modelLaw_repr Aop (μx + w • v) Lx Lε (↑Ly) hcov]
    congr 1
    funext z
    simp [map_add, map_smul]
  have h₂ : modelLaw Aop μx Lx Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly x) :=
    modelLaw_repr Aop μx Lx Lε (↑Ly) hcov
  rw [show (fun _ : Fin n => modelLaw Aop (μx + w • v) Lx Lε)
      = (fun _ : Fin n =>
          (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)) from
    funext fun _ => h₁,
    show (fun _ : Fin n => modelLaw Aop μx Lx Lε)
      = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx + Ly x)) from
    funext fun _ => h₂]
  exact operator_mean_nsurvey_undetectable Aop μx v w Ly n hA

/-- **`n`-survey spread misreport on the record laws** — `p:nearnull`(ii)'s `n`-survey assembly,
end to end: summed error at least `1 − √(((1+β)/√(1+2β))ⁿ − 1)`, `β = w‖Ly⁻¹(Av)‖²`. -/
theorem model_spread_nsurvey_undetectable {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w)
    (Lx Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hcov : ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2)
    (hbump : ∀ η : EuclideanSpace ℝ (Fin d),
      ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
        = ‖ContinuousLinearMap.adjoint Lx η‖ ^ 2 + w * (inner ℝ v η) ^ 2)
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    (n : ℕ) {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 - Real.sqrt (((1 + w * ‖Ly.symm (Aop v)‖ ^ 2)
          / Real.sqrt (1 + 2 * (w * ‖Ly.symm (Aop v)‖ ^ 2))) ^ n - 1)
      ≤ (Measure.pi fun _ : Fin n => modelLaw Aop μx Lx Lε).real A
        + (Measure.pi fun _ : Fin n => modelLaw Aop μx Lx' Lε).real Aᶜ := by
  have h₁ : modelLaw Aop μx Lx Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly x) :=
    modelLaw_repr Aop μx Lx Lε (↑Ly) hcov
  have h₂ : modelLaw Aop μx Lx' Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)) :=
    modelLaw_repr Aop μx Lx' Lε ((↑Ly : (E m) →L[ℝ] (E m)) ∘L M)
      (bump_cov_transfer Aop v hw Lx Lx' Lε Ly M hcov hbump hM)
  rw [show (fun _ : Fin n => modelLaw Aop μx Lx Lε)
      = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx + Ly x)) from
    funext fun _ => h₁,
    show (fun _ : Fin n => modelLaw Aop μx Lx' Lε)
      = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))) from
    funext fun _ => h₂]
  exact operator_spread_nsurvey_undetectable Aop μx v hw Ly M hM n hA

/-! ### (6) (B3) The certified crossover constant -/

/-- **The χ² comparison** `(1+β)/√(1+2β) − 1 ≤ β²/2` for all `β ≥ 0` (hence `≤ β²` a fortiori) —
the algebraic heart of the crossover constant, replacing the paper's KL bound `KL ≤ t²/4`. -/
theorem chi2_spread_le_half_sq {β : ℝ} (hβ : 0 ≤ β) :
    (1 + β) / Real.sqrt (1 + 2 * β) - 1 ≤ β ^ 2 / 2 := by
  have h2 : (0 : ℝ) < 1 + 2 * β := by linarith
  set s := Real.sqrt (1 + 2 * β) with hs
  have hs2 : s ^ 2 = 1 + 2 * β := Real.sq_sqrt h2.le
  have hs1 : 1 ≤ s := by
    have h := Real.sqrt_le_sqrt (show (1 : ℝ) ≤ 1 + 2 * β by linarith)
    rwa [Real.sqrt_one] at h
  have hs0 : (0 : ℝ) < s := lt_of_lt_of_le one_pos hs1
  have hsq : (1 + β) ^ 2 ≤ ((1 + β ^ 2 / 2) * s) ^ 2 := by
    have hexp : ((1 + β ^ 2 / 2) * s) ^ 2 = (1 + β ^ 2 / 2) ^ 2 * (1 + 2 * β) := by
      rw [mul_pow, hs2]
    rw [hexp]
    nlinarith [pow_nonneg hβ 3, pow_nonneg hβ 4, pow_nonneg hβ 5]
  have key : 1 + β ≤ (1 + β ^ 2 / 2) * s := by
    nlinarith [hsq, hs0, hβ, mul_pos (by positivity : (0 : ℝ) < 1 + β ^ 2 / 2) hs0]
  have hdiv : (1 + β) / s ≤ 1 + β ^ 2 / 2 := (div_le_iff₀ hs0).mpr key
  linarith

/-- `exp(1/5) < 5/4` — certified numerically via `exp 1 < 2.7182818286 < (5/4)⁵`. -/
private lemma exp_one_fifth_lt : Real.exp (1 / 5) < 5 / 4 := by
  have h5 : Real.exp (1 / 5) ^ 5 = Real.exp 1 := by
    rw [← Real.exp_nat_mul]
    norm_num
  have hlt : Real.exp (1 / 5) ^ 5 < (5 / 4 : ℝ) ^ 5 := by
    rw [h5]
    calc Real.exp 1 < 2.7182818286 := Real.exp_one_lt_d9
      _ < (5 / 4 : ℝ) ^ 5 := by norm_num
  by_contra hcon
  rw [not_lt] at hcon
  have h := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 5 / 4) hcon 5
  linarith

/-- **The certified spread crossover inequality:** if `n·β² < 2/5` then
`((1+β)/√(1+2β))ⁿ < 5/4` — the χ²-route substitute for the paper's Pinsker step, in the same
variable. (The chain: `1 + χ²(β) ≤ 1 + β²/2 ≤ exp(β²/2)`, so the `n`-fold product is below
`exp(n β²/2) < exp(1/5) < 5/4`.) -/
theorem chi2_pow_lt_of_mul_lt {β : ℝ} (hβ : 0 ≤ β) (n : ℕ)
    (hn : (n : ℝ) * β ^ 2 < 2 / 5) :
    ((1 + β) / Real.sqrt (1 + 2 * β)) ^ n < 5 / 4 := by
  have hbase0 : (0 : ℝ) ≤ (1 + β) / Real.sqrt (1 + 2 * β) := by positivity
  have hbase : (1 + β) / Real.sqrt (1 + 2 * β) ≤ 1 + β ^ 2 / 2 := by
    have := chi2_spread_le_half_sq hβ
    linarith
  have hexp : (1 : ℝ) + β ^ 2 / 2 ≤ Real.exp (β ^ 2 / 2) := by
    have := Real.add_one_le_exp (β ^ 2 / 2)
    linarith
  calc ((1 + β) / Real.sqrt (1 + 2 * β)) ^ n
      ≤ (1 + β ^ 2 / 2) ^ n := pow_le_pow_left₀ hbase0 hbase n
    _ ≤ Real.exp (β ^ 2 / 2) ^ n := pow_le_pow_left₀ (by positivity) hexp n
    _ = Real.exp ((n : ℝ) * (β ^ 2 / 2)) := (Real.exp_nat_mul _ n).symm
    _ < Real.exp (1 / 5) := by
        apply Real.exp_lt_exp.mpr
        linarith
    _ < 5 / 4 := exp_one_fifth_lt

/-- **The certified mean crossover inequality:** if `n·β < 1/5` then `exp(β)ⁿ < 5/4`. -/
theorem exp_pow_lt_of_mul_lt {β : ℝ} (n : ℕ) (hn : (n : ℝ) * β < 1 / 5) :
    Real.exp β ^ n < 5 / 4 :=
  calc Real.exp β ^ n = Real.exp ((n : ℝ) * β) := (Real.exp_nat_mul _ n).symm
    _ < Real.exp (1 / 5) := Real.exp_lt_exp.mpr hn
    _ < 5 / 4 := exp_one_fifth_lt

/-- **Certified operator-level spread crossover — `eq:nearnull-cross` with an explicit machine
constant.** With `t = w‖Ly⁻¹(Av)‖² (= w (Av)ᵀ S_y⁻¹ (Av))`: whenever `n·t² < 2/5`, every test on
`n` surveys of the spread misreport has summed type-I + type-II error strictly above `1/2`. The
paper's Pinsker-route constant is `n < 2/t²`; the certified χ²-route constant is `n < (2/5)/t²` —
same `1/t²` scaling, explicit constant, no Pinsker needed. -/
theorem operator_spread_crossover_certified {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    (n : ℕ) (hn : (n : ℝ) * (w * ‖Ly.symm (Aop v)‖ ^ 2) ^ 2 < 2 / 5)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))).real Aᶜ :=
  operator_spread_crossover Aop μx v hw Ly M hM n
    (chi2_pow_lt_of_mul_lt (mul_nonneg hw (sq_nonneg _)) n hn) hA

/-- **Certified operator-level mean crossover:** with `β = w²‖Ly⁻¹(Av)‖²`, whenever `n·β < 1/5`
every test on `n` surveys of the mean misreport has summed error strictly above `1/2`. -/
theorem operator_mean_crossover_certified {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Ly : (E m) ≃L[ℝ] (E m)) (n : ℕ)
    (hn : (n : ℝ) * (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) < 1 / 5)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)).real A
        + (Measure.pi fun _ : Fin n =>
            (stdGaussian (E m)).map (fun x => Aop μx + Ly x)).real Aᶜ :=
  operator_mean_crossover Aop μx v w Ly n (exp_pow_lt_of_mul_lt n hn) hA

/-- **The certified crossover on the physical record laws — the module's final assembly.** For the
record `y = A x + ε` of two Gaussian truths differing by the spread misreport `Σ⋆ ↦ Σ⋆ + w v vᵀ`:
whenever `n · t² < 2/5` with `t = w‖Ly⁻¹(Av)‖² = w (Av)ᵀ S_y⁻¹ (Av)`, every test on `n`
independent surveys has summed type-I + type-II error strictly above `1/2` — `eq:nearnull-cross`
with the certified machine constant `2/5`, stated on the model itself. -/
theorem model_spread_crossover_certified {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) {w : ℝ}
    (hw : 0 ≤ w)
    (Lx Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) (Ly : (E m) ≃L[ℝ] (E m)) (M : (E m) →L[ℝ] (E m))
    (hcov : ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2)
    (hbump : ∀ η : EuclideanSpace ℝ (Fin d),
      ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
        = ‖ContinuousLinearMap.adjoint Lx η‖ ^ 2 + w * (inner ℝ v η) ^ 2)
    (hM : ∀ ξ : E m, ‖ContinuousLinearMap.adjoint M ξ‖ ^ 2
      = ‖ξ‖ ^ 2 + (inner ℝ (Real.sqrt w • Ly.symm (Aop v)) ξ) ^ 2)
    (n : ℕ) (hn : (n : ℝ) * (w * ‖Ly.symm (Aop v)‖ ^ 2) ^ 2 < 2 / 5)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n => modelLaw Aop μx Lx Lε).real A
        + (Measure.pi fun _ : Fin n => modelLaw Aop μx Lx' Lε).real Aᶜ := by
  have h₁ : modelLaw Aop μx Lx Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly x) :=
    modelLaw_repr Aop μx Lx Lε (↑Ly) hcov
  have h₂ : modelLaw Aop μx Lx' Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x)) :=
    modelLaw_repr Aop μx Lx' Lε ((↑Ly : (E m) →L[ℝ] (E m)) ∘L M)
      (bump_cov_transfer Aop v hw Lx Lx' Lε Ly M hcov hbump hM)
  rw [show (fun _ : Fin n => modelLaw Aop μx Lx Lε)
      = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx + Ly x)) from
    funext fun _ => h₁,
    show (fun _ : Fin n => modelLaw Aop μx Lx' Lε)
      = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx + Ly (M x))) from
    funext fun _ => h₂]
  exact operator_spread_crossover_certified Aop μx v hw Ly M hM n hn hA

/-- **The certified mean crossover on the physical record laws:** with `β = w²‖Ly⁻¹(Av)‖²`,
whenever `n·β < 1/5` every test on `n` surveys of the mean misreport (two truths differing by
`w·v` in the mean) has summed error strictly above `1/2`. -/
theorem model_mean_crossover_certified {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) →L[ℝ] (E m)) (Ly : (E m) ≃L[ℝ] (E m))
    (hcov : ∀ ξ : E m,
      ‖ContinuousLinearMap.adjoint (↑Ly : (E m) →L[ℝ] (E m)) ξ‖ ^ 2
        = ‖ContinuousLinearMap.adjoint (Aop ∘L Lx) ξ‖ ^ 2
          + ‖ContinuousLinearMap.adjoint Lε ξ‖ ^ 2)
    (n : ℕ) (hn : (n : ℝ) * (w ^ 2 * ‖Ly.symm (Aop v)‖ ^ 2) < 1 / 5)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n => modelLaw Aop (μx + w • v) Lx Lε).real A
        + (Measure.pi fun _ : Fin n => modelLaw Aop μx Lx Lε).real Aᶜ := by
  have h₁ : modelLaw Aop (μx + w • v) Lx Lε
      = (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x) := by
    rw [modelLaw_repr Aop (μx + w • v) Lx Lε (↑Ly) hcov]
    congr 1
    funext z
    simp [map_add, map_smul]
  have h₂ : modelLaw Aop μx Lx Lε
      = (stdGaussian (E m)).map (fun x => Aop μx + Ly x) :=
    modelLaw_repr Aop μx Lx Lε (↑Ly) hcov
  rw [show (fun _ : Fin n => modelLaw Aop (μx + w • v) Lx Lε)
      = (fun _ : Fin n =>
          (stdGaussian (E m)).map (fun x => (Aop μx + w • Aop v) + Ly x)) from
    funext fun _ => h₁,
    show (fun _ : Fin n => modelLaw Aop μx Lx Lε)
      = (fun _ : Fin n => (stdGaussian (E m)).map (fun x => Aop μx + Ly x)) from
    funext fun _ => h₂]
  exact operator_mean_crossover_certified Aop μx v w Ly n hn hA

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.signal_noise_law
#print axioms NearNull.modelLaw_eq_conv
#print axioms NearNull.modelLaw_repr
#print axioms NearNull.dataCovariance_comp_dataPrecision
#print axioms NearNull.dataPrecision_comp_dataCovariance
#print axioms NearNull.dataPrecision_unique
#print axioms NearNull.inner_dataCovariance_self
#print axioms NearNull.norm_symm_sq_eq_inner_dataPrecision
#print axioms NearNull.operator_mean_misreport_chi2_precision
#print axioms NearNull.operator_spread_misreport_chi2_precision
#print axioms NearNull.exists_rank1_sqrt_x
#print axioms NearNull.exists_bump_sqrt
#print axioms NearNull.bump_cov_transfer
#print axioms NearNull.model_mean_misreport_chi2
#print axioms NearNull.model_mean_misreport_undetectable
#print axioms NearNull.model_spread_misreport_chi2
#print axioms NearNull.model_spread_misreport_undetectable
#print axioms NearNull.model_mean_nsurvey_undetectable
#print axioms NearNull.model_spread_nsurvey_undetectable
#print axioms NearNull.chi2_spread_le_half_sq
#print axioms NearNull.chi2_pow_lt_of_mul_lt
#print axioms NearNull.exp_pow_lt_of_mul_lt
#print axioms NearNull.operator_spread_crossover_certified
#print axioms NearNull.operator_mean_crossover_certified
#print axioms NearNull.model_spread_crossover_certified
#print axioms NearNull.model_mean_crossover_certified
