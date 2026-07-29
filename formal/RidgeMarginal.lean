/-
# `p:mapnearnull` (iii) and the `E‖d‖²` bound of (ii): the ridge marginal, the coupled
counterexample, and the objective comparison

Two leftover computations of Proposition `p:mapnearnull` (manuscript ~line 807), machine-checked.

## (A) The ridge marginal bound — part (iii)'s positive half

In whitened coordinates (`Ã = Γ^{-1/2}A`, data covariance `S̃_y`, both *local to this
proposition*), the ridge estimator is `x̂(ỹ) = (ÃᵀÃ + λI)⁻¹ Ãᵀ ỹ`.  The blind readout
`vᵀx̂(ỹ) = ⟪c_v, ỹ⟫` is the linear image of the whitened data at the coefficient vector
`c_v = ((ÃᵀÃ + λI)⁻¹ Ãᵀ)ᵀ v` (`ridge_readout`), so its variance is the quadratic form of the
data covariance at `c_v` — the house variational convention.  The chain is the paper's:

* `push_through` — `(ÃᵀÃ + λI)⁻¹ Ãᵀ = Ãᵀ (ÃÃᵀ + λI)⁻¹`, by multiplying
  `Ãᵀ(ÃÃᵀ + λI) = (ÃᵀÃ + λI)Ãᵀ` on both sides by the two inverses (both matrices `≻ 0`
  since `λ > 0`); hence `c_v = (ÃÃᵀ + λI)⁻¹ (Ãv)` (`ridgeCoeff_eq`).
* `gram_inv_sq_norm_le` — the operator-norm bound `‖(ÃÃᵀ + λI)⁻¹‖ ≤ 1/λ` in variational form:
  `‖(ÃÃᵀ+λI)⁻¹u‖² ≤ ‖u‖²/λ²`, from `λ‖w‖² ≤ ⟪w, (ÃÃᵀ+λI)w⟫` and the dot-product
  Cauchy–Schwarz (`NearNullDefinitional.dot_cauchy_schwarz`, itself `discrim_le_zero` — no
  spectral theorem anywhere).
* `ridge_marginal_variance_bound` — the capstone
  `⟪c_v, S̃_y c_v⟫ ≤ s ‖Ãv‖² / λ²` for any variational upper band `⟪w, S̃_y w⟫ ≤ s‖w‖²`
  (the `λ_max(S̃_y)` reading), i.e. the paper's `Var(vᵀx̂) ≤ λ_max(S̃_y)‖Ãv‖²/λ²`; with
  the exact-kernel endpoint `ridge_marginal_zero_of_kernel` (`Ãv = 0` gives variance exactly 0).
* `ridge_estimator_minimizes` — the estimator IS the minimizer of the whitened ridge objective
  `½‖ỹ − Ãx‖² + (λ/2)‖x‖²` (normal equation + exact quadratic expansion), anchoring the
  closed form to `p:mapnearnull`'s archive `x̂(y) = argmin`.

## (B) The coupled counterexample — part (iii)'s closing clause

A fully explicit two-dimensional witness that *no* marginal bound holds off alignment: data
row `A = (1 0)` (one measurement), blind direction `v = e₂ ∈ ker A` (`coupled_blind_direction`),
coupled quadratic penalty `½ xᵀPx`, `P = [[p, c], [c, q]]` with `c ≠ 0`.  The unique minimizer
of `½(y − x₁)² + ½ xᵀPx` is `x̂(y) = (qy/Δ, −cy/Δ)`, `Δ = (1+p)q − c²` (`coupled_minimizes`,
`coupled_minimizer_unique` — so every measurable selection is forced, `coupled_selection_forced`),
and the blind marginal is the deterministic image `vᵀx̂(y) = (−c/Δ)·y` with *nonzero*
coefficient: under `y ∼ N(0, s)` its law is exactly `N(0, (c/Δ)² s)` (`coupled_selection_law`,
via mathlib's `gaussianReal_map_const_mul`) with variance `(c/Δ)² s > 0`
(`coupled_marginal_variance`, `coupled_marginal_variance_pos`) — order one, bounded below by a
positive constant while `‖Av‖ = 0` (`coupled_marginal_order_one`).  Only the conditional
statement survives off alignment: the near-null form of the alignment dichotomy of `r:map-lg`.

## (C) The `E‖d‖²` objective comparison — the bound part (ii) states as a hypothesis

For the Gaussian data term `D(y, Ax) = ½‖y − Ax‖²_{Γ⁻¹}`, comparing the objective at any
minimizer `x̂(y)` with a fixed reference `x₀` gives the paper's display

    ‖y − Ax̂‖²_{Γ⁻¹} ≤ ‖y − Ax₀‖²_{Γ⁻¹} + 2(R(x₀) − inf R)          (`misfit_comparison`),

and the Gaussian gradient `d(y) = Γ⁻¹(y − Ax̂(y))` is controlled by the misfit:
`grad_sq_pointwise_bound` gives `‖d(y)‖² ≤ (2/γ²)‖y‖² + C(γ, x₀, R(x₀) − R_inf)` where
`γ ≤ λ_min(Γ)` variationally — constants containing **no reference to `v` or `‖Av‖`**, which
is exactly the paper's "uniformly as `‖Av‖ → 0`".  `expected_grad_sq_le` /
`expected_grad_sq_lt_top` integrate it: for any measurement law `μ` of finite second moment,
`E‖d‖² < ∞` — stated as a lower Lebesgue integral, so **no measurability of the minimizer
selection `x̂(·)` is assumed anywhere** (`lintegral_mono` needs none, and the majorant is
handled by `lintegral_const_mul'`/`lintegral_add_right` against constants).

## Honesty

No `sorry`/`admit`/`native_decide`.  Every hypothesis anywhere is one of the proposition's own
standing assumptions, stated variationally where spectral: `λ > 0` (the ridge constant),
`S̃_y ⪰ 0` with its upper band `s` (a covariance and the `λ_max` reading), `Γ ≻ 0` with its
lower band `γ` (the noise floor), the minimizer property of `x̂` (the archive's definition,
discharged *exactly* for the ridge case by `ridge_estimator_minimizes` and for the 2-D witness
by `coupled_minimizes`), the penalty lower bound `R ≥ R_inf`, the finite second moment of the
measurement law, and for the witness `q > 0`, `Δ > 0` (implied by `P ≻ 0`,
`coupledDelta_pos_of_posDef`) and `c ≠ 0` (the coupling).  Nothing else remains: no flagged
analytic gaps in this module.  `#print axioms` on every public result at the end.
-/
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Integral.Lebesgue.Add
import NearNullDefinitional

namespace RidgeMarginal

open Matrix MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

/-! ## Dot-product expansion helpers (free-variable form, so rewrites stay at the atoms) -/

section DotHelpers

variable {ι : Type*} [Fintype ι]

/-- `(r − w)·(r − w) = r·r − 2 r·w + w·w`. -/
theorem dot_expand_sub (r w : ι → ℝ) :
    (r - w) ⬝ᵥ (r - w) = r ⬝ᵥ r - 2 * (r ⬝ᵥ w) + w ⬝ᵥ w := by
  rw [sub_dotProduct, dotProduct_sub, dotProduct_sub, dotProduct_comm w r]
  ring

/-- `(a + b)·(a + b) = a·a + 2 a·b + b·b`. -/
theorem dot_expand_add (a b : ι → ℝ) :
    (a + b) ⬝ᵥ (a + b) = a ⬝ᵥ a + 2 * (a ⬝ᵥ b) + b ⬝ᵥ b := by
  rw [add_dotProduct, dotProduct_add, dotProduct_add, dotProduct_comm b a]
  ring

end DotHelpers

/-! ## (A) The ridge marginal: push-through, contraction, and the variance bound -/

section Ridge

variable {m n : Type*} [Fintype m] [DecidableEq m] [Fintype n] [DecidableEq n]

/-- The whitened ridge normal matrix `ÃᵀÃ + λI`. -/
def ridgeNormal (At : Matrix m n ℝ) (lam : ℝ) : Matrix n n ℝ :=
  Atᵀ * At + lam • 1

/-- The whitened ridge Gram matrix `ÃÃᵀ + λI`. -/
def ridgeGram (At : Matrix m n ℝ) (lam : ℝ) : Matrix m m ℝ :=
  At * Atᵀ + lam • 1

/-- The ridge estimator matrix `(ÃᵀÃ + λI)⁻¹ Ãᵀ`: the archive stores `x̂(ỹ) = ridgeEstimator *ᵥ ỹ`. -/
noncomputable def ridgeEstimator (At : Matrix m n ℝ) (lam : ℝ) : Matrix n m ℝ :=
  (ridgeNormal At lam)⁻¹ * Atᵀ

/-- The blind readout's coefficient vector: `vᵀx̂(ỹ) = ⟪ridgeCoeff, ỹ⟫` (`ridge_readout`), so the
variance of the blind marginal is the quadratic form of the data covariance at this vector. -/
noncomputable def ridgeCoeff (At : Matrix m n ℝ) (lam : ℝ) (v : n → ℝ) : m → ℝ :=
  (ridgeEstimator At lam)ᵀ *ᵥ v

omit [DecidableEq m] [DecidableEq n] in
/-- `ÃᵀÃ ⪰ 0` (real carrier). -/
theorem transpose_mul_self_posSemidef (At : Matrix m n ℝ) : (Atᵀ * At).PosSemidef := by
  have h := Matrix.posSemidef_conjTranspose_mul_self At
  rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at h

omit [DecidableEq m] [DecidableEq n] in
/-- `ÃÃᵀ ⪰ 0` (real carrier). -/
theorem self_mul_transpose_posSemidef (At : Matrix m n ℝ) : (At * Atᵀ).PosSemidef := by
  have h := Matrix.posSemidef_self_mul_conjTranspose At
  rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at h

omit [DecidableEq m] in
/-- `ÃᵀÃ + λI ≻ 0` for `λ > 0`. -/
theorem ridgeNormal_posDef (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) :
    (ridgeNormal At lam).PosDef :=
  Matrix.PosDef.posSemidef_add (transpose_mul_self_posSemidef At) (Matrix.PosDef.one.smul hlam)

omit [DecidableEq n] in
/-- `ÃÃᵀ + λI ≻ 0` for `λ > 0`. -/
theorem ridgeGram_posDef (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) :
    (ridgeGram At lam).PosDef :=
  Matrix.PosDef.posSemidef_add (self_mul_transpose_posSemidef At) (Matrix.PosDef.one.smul hlam)

/-- **The push-through identity.**  `(ÃᵀÃ + λI)⁻¹ Ãᵀ = Ãᵀ (ÃÃᵀ + λI)⁻¹` — from
`Ãᵀ(ÃÃᵀ + λI) = (ÃᵀÃ + λI)Ãᵀ`, multiplying by the two inverses (both sides `≻ 0`). -/
theorem push_through (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) :
    (ridgeNormal At lam)⁻¹ * Atᵀ = Atᵀ * (ridgeGram At lam)⁻¹ := by
  have hNdet : IsUnit (ridgeNormal At lam).det :=
    (Matrix.isUnit_iff_isUnit_det _).mp (ridgeNormal_posDef At hlam).isUnit
  have hGdet : IsUnit (ridgeGram At lam).det :=
    (Matrix.isUnit_iff_isUnit_det _).mp (ridgeGram_posDef At hlam).isUnit
  have key : Atᵀ * ridgeGram At lam = ridgeNormal At lam * Atᵀ := by
    simp only [ridgeGram, ridgeNormal, Matrix.mul_add, Matrix.add_mul, Matrix.mul_smul,
      Matrix.smul_mul, Matrix.mul_one, Matrix.one_mul, Matrix.mul_assoc]
  calc (ridgeNormal At lam)⁻¹ * Atᵀ
      = (ridgeNormal At lam)⁻¹ * (Atᵀ * ridgeGram At lam) * (ridgeGram At lam)⁻¹ := by
        rw [← Matrix.mul_assoc, Matrix.mul_nonsing_inv_cancel_right _ _ hGdet]
    _ = (ridgeNormal At lam)⁻¹ * (ridgeNormal At lam * Atᵀ) * (ridgeGram At lam)⁻¹ := by
        rw [key]
    _ = Atᵀ * (ridgeGram At lam)⁻¹ := by
        rw [Matrix.nonsing_inv_mul_cancel_left _ _ hNdet]

omit [DecidableEq m] in
/-- The estimator's transpose: `((ÃᵀÃ + λI)⁻¹ Ãᵀ)ᵀ = Ã (ÃᵀÃ + λI)⁻¹` (the normal matrix and
its inverse are symmetric). -/
theorem ridgeEstimator_transpose (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) :
    (ridgeEstimator At lam)ᵀ = At * (ridgeNormal At lam)⁻¹ := by
  simp only [ridgeEstimator]
  rw [Matrix.transpose_mul, Matrix.transpose_transpose,
    NearNullDefinitional.pd_inv_transpose_eq (ridgeNormal_posDef At hlam)]

/-- **The coefficient vector through the push-through.**  `c_v = (ÃÃᵀ + λI)⁻¹ (Ãv)`. -/
theorem ridgeCoeff_eq (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) (v : n → ℝ) :
    ridgeCoeff At lam v = (ridgeGram At lam)⁻¹ *ᵥ (At *ᵥ v) := by
  have h1 : At * (ridgeNormal At lam)⁻¹ = (ridgeGram At lam)⁻¹ * At := by
    have h := congrArg Matrix.transpose (push_through At hlam)
    rwa [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
      NearNullDefinitional.pd_inv_transpose_eq (ridgeNormal_posDef At hlam),
      NearNullDefinitional.pd_inv_transpose_eq (ridgeGram_posDef At hlam)] at h
  simp only [ridgeCoeff]
  rw [ridgeEstimator_transpose At hlam, h1, ← Matrix.mulVec_mulVec]

omit [DecidableEq m] in
/-- **The blind readout is the linear image of the whitened data at the coefficient vector**:
`vᵀx̂(ỹ) = ⟪c_v, ỹ⟫` for every `ỹ`.  This is what makes `⟪c_v, S̃_y c_v⟫` *the* variance of the
blind marginal in the house variational convention. -/
theorem ridge_readout (At : Matrix m n ℝ) (lam : ℝ) (v : n → ℝ) (y : m → ℝ) :
    v ⬝ᵥ (ridgeEstimator At lam *ᵥ y) = ridgeCoeff At lam v ⬝ᵥ y := by
  simp only [ridgeCoeff]
  rw [dotProduct_mulVec, Matrix.mulVec_transpose]

omit [DecidableEq n] in
/-- **The Gram-inverse contraction, variationally.**  `‖(ÃÃᵀ + λI)⁻¹ u‖² ≤ ‖u‖²/λ²` — the
paper's `‖(ÃÃᵀ + λI)⁻¹‖ ≤ 1/λ`, proved from `λ‖w‖² ≤ ⟪w, (ÃÃᵀ + λI)w⟫` and the dot-product
Cauchy–Schwarz; no eigenvalue API. -/
theorem gram_inv_sq_norm_le (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) (u : m → ℝ) :
    ((ridgeGram At lam)⁻¹ *ᵥ u) ⬝ᵥ ((ridgeGram At lam)⁻¹ *ᵥ u) ≤ (u ⬝ᵥ u) / lam ^ 2 := by
  have hGdet : IsUnit (ridgeGram At lam).det :=
    (Matrix.isUnit_iff_isUnit_det _).mp (ridgeGram_posDef At hlam).isUnit
  have hGc : ridgeGram At lam *ᵥ ((ridgeGram At lam)⁻¹ *ᵥ u) = u := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hGdet, Matrix.one_mulVec]
  have hexp : ((ridgeGram At lam)⁻¹ *ᵥ u) ⬝ᵥ (ridgeGram At lam *ᵥ ((ridgeGram At lam)⁻¹ *ᵥ u))
      = ((ridgeGram At lam)⁻¹ *ᵥ u) ⬝ᵥ ((At * Atᵀ) *ᵥ ((ridgeGram At lam)⁻¹ *ᵥ u))
        + lam * (((ridgeGram At lam)⁻¹ *ᵥ u) ⬝ᵥ ((ridgeGram At lam)⁻¹ *ᵥ u)) := by
    simp only [ridgeGram, Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec,
      dotProduct_add, dotProduct_smul, smul_eq_mul]
  have hpsd : 0 ≤ ((ridgeGram At lam)⁻¹ *ᵥ u) ⬝ᵥ ((At * Atᵀ) *ᵥ ((ridgeGram At lam)⁻¹ *ᵥ u)) :=
    NearNullDefinitional.psd_form_nonneg (self_mul_transpose_posSemidef At) _
  have hlow : lam * (((ridgeGram At lam)⁻¹ *ᵥ u) ⬝ᵥ ((ridgeGram At lam)⁻¹ *ᵥ u))
      ≤ ((ridgeGram At lam)⁻¹ *ᵥ u) ⬝ᵥ u := by
    have h := hexp
    rw [hGc] at h
    linarith
  have hCS := NearNullDefinitional.dot_cauchy_schwarz ((ridgeGram At lam)⁻¹ *ᵥ u) u
  have hcc := NearNullComposed.dotProduct_self_nonneg ((ridgeGram At lam)⁻¹ *ᵥ u)
  have huu := NearNullComposed.dotProduct_self_nonneg u
  rcases hcc.lt_or_eq with hpos | hzero
  · rw [le_div_iff₀ (by positivity : (0:ℝ) < lam ^ 2)]
    have hcu0 : 0 ≤ ((ridgeGram At lam)⁻¹ *ᵥ u) ⬝ᵥ u :=
      le_trans (mul_nonneg hlam.le hcc) hlow
    have hsq := mul_self_le_mul_self (mul_nonneg hlam.le hcc) hlow
    nlinarith [hCS, hsq, hpos, hcu0, huu]
  · rw [← hzero]
    exact div_nonneg huu (by positivity)

/-- The coefficient vector's norm bound: `‖c_v‖² ≤ ‖Ãv‖²/λ²`. -/
theorem ridgeCoeff_sq_norm_le (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) (v : n → ℝ) :
    ridgeCoeff At lam v ⬝ᵥ ridgeCoeff At lam v ≤ ((At *ᵥ v) ⬝ᵥ (At *ᵥ v)) / lam ^ 2 := by
  rw [ridgeCoeff_eq At hlam v]
  exact gram_inv_sq_norm_le At hlam (At *ᵥ v)

/-- **`p:mapnearnull` (iii), the ridge marginal bound.**  In whitened coordinates, for any
covariance `S̃_y ⪰ 0` with variational upper band `⟪w, S̃_y w⟫ ≤ s‖w‖²` (the `λ_max(S̃_y)`
reading), the blind marginal's variance — the quadratic form of the data covariance at the
readout coefficient — obeys

    Var(vᵀx̂) = ⟪c_v, S̃_y c_v⟫ ≤ s ‖Ãv‖² / λ²,

collapsing at rate `‖Ãv‖²` as the illumination vanishes.  Hypotheses: `λ > 0` and the band —
nothing else. -/
theorem ridge_marginal_variance_bound (At : Matrix m n ℝ) (Sy : Matrix m m ℝ)
    {lam s : ℝ} (hlam : 0 < lam) (hSy : Sy.PosSemidef)
    (hSy_ub : ∀ w : m → ℝ, w ⬝ᵥ (Sy *ᵥ w) ≤ s * (w ⬝ᵥ w)) (v : n → ℝ) :
    ridgeCoeff At lam v ⬝ᵥ (Sy *ᵥ ridgeCoeff At lam v)
      ≤ s * ((At *ᵥ v) ⬝ᵥ (At *ᵥ v)) / lam ^ 2 := by
  have h1 := hSy_ub (ridgeCoeff At lam v)
  have h2 := ridgeCoeff_sq_norm_le At hlam v
  have h0 := NearNullDefinitional.psd_form_nonneg hSy (ridgeCoeff At lam v)
  rcases (NearNullComposed.dotProduct_self_nonneg (ridgeCoeff At lam v)).lt_or_eq with
    hpos | hzero
  · have hs : 0 ≤ s := by nlinarith [h0, h1, hpos]
    calc ridgeCoeff At lam v ⬝ᵥ (Sy *ᵥ ridgeCoeff At lam v)
        ≤ s * (ridgeCoeff At lam v ⬝ᵥ ridgeCoeff At lam v) := h1
      _ ≤ s * (((At *ᵥ v) ⬝ᵥ (At *ᵥ v)) / lam ^ 2) := mul_le_mul_of_nonneg_left h2 hs
      _ = s * ((At *ᵥ v) ⬝ᵥ (At *ᵥ v)) / lam ^ 2 := by ring
  · have hc0 : ridgeCoeff At lam v = 0 := dotProduct_self_eq_zero.mp hzero.symm
    have hu0 : At *ᵥ v = 0 := by
      have hGdet : IsUnit (ridgeGram At lam).det :=
        (Matrix.isUnit_iff_isUnit_det _).mp (ridgeGram_posDef At hlam).isUnit
      have hGc : ridgeGram At lam *ᵥ ridgeCoeff At lam v = At *ᵥ v := by
        rw [ridgeCoeff_eq At hlam v, Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hGdet,
          Matrix.one_mulVec]
      rw [hc0, Matrix.mulVec_zero] at hGc
      exact hGc.symm
    rw [hc0, hu0]
    simp

/-- **The exact-kernel endpoint.**  `Ãv = 0` makes the readout coefficient exactly `0`. -/
theorem ridgeCoeff_zero_of_kernel (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam)
    {v : n → ℝ} (hv : At *ᵥ v = 0) : ridgeCoeff At lam v = 0 := by
  rw [ridgeCoeff_eq At hlam v, hv, Matrix.mulVec_zero]

/-- On the exact kernel the ridge blind marginal has variance exactly `0` — `p:map`'s collapse
recovered as the endpoint of the graded bound. -/
theorem ridge_marginal_zero_of_kernel (At : Matrix m n ℝ) (Sy : Matrix m m ℝ)
    {lam : ℝ} (hlam : 0 < lam) {v : n → ℝ} (hv : At *ᵥ v = 0) :
    ridgeCoeff At lam v ⬝ᵥ (Sy *ᵥ ridgeCoeff At lam v) = 0 := by
  rw [ridgeCoeff_zero_of_kernel At hlam hv]
  simp

omit [DecidableEq m] in
/-- **The normal equation.**  `(ÃᵀÃ + λI) x̂(ỹ) = Ãᵀ ỹ`. -/
theorem ridge_normal_equation (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam) (y : m → ℝ) :
    ridgeNormal At lam *ᵥ (ridgeEstimator At lam *ᵥ y) = Atᵀ *ᵥ y := by
  have hNdet : IsUnit (ridgeNormal At lam).det :=
    (Matrix.isUnit_iff_isUnit_det _).mp (ridgeNormal_posDef At hlam).isUnit
  simp only [ridgeEstimator]
  rw [Matrix.mulVec_mulVec, ← Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hNdet, Matrix.one_mul]

omit [DecidableEq m] in
/-- **Quadratic minimization from the normal equation.**  Any `x̂` satisfying
`(ÃᵀÃ + λI)x̂ = Ãᵀỹ` minimizes the whitened ridge objective `½‖ỹ − Ãx‖² + (λ/2)‖x‖²` —
the exact expansion `obj(x) − obj(x̂) = ½‖Ã(x − x̂)‖² + (λ/2)‖x − x̂‖² ≥ 0`. -/
theorem ridge_objective_min (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 ≤ lam)
    {y : m → ℝ} {xh : n → ℝ}
    (hnormal : ridgeNormal At lam *ᵥ xh = Atᵀ *ᵥ y) (x : n → ℝ) :
    (y - At *ᵥ xh) ⬝ᵥ (y - At *ᵥ xh) / 2 + lam * (xh ⬝ᵥ xh) / 2
      ≤ (y - At *ᵥ x) ⬝ᵥ (y - At *ᵥ x) / 2 + lam * (x ⬝ᵥ x) / 2 := by
  have hgrad : Atᵀ *ᵥ (y - At *ᵥ xh) = lam • xh := by
    have h2 : Atᵀ *ᵥ (At *ᵥ xh) + lam • xh = Atᵀ *ᵥ y := by
      have h3 := hnormal
      simp only [ridgeNormal, Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec,
        ← Matrix.mulVec_mulVec] at h3
      exact h3
    rw [Matrix.mulVec_sub, ← h2]
    abel
  have hcross : (y - At *ᵥ xh) ⬝ᵥ (At *ᵥ (x - xh)) = lam * (xh ⬝ᵥ (x - xh)) := by
    rw [dotProduct_mulVec, ← Matrix.mulVec_transpose, hgrad, smul_dotProduct, smul_eq_mul]
  have hsplit : y - At *ᵥ x = y - At *ᵥ xh - At *ᵥ (x - xh) := by
    rw [Matrix.mulVec_sub]
    abel
  have e1 : (y - At *ᵥ x) ⬝ᵥ (y - At *ᵥ x)
      = (y - At *ᵥ xh) ⬝ᵥ (y - At *ᵥ xh)
        - 2 * ((y - At *ᵥ xh) ⬝ᵥ (At *ᵥ (x - xh)))
        + (At *ᵥ (x - xh)) ⬝ᵥ (At *ᵥ (x - xh)) := by
    rw [hsplit]
    exact dot_expand_sub (y - At *ᵥ xh) (At *ᵥ (x - xh))
  have e2 : x ⬝ᵥ x = xh ⬝ᵥ xh + 2 * (xh ⬝ᵥ (x - xh)) + (x - xh) ⬝ᵥ (x - xh) := by
    have hxsplit : x = xh + (x - xh) := by abel
    conv_lhs => rw [hxsplit]
    exact dot_expand_add xh (x - xh)
  have e2l : lam * (x ⬝ᵥ x)
      = lam * (xh ⬝ᵥ xh) + 2 * (lam * (xh ⬝ᵥ (x - xh))) + lam * ((x - xh) ⬝ᵥ (x - xh)) := by
    rw [e2]; ring
  have hb0 : 0 ≤ (At *ᵥ (x - xh)) ⬝ᵥ (At *ᵥ (x - xh)) :=
    NearNullComposed.dotProduct_self_nonneg _
  have hh0 : 0 ≤ (x - xh) ⬝ᵥ (x - xh) := NearNullComposed.dotProduct_self_nonneg _
  linarith [hcross, e1, e2l, hb0, mul_nonneg hlam hh0]

omit [DecidableEq m] in
/-- **The ridge estimator is the archive's minimizer.**  `x̂(ỹ) = (ÃᵀÃ + λI)⁻¹Ãᵀỹ` minimizes
`½‖ỹ − Ãx‖² + (λ/2)‖x‖²` — so the closed form the marginal bound is proved for is exactly
`p:mapnearnull`'s estimator (Gaussian data term, ridge penalty, whitened coordinates). -/
theorem ridge_estimator_minimizes (At : Matrix m n ℝ) {lam : ℝ} (hlam : 0 < lam)
    (y : m → ℝ) (x : n → ℝ) :
    (y - At *ᵥ (ridgeEstimator At lam *ᵥ y)) ⬝ᵥ (y - At *ᵥ (ridgeEstimator At lam *ᵥ y)) / 2
        + lam * ((ridgeEstimator At lam *ᵥ y) ⬝ᵥ (ridgeEstimator At lam *ᵥ y)) / 2
      ≤ (y - At *ᵥ x) ⬝ᵥ (y - At *ᵥ x) / 2 + lam * (x ⬝ᵥ x) / 2 :=
  ridge_objective_min At hlam.le (ridge_normal_equation At hlam y) x

end Ridge

/-! ## (B) The coupled counterexample: order-one blind marginal at exact blindness -/

section Coupled

/-- The witness operator: one measurement reading the first coordinate, `A = (1 0)`. -/
def coupledA : Matrix (Fin 1) (Fin 2) ℝ := !![1, 0]

/-- The blind direction `v = e₂`. -/
def coupledV : Fin 2 → ℝ := ![0, 1]

/-- The coupled penalty precision `P = [[p, c], [c, q]]` — `c ≠ 0` is the coupling. -/
def coupledPrec (p c q : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![p, c; c, q]

/-- The witness's determinant scale `Δ = (1+p)q − c²` (of `AᵀA + P`). -/
def coupledDelta (p c q : ℝ) : ℝ := (1 + p) * q - c ^ 2

/-- The MAP objective `½(y − x₁)² + ½ xᵀPx`, written out in coordinates. -/
noncomputable def coupledObj (p c q y : ℝ) (x : Fin 2 → ℝ) : ℝ :=
  (y - x 0) ^ 2 / 2 + (p * x 0 ^ 2 + 2 * c * (x 0 * x 1) + q * x 1 ^ 2) / 2

/-- The closed-form minimizer `x̂(y) = (qy/Δ, −cy/Δ)`. -/
noncomputable def coupledXhat (p c q y : ℝ) : Fin 2 → ℝ :=
  ![q * y / coupledDelta p c q, -c * y / coupledDelta p c q]

/-- **The direction is exactly blind**: `A v = 0`, i.e. `‖Av‖ = 0`. -/
theorem coupled_blind_direction : coupledA *ᵥ coupledV = 0 := by
  funext i
  fin_cases i
  simp [coupledA, coupledV, Matrix.mulVec, dotProduct, Fin.sum_univ_two]

/-- The coordinate objective IS the matrix-form MAP objective
`½‖![y] − Ax‖² + ½ xᵀPx` — anchoring the scalar computation to the paper's objects. -/
theorem coupledObj_eq_matrix (p c q y : ℝ) (x : Fin 2 → ℝ) :
    coupledObj p c q y x
      = (![y] - coupledA *ᵥ x) ⬝ᵥ (![y] - coupledA *ᵥ x) / 2
        + x ⬝ᵥ (coupledPrec p c q *ᵥ x) / 2 := by
  simp only [coupledObj, coupledA, coupledPrec, Matrix.mulVec, dotProduct, Fin.sum_univ_two,
    Fin.sum_univ_one, Pi.sub_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.empty_val', Matrix.cons_val_fin_one, Matrix.of_apply]
  ring

/-- A positive-definite coupled penalty (`p, q > 0`, `c² < pq`) has `Δ > 0`. -/
theorem coupledDelta_pos_of_posDef {p c q : ℝ} (_hp : 0 < p) (hq : 0 < q)
    (hpd : c ^ 2 < p * q) : 0 < coupledDelta p c q := by
  simp only [coupledDelta]
  nlinarith

/-- The exact quadratic expansion around the closed-form point:
`obj(x) − obj(x̂) = ½[(1+p)h₀² + 2c h₀h₁ + q h₁²]`, `h = x − x̂`. -/
theorem coupledObj_sub_eq (p c q y : ℝ) (hD : coupledDelta p c q ≠ 0) (x : Fin 2 → ℝ) :
    coupledObj p c q y x - coupledObj p c q y (coupledXhat p c q y)
      = ((1 + p) * (x 0 - coupledXhat p c q y 0) ^ 2
          + 2 * c * ((x 0 - coupledXhat p c q y 0) * (x 1 - coupledXhat p c q y 1))
          + q * (x 1 - coupledXhat p c q y 1) ^ 2) / 2 := by
  have hD' : (1 + p) * q - c ^ 2 ≠ 0 := by simpa [coupledDelta] using hD
  -- the closed-form point is stationary: the two components of `∇obj(x̂) = 0`
  have h1 : (1 + p) * (q * y / coupledDelta p c q) + c * (-c * y / coupledDelta p c q) = y := by
    simp only [coupledDelta]
    field_simp
    ring
  have h2 : c * (q * y / coupledDelta p c q) + q * (-c * y / coupledDelta p c q) = 0 := by
    simp only [coupledDelta]
    field_simp
    ring
  have hxhat : coupledXhat p c q y
      = ![q * y / coupledDelta p c q, -c * y / coupledDelta p c q] := rfl
  rw [hxhat]
  simp only [coupledObj, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
  -- exact second-order expansion around a stationary point: pure ring identity
  linear_combination (x 0 - q * y / coupledDelta p c q) * h1
    + (x 1 - -c * y / coupledDelta p c q) * h2

/-- **`x̂` minimizes.**  Under `q > 0`, `Δ > 0` the closed-form point minimizes the coupled
objective for every measurement `y`. -/
theorem coupled_minimizes {p c q : ℝ} (hq : 0 < q) (hD : 0 < coupledDelta p c q) (y : ℝ)
    (x : Fin 2 → ℝ) :
    coupledObj p c q y (coupledXhat p c q y) ≤ coupledObj p c q y x := by
  have hD' : (0:ℝ) < (1 + p) * q - c ^ 2 := by simpa [coupledDelta] using hD
  have hsub := coupledObj_sub_eq p c q y hD.ne' x
  have hkey : q * (coupledObj p c q y x - coupledObj p c q y (coupledXhat p c q y))
      = ((q * (x 1 - coupledXhat p c q y 1) + c * (x 0 - coupledXhat p c q y 0)) ^ 2
        + ((1 + p) * q - c ^ 2) * (x 0 - coupledXhat p c q y 0) ^ 2) / 2 := by
    rw [hsub]; ring
  nlinarith [hkey, hq,
    sq_nonneg (q * (x 1 - coupledXhat p c q y 1) + c * (x 0 - coupledXhat p c q y 0)),
    mul_nonneg hD'.le (sq_nonneg (x 0 - coupledXhat p c q y 0))]

/-- **The minimizer is unique** — strict convexity of the `2×2` form (`q > 0`, `Δ > 0`), so any
point doing at least as well as `x̂` *is* `x̂`: every archive selection is forced. -/
theorem coupled_minimizer_unique {p c q : ℝ} (hq : 0 < q) (hD : 0 < coupledDelta p c q) (y : ℝ)
    {x : Fin 2 → ℝ} (hx : ∀ x', coupledObj p c q y x ≤ coupledObj p c q y x') :
    x = coupledXhat p c q y := by
  have hD' : (0:ℝ) < (1 + p) * q - c ^ 2 := by simpa [coupledDelta] using hD
  have heq : coupledObj p c q y x = coupledObj p c q y (coupledXhat p c q y) :=
    le_antisymm (hx _) (coupled_minimizes hq hD y x)
  have hsub := coupledObj_sub_eq p c q y hD.ne' x
  rw [heq, sub_self] at hsub
  have hE : (1 + p) * (x 0 - coupledXhat p c q y 0) ^ 2
      + 2 * c * ((x 0 - coupledXhat p c q y 0) * (x 1 - coupledXhat p c q y 1))
      + q * (x 1 - coupledXhat p c q y 1) ^ 2 = 0 := by linarith [hsub]
  have hkey : (q * (x 1 - coupledXhat p c q y 1) + c * (x 0 - coupledXhat p c q y 0)) ^ 2
      + ((1 + p) * q - c ^ 2) * (x 0 - coupledXhat p c q y 0) ^ 2 = 0 := by
    linear_combination q * hE
  have h0le : (x 0 - coupledXhat p c q y 0) ^ 2 ≤ 0 := by
    nlinarith [sq_nonneg (q * (x 1 - coupledXhat p c q y 1)
      + c * (x 0 - coupledXhat p c q y 0)), hkey, hD']
  have h0 : x 0 - coupledXhat p c q y 0 = 0 :=
    sq_eq_zero_iff.mp (le_antisymm h0le (sq_nonneg _))
  have h1 : x 1 - coupledXhat p c q y 1 = 0 := by
    rw [h0] at hkey
    have hsq : (q * (x 1 - coupledXhat p c q y 1)) ^ 2 = 0 := by linear_combination hkey
    have hq1 : q * (x 1 - coupledXhat p c q y 1) = 0 := sq_eq_zero_iff.mp hsq
    rcases mul_eq_zero.mp hq1 with h | h
    · exact absurd h hq.ne'
    · exact h
  funext i
  fin_cases i
  · show x 0 = coupledXhat p c q y 0
    have := sub_eq_zero.mp h0
    exact this
  · show x 1 = coupledXhat p c q y 1
    have := sub_eq_zero.mp h1
    exact this

/-- **The blind marginal is a deterministic linear image of the data** with coefficient
`−c/Δ` — nonzero whenever the penalty couples (`c ≠ 0`). -/
theorem coupled_blind_readout (p c q y : ℝ) :
    coupledV ⬝ᵥ coupledXhat p c q y = (-c / coupledDelta p c q) * y := by
  simp only [coupledV, coupledXhat, dotProduct, Fin.sum_univ_two, Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.head_cons]
  ring

/-- Every minimizer selection `T` reads out the same blind coordinate `(−c/Δ)·y`. -/
theorem coupled_selection_forced {p c q : ℝ} (hq : 0 < q) (hD : 0 < coupledDelta p c q)
    (T : ℝ → Fin 2 → ℝ) (hT : ∀ y x, coupledObj p c q y (T y) ≤ coupledObj p c q y x)
    (y : ℝ) :
    coupledV ⬝ᵥ T y = (-c / coupledDelta p c q) * y := by
  rw [coupled_minimizer_unique hq hD y (hT y), coupled_blind_readout]

/-- **The blind marginal's law, exactly.**  Under `y ∼ N(0, s)` the archived blind coordinate of
any minimizer selection has law `N(0, (c/Δ)² s)` — mathlib's `gaussianReal_map_const_mul` on the
forced linear readout. -/
theorem coupled_selection_law {p c q : ℝ} (hq : 0 < q) (hD : 0 < coupledDelta p c q)
    (T : ℝ → Fin 2 → ℝ) (hT : ∀ y x, coupledObj p c q y (T y) ≤ coupledObj p c q y x)
    (s : ℝ≥0) :
    (gaussianReal 0 s).map (fun y => coupledV ⬝ᵥ T y)
      = gaussianReal 0
          (NNReal.mk ((-c / coupledDelta p c q) ^ 2) (sq_nonneg _) * s) := by
  have hTeq : (fun y => coupledV ⬝ᵥ T y) = fun y => (-c / coupledDelta p c q) * y :=
    funext fun y => coupled_selection_forced hq hD T hT y
  rw [hTeq, gaussianReal_map_const_mul, mul_zero]

/-- The blind marginal's variance: exactly `(c/Δ)² s`. -/
theorem coupled_marginal_variance {p c q : ℝ} (hq : 0 < q) (hD : 0 < coupledDelta p c q)
    (T : ℝ → Fin 2 → ℝ) (hT : ∀ y x, coupledObj p c q y (T y) ≤ coupledObj p c q y x)
    (s : ℝ≥0) :
    Var[fun x => x; (gaussianReal 0 s).map (fun y => coupledV ⬝ᵥ T y)]
      = (c / coupledDelta p c q) ^ 2 * (s : ℝ) := by
  rw [coupled_selection_law hq hD T hT s, variance_fun_id_gaussianReal, NNReal.coe_mul,
    NNReal.coe_mk]
  ring

/-- The variance is bounded below by a positive constant (in the coupling and the data spread
alone), even though `‖Av‖ = 0` exactly. -/
theorem coupled_marginal_variance_pos {p c q : ℝ} (hc : c ≠ 0) (hD : 0 < coupledDelta p c q)
    {s : ℝ≥0} (hs : 0 < s) :
    0 < (c / coupledDelta p c q) ^ 2 * (s : ℝ) := by
  have h1 : c / coupledDelta p c q ≠ 0 := div_ne_zero hc hD.ne'
  have h2 : 0 < (c / coupledDelta p c q) ^ 2 :=
    (sq_nonneg _).lt_of_ne (Ne.symm (pow_ne_zero 2 h1))
  have hs' : (0:ℝ) < (s : ℝ) := hs
  exact mul_pos h2 hs'

/-- **`p:mapnearnull` (iii), the closing clause.**  The witness direction is exactly blind
(`Av = 0`, so `‖Av‖ = 0`), yet the blind *marginal* of every minimizer selection has strictly
positive variance — of order one in the coupling, independent of `‖Av‖`.  No bound
`Var(vᵀx̂) ≤ C‖Ãv‖²` (or any function of `‖Av‖` vanishing at `0`) can hold off alignment:
only the conditional statement of part (ii) survives. -/
theorem coupled_marginal_order_one {p c q : ℝ} (hq : 0 < q) (hc : c ≠ 0)
    (hD : 0 < coupledDelta p c q)
    (T : ℝ → Fin 2 → ℝ) (hT : ∀ y x, coupledObj p c q y (T y) ≤ coupledObj p c q y x)
    {s : ℝ≥0} (hs : 0 < s) :
    coupledA *ᵥ coupledV = 0 ∧
      0 < Var[fun x => x; (gaussianReal 0 s).map (fun y => coupledV ⬝ᵥ T y)] := by
  refine ⟨coupled_blind_direction, ?_⟩
  rw [coupled_marginal_variance hq hD T hT s]
  exact coupled_marginal_variance_pos hc hD hs

end Coupled

/-! ## (C) The `E‖d‖²` objective comparison: the display, the pointwise bound, the integral -/

section Comparison

variable {m n : Type*} [Fintype m] [DecidableEq m] [Fintype n] [DecidableEq n]

omit [DecidableEq m] [DecidableEq n] in
/-- **The paper's comparison display.**  If `x̂` does at least as well as the reference `x₀` on
the objective `½‖y − Ax‖²_{Γ⁻¹} + R(x)` and `R(x̂) ≥ R_inf`, then

    ‖y − Ax̂‖²_{Γ⁻¹} ≤ ‖y − Ax₀‖²_{Γ⁻¹} + 2(R(x₀) − R_inf).

Purely deterministic; `Ginv` is any matrix (instantiated at `Γ⁻¹`). -/
theorem misfit_comparison (A : Matrix m n ℝ) (Ginv : Matrix m m ℝ) (R : (n → ℝ) → ℝ)
    (y : m → ℝ) (xh x0 : n → ℝ) {Rinf : ℝ}
    (hmin : (y - A *ᵥ xh) ⬝ᵥ (Ginv *ᵥ (y - A *ᵥ xh)) / 2 + R xh
          ≤ (y - A *ᵥ x0) ⬝ᵥ (Ginv *ᵥ (y - A *ᵥ x0)) / 2 + R x0)
    (hRlb : Rinf ≤ R xh) :
    (y - A *ᵥ xh) ⬝ᵥ (Ginv *ᵥ (y - A *ᵥ xh))
      ≤ (y - A *ᵥ x0) ⬝ᵥ (Ginv *ᵥ (y - A *ᵥ x0)) + 2 * (R x0 - Rinf) := by
  linarith

omit [DecidableEq n] in
/-- The Gaussian gradient is controlled by the misfit: with the variational noise floor
`γ‖w‖² ≤ ⟪w, Γw⟫`, `‖Γ⁻¹r‖² ≤ γ⁻¹ ⟪r, Γ⁻¹r⟫` for every residual `r`. -/
theorem grad_sq_le_of_band {Gam : Matrix m m ℝ} (hGam : Gam.PosDef) {gam : ℝ}
    (hgam : 0 < gam)
    (hband : ∀ w : m → ℝ, gam * (w ⬝ᵥ w) ≤ w ⬝ᵥ (Gam *ᵥ w)) (r : m → ℝ) :
    (Gam⁻¹ *ᵥ r) ⬝ᵥ (Gam⁻¹ *ᵥ r) ≤ gam⁻¹ * (r ⬝ᵥ (Gam⁻¹ *ᵥ r)) := by
  have hdet : IsUnit Gam.det := (Matrix.isUnit_iff_isUnit_det _).mp hGam.isUnit
  have hGz : Gam *ᵥ (Gam⁻¹ *ᵥ r) = r := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hdet, Matrix.one_mulVec]
  have h1 : gam * ((Gam⁻¹ *ᵥ r) ⬝ᵥ (Gam⁻¹ *ᵥ r)) ≤ (Gam⁻¹ *ᵥ r) ⬝ᵥ r := by
    have h := hband (Gam⁻¹ *ᵥ r)
    rwa [hGz] at h
  have h2 : (Gam⁻¹ *ᵥ r) ⬝ᵥ r = r ⬝ᵥ (Gam⁻¹ *ᵥ r) := dotProduct_comm _ _
  calc (Gam⁻¹ *ᵥ r) ⬝ᵥ (Gam⁻¹ *ᵥ r)
      = gam⁻¹ * (gam * ((Gam⁻¹ *ᵥ r) ⬝ᵥ (Gam⁻¹ *ᵥ r))) :=
        (inv_mul_cancel_left₀ (ne_of_gt hgam) _).symm
    _ ≤ gam⁻¹ * (r ⬝ᵥ (Gam⁻¹ *ᵥ r)) := by
        refine mul_le_mul_of_nonneg_left ?_ (inv_pos.mpr hgam).le
        rw [← h2]
        exact h1

omit [DecidableEq m] in
/-- `‖a − b‖² ≤ 2‖a‖² + 2‖b‖²`, dot-product form. -/
theorem dot_sub_le_two (a b : m → ℝ) :
    (a - b) ⬝ᵥ (a - b) ≤ 2 * (a ⬝ᵥ a) + 2 * (b ⬝ᵥ b) := by
  have h0 : 0 ≤ (a + b) ⬝ᵥ (a + b) := NearNullComposed.dotProduct_self_nonneg _
  have e1 := dot_expand_sub a b
  have e2 := dot_expand_add a b
  linarith

omit [DecidableEq n] in
/-- **The pointwise `‖d‖²` bound, uniform in the direction.**  For the Gaussian data term with
noise floor `γ`, a penalty bounded below by `R_inf`, and any minimizer `x̂` of the objective at
the measurement `y`, the gradient `d(y) = Γ⁻¹(y − Ax̂)` obeys

    ‖d(y)‖² ≤ (2/γ²)‖y‖² + [(2/γ²)‖Ax₀‖² + (2/γ)(R(x₀) − R_inf)].

Every constant on the right depends only on `(γ, x₀, R(x₀), R_inf)` — no `v`, no `‖Av‖` —
which is the paper's "`E‖d‖²` bounded uniformly as `‖Av‖ → 0`". -/
theorem grad_sq_pointwise_bound (A : Matrix m n ℝ) {Gam : Matrix m m ℝ} (hGam : Gam.PosDef)
    {gam : ℝ} (hgam : 0 < gam)
    (hband : ∀ w : m → ℝ, gam * (w ⬝ᵥ w) ≤ w ⬝ᵥ (Gam *ᵥ w))
    (R : (n → ℝ) → ℝ) {Rinf : ℝ} (hRlb : ∀ x, Rinf ≤ R x)
    (y : m → ℝ) (xh x0 : n → ℝ)
    (hmin : (y - A *ᵥ xh) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xh)) / 2 + R xh
          ≤ (y - A *ᵥ x0) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ x0)) / 2 + R x0) :
    (Gam⁻¹ *ᵥ (y - A *ᵥ xh)) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xh))
      ≤ 2 / gam ^ 2 * (y ⬝ᵥ y)
        + (2 / gam ^ 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)) + 2 / gam * (R x0 - Rinf)) := by
  have hcomp := misfit_comparison A Gam⁻¹ R y xh x0 hmin (hRlb xh)
  have hstep1 := grad_sq_le_of_band hGam hgam hband (y - A *ᵥ xh)
  have hstep2 : (y - A *ᵥ x0) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ x0))
      ≤ gam⁻¹ * ((y - A *ᵥ x0) ⬝ᵥ (y - A *ᵥ x0)) :=
    NearNullDefinitional.inv_form_le hGam hgam hband (y - A *ᵥ x0)
  have hstep3 : (y - A *ᵥ x0) ⬝ᵥ (y - A *ᵥ x0)
      ≤ 2 * (y ⬝ᵥ y) + 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)) := dot_sub_le_two y (A *ᵥ x0)
  have hg1 : (0:ℝ) ≤ gam⁻¹ := (inv_pos.mpr hgam).le
  have h4 : (y - A *ᵥ xh) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xh))
      ≤ gam⁻¹ * (2 * (y ⬝ᵥ y) + 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0))) + 2 * (R x0 - Rinf) := by
    have h5 := mul_le_mul_of_nonneg_left hstep3 hg1
    linarith [hcomp, hstep2, h5]
  have hchain : (Gam⁻¹ *ᵥ (y - A *ᵥ xh)) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xh))
      ≤ gam⁻¹ * (gam⁻¹ * (2 * (y ⬝ᵥ y) + 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)))
          + 2 * (R x0 - Rinf)) :=
    hstep1.trans (mul_le_mul_of_nonneg_left h4 hg1)
  have heq : gam⁻¹ * (gam⁻¹ * (2 * (y ⬝ᵥ y) + 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)))
        + 2 * (R x0 - Rinf))
      = 2 / gam ^ 2 * (y ⬝ᵥ y)
        + (2 / gam ^ 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)) + 2 / gam * (R x0 - Rinf)) := by
    field_simp
    ring
  rw [← heq]
  exact hchain

omit [DecidableEq n] in
/-- **`E‖d‖²` bounded, with the explicit uniform constants.**  Integrating the pointwise bound
against any measurement law `μ` (as a lower Lebesgue integral — the minimizer selection
`x̂(·)` need not be measurable):

    ∫⁻ ‖d(y)‖² dμ ≤ (2/γ²) ∫⁻ ‖y‖² dμ + C(γ, x₀, R(x₀) − R_inf) · μ(univ).

Nothing on the right refers to `v` — the bound is uniform as `‖Av‖ → 0`. -/
theorem expected_grad_sq_le (A : Matrix m n ℝ) {Gam : Matrix m m ℝ} (hGam : Gam.PosDef)
    {gam : ℝ} (hgam : 0 < gam)
    (hband : ∀ w : m → ℝ, gam * (w ⬝ᵥ w) ≤ w ⬝ᵥ (Gam *ᵥ w))
    (R : (n → ℝ) → ℝ) {Rinf : ℝ} (hRlb : ∀ x, Rinf ≤ R x)
    (xhat : (m → ℝ) → n → ℝ) (x0 : n → ℝ)
    (hmin : ∀ y : m → ℝ,
      (y - A *ᵥ xhat y) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xhat y)) / 2 + R (xhat y)
        ≤ (y - A *ᵥ x0) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ x0)) / 2 + R x0)
    (μ : Measure (m → ℝ)) :
    ∫⁻ y, ENNReal.ofReal ((Gam⁻¹ *ᵥ (y - A *ᵥ xhat y)) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xhat y))) ∂μ
      ≤ ENNReal.ofReal (2 / gam ^ 2) * ∫⁻ y, ENNReal.ofReal (y ⬝ᵥ y) ∂μ
        + ENNReal.ofReal (2 / gam ^ 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)) + 2 / gam * (R x0 - Rinf))
          * μ Set.univ := by
  have hC1 : (0:ℝ) ≤ 2 / gam ^ 2 := by positivity
  have hmono : ∀ y : m → ℝ,
      ENNReal.ofReal ((Gam⁻¹ *ᵥ (y - A *ᵥ xhat y)) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xhat y)))
        ≤ ENNReal.ofReal (2 / gam ^ 2) * ENNReal.ofReal (y ⬝ᵥ y)
          + ENNReal.ofReal
              (2 / gam ^ 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)) + 2 / gam * (R x0 - Rinf)) := by
    intro y
    have hpt := grad_sq_pointwise_bound A hGam hgam hband R hRlb y (xhat y) x0 (hmin y)
    have h1 := ENNReal.ofReal_le_ofReal hpt
    refine h1.trans ?_
    refine ENNReal.ofReal_add_le.trans ?_
    rw [ENNReal.ofReal_mul hC1]
  calc ∫⁻ y, ENNReal.ofReal
        ((Gam⁻¹ *ᵥ (y - A *ᵥ xhat y)) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xhat y))) ∂μ
      ≤ ∫⁻ y, (ENNReal.ofReal (2 / gam ^ 2) * ENNReal.ofReal (y ⬝ᵥ y)
          + ENNReal.ofReal
              (2 / gam ^ 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)) + 2 / gam * (R x0 - Rinf))) ∂μ :=
        lintegral_mono hmono
    _ = (∫⁻ y, ENNReal.ofReal (2 / gam ^ 2) * ENNReal.ofReal (y ⬝ᵥ y) ∂μ)
        + ∫⁻ _, ENNReal.ofReal
            (2 / gam ^ 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)) + 2 / gam * (R x0 - Rinf)) ∂μ :=
        lintegral_add_right _ measurable_const
    _ = ENNReal.ofReal (2 / gam ^ 2) * (∫⁻ y, ENNReal.ofReal (y ⬝ᵥ y) ∂μ)
        + ENNReal.ofReal (2 / gam ^ 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)) + 2 / gam * (R x0 - Rinf))
          * μ Set.univ := by
        rw [lintegral_const_mul' _ _ ENNReal.ofReal_ne_top, lintegral_const]

omit [DecidableEq n] in
/-- **`p:mapnearnull` (ii)'s hypothesis, closed.**  For the Gaussian data term (noise floor
`γ > 0`), a penalty bounded below, and a measurement law of finite second moment,
`E‖d‖² < ∞` — uniformly as `‖Av‖ → 0` (no constant refers to `v`).  The finite-second-moment
hypothesis and the conclusion are lower Lebesgue integrals, so no measurability of the
minimizer selection is assumed. -/
theorem expected_grad_sq_lt_top (A : Matrix m n ℝ) {Gam : Matrix m m ℝ} (hGam : Gam.PosDef)
    {gam : ℝ} (hgam : 0 < gam)
    (hband : ∀ w : m → ℝ, gam * (w ⬝ᵥ w) ≤ w ⬝ᵥ (Gam *ᵥ w))
    (R : (n → ℝ) → ℝ) {Rinf : ℝ} (hRlb : ∀ x, Rinf ≤ R x)
    (xhat : (m → ℝ) → n → ℝ) (x0 : n → ℝ)
    (hmin : ∀ y : m → ℝ,
      (y - A *ᵥ xhat y) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xhat y)) / 2 + R (xhat y)
        ≤ (y - A *ᵥ x0) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ x0)) / 2 + R x0)
    (μ : Measure (m → ℝ)) [IsFiniteMeasure μ]
    (hmom : ∫⁻ y, ENNReal.ofReal (y ⬝ᵥ y) ∂μ ≠ ⊤) :
    ∫⁻ y, ENNReal.ofReal
        ((Gam⁻¹ *ᵥ (y - A *ᵥ xhat y)) ⬝ᵥ (Gam⁻¹ *ᵥ (y - A *ᵥ xhat y))) ∂μ < ⊤ := by
  refine lt_of_le_of_lt (expected_grad_sq_le A hGam hgam hband R hRlb xhat x0 hmin μ) ?_
  have h1 : ENNReal.ofReal (2 / gam ^ 2) * (∫⁻ y, ENNReal.ofReal (y ⬝ᵥ y) ∂μ) < ⊤ :=
    ENNReal.mul_lt_top ENNReal.ofReal_lt_top (lt_top_iff_ne_top.mpr hmom)
  have h2 : ENNReal.ofReal (2 / gam ^ 2 * ((A *ᵥ x0) ⬝ᵥ (A *ᵥ x0)) + 2 / gam * (R x0 - Rinf))
      * μ Set.univ < ⊤ :=
    ENNReal.mul_lt_top ENNReal.ofReal_lt_top (measure_lt_top μ _)
  exact ENNReal.add_lt_top.mpr ⟨h1, h2⟩

end Comparison

end RidgeMarginal

-- #print axioms audit (p:mapnearnull (iii) + the E‖d‖² comparison of (ii))
#print axioms RidgeMarginal.transpose_mul_self_posSemidef
#print axioms RidgeMarginal.self_mul_transpose_posSemidef
#print axioms RidgeMarginal.ridgeNormal_posDef
#print axioms RidgeMarginal.ridgeGram_posDef
#print axioms RidgeMarginal.push_through
#print axioms RidgeMarginal.ridgeEstimator_transpose
#print axioms RidgeMarginal.ridgeCoeff_eq
#print axioms RidgeMarginal.ridge_readout
#print axioms RidgeMarginal.gram_inv_sq_norm_le
#print axioms RidgeMarginal.ridgeCoeff_sq_norm_le
#print axioms RidgeMarginal.ridge_marginal_variance_bound
#print axioms RidgeMarginal.ridgeCoeff_zero_of_kernel
#print axioms RidgeMarginal.ridge_marginal_zero_of_kernel
#print axioms RidgeMarginal.ridge_normal_equation
#print axioms RidgeMarginal.ridge_objective_min
#print axioms RidgeMarginal.ridge_estimator_minimizes
#print axioms RidgeMarginal.coupled_blind_direction
#print axioms RidgeMarginal.coupledObj_eq_matrix
#print axioms RidgeMarginal.coupledDelta_pos_of_posDef
#print axioms RidgeMarginal.coupledObj_sub_eq
#print axioms RidgeMarginal.coupled_minimizes
#print axioms RidgeMarginal.coupled_minimizer_unique
#print axioms RidgeMarginal.coupled_blind_readout
#print axioms RidgeMarginal.coupled_selection_forced
#print axioms RidgeMarginal.coupled_selection_law
#print axioms RidgeMarginal.coupled_marginal_variance
#print axioms RidgeMarginal.coupled_marginal_variance_pos
#print axioms RidgeMarginal.coupled_marginal_order_one
#print axioms RidgeMarginal.misfit_comparison
#print axioms RidgeMarginal.grad_sq_le_of_band
#print axioms RidgeMarginal.dot_sub_le_two
#print axioms RidgeMarginal.grad_sq_pointwise_bound
#print axioms RidgeMarginal.expected_grad_sq_le
#print axioms RidgeMarginal.expected_grad_sq_lt_top
