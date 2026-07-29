/-
# `eq:covmu` — coverage against damping, and its monotonicity reading

Machine-checks the display \eqref{eq:covmu} of the paper and the two sentences around it
(the passage between `p:nearnull` and `p:mapnearnull`):

> "a Gaussian regularizer with curvature `μ_v` along a blind direction advertises a conditional
> spread of at least `μ_v^{−1/2}` there, so the coverage its central-`(1−α)` band attains against
> a truth of spread `s_⋆` and matching blind mean (a mean gap only lowers it) obeys
> `C ≥ 2Φ(z/(s_⋆√μ_v)) − 1` (eq:covmu), with equality when the blind subspace is one-dimensional;
> on a larger one the curvature alone understates the fiber spread away from the blind block's
> eigendirections, so the display bounds the shortfall rather than giving it.  The right side is a decreasing function
> of the damping, and so, by the Loewner monotonicity of the fiber conditional in the damping, is
> the coverage itself."

## Setup

The state space splits as resolved ⊕ blind (`R ⊕ B`, the `BlindFreeze` convention: blind = second
summand).  The Gaussian regularizer has precision `P ≻ 0`; along a unit blind direction `v`:

* `curvature P v  := ⟪ι v, P (ι v)⟫`  (`ι` = zero-padding into `R ⊕ B`) — the paper's `μ_v`
  (`p:mapnearnull`: "a quadratic penalty of precision `P` has `μ_v = vᵀPv`"), equal to the
  blind-block form `⟪v, P₂₂ v⟫` (`curvature_eq_block`);
* `condVar P v := ⟪v, (P₂₂)⁻¹ v⟫`, `spread P v := √(condVar P v)` — the **advertised
  fiber-conditional variance/spread** along `v` given the resolved coordinates.

## The modeling identification (the one non-theorem input, definitional)

`condVar` *defines* the advertised fiber-conditional variance as the `v`-form of the inverse
blind-block precision.  That this is the fiber-conditional covariance of the Gaussian `𝒩(·, P⁻¹)`
given the resolved block is the standard Gaussian-conditioning identity; its linear-algebra half —
inverse blind-block precision = Schur complement of the covariance — is machine-checked
(`condVar_eq_schur_form`, via `BlindFreeze.invBlock₂₂_eq_schur_inv`), and the probabilistic half
(Gaussian conditional covariance = Schur complement) is the same modeling posture `BlindFreeze`
Part 2 takes for `p:blind`.  Nothing else is assumed.

**Alignment.**  `eq:covmu` is proved in the aligned reading the paper states it in: the band is
centered at the truth's blind mean (the display trades off *spreads*; a center mismatch only lowers
coverage further, `Coverage.C_le_C_zero`).  The truth's blind marginal is `𝒩(m, s_⋆²)`,
`s_⋆ = √vstar`, exactly as in `Coverage`.

## Results (all with `Γ`-free hypotheses: `P ≻ 0`, damping PSD, unit `v`)

**Step 1 — the spread lower bound** (completing the square = the quadratic-form Cauchy–Schwarz of
`NearNullDefinitional`, sharpened to carry its own equality case):
* `inv_form_defect` — the defect identity `⟪v, M⁻¹v⟫ − ⟪v, Mv⟫⁻¹ = ⟪w, Mw⟫`,
  `w = M⁻¹v − ⟪v, Mv⟫⁻¹ v`, for `M ≻ 0` and unit `v`;
* `inv_curv_le_inv_form` — hence `⟪v, Mv⟫⁻¹ ≤ ⟪v, M⁻¹v⟫`;
* `inv_form_eq_iff_eigen` — equality **iff** `v` is an eigenvector of `M`;
* `inv_curv_lt_inv_form_of_not_eigen` — strictly `<` otherwise ("the curvature alone understates
  the fiber spread").

**Step 2 — at the blind block** (`M = P₂₂`, conditioning on ALL resolved coordinates — the
fiber conditional, so no variance-coarsening step is needed):
* `inv_curvature_le_condVar` / `inv_sqrt_curvature_le_spread` — `condVar ≥ μ_v⁻¹`, so the
  advertised conditional spread is at least `μ_v^{−1/2}`;
* `condVar_eq_of_unique` — the one-dimensional blind block attains equality exactly;
* `inv_curvature_lt_condVar_of_not_eigen` — on a larger block, strict unless `v` is an
  eigenvector of the blind block.

**Step 3 — `eq:covmu`** (plugging into the machine-checked coverage law
`C = 2Φ(z·s_ρ/s_⋆) − 1` of `Coverage`/`GaussianCDF`):
* `covmu` — `2Φ(z/(s_⋆√μ_v)) − 1 ≤ 𝒩(m, s_⋆²)(m ± z·spread)` — the display, at the measure;
* `covmu_eq_of_unique` — equality for a one-dimensional blind subspace.

**Step 4 — monotone in the damping** (`P ↦ P + D`, `D ⪰ 0`; ridge `D = λ·I`):
* `curvature_le_add_damping`, `curvature_add_ridge` — damping only adds curvature (ridge: exactly `λ`);
* `rhs_antitone` / `rhs_strictAnti` / `rhs_antitone_in_damping` / `rhs_strictAnti_ridge` — the
  right side of `eq:covmu` is nonincreasing in the damping (strictly, for a genuine ridge);
* `condVar_antitone_damping` — the **Loewner antitonicity of the fiber conditional in the
  damping**, by `NearNullDefinitional.inv_antitone_form` (the variational quadratic-form
  Cauchy–Schwarz route — the house trick, reused not re-derived);
* `spread_antitone_damping`, `coverage_antitone_damping` — hence the advertised spread and the
  attained coverage itself are nonincreasing in the damping;
* `covmu_damping` — the bundle: `eq:covmu` at the damped precision, the right side's decrease,
  and the coverage's decrease, in one statement.

## Honesty

No `sorry`/`admit`/`native_decide`.  No named hypotheses remain: every step is proved from
`P ≻ 0`, `D ⪰ 0`, `v` unit, `z ≥ 0`, `vstar ≠ 0` — the paper's standing assumptions.  The single
modeling identification (advertised fiber conditional = inverse blind-block precision, its
linear-algebra half proved in `condVar_eq_schur_form`) is definitional, flagged above.
`#print axioms` on every public result at the end: only `propext`, `Classical.choice`,
`Quot.sound`.
-/
import Coverage
import NearNullDefinitional
import BlindFreeze
import Mathlib.Data.Matrix.Block
import Mathlib.Analysis.Real.Sqrt

namespace CoverageDamping

open Matrix MeasureTheory ProbabilityTheory Set GaussianCDF
open scoped NNReal

/-! ## Helpers -/

/-- `star` is trivial on real vectors (public twin of the private `NearNullDefinitional` helper). -/
theorem star_vec {ι : Type*} (w : ι → ℝ) : star w = w := by
  funext i; simp

/-- A positive-definite form vanishes only at zero. -/
theorem eq_zero_of_form_eq_zero {B : Type*} [Fintype B] {M : Matrix B B ℝ} (hM : M.PosDef)
    {w : B → ℝ} (h0 : w ⬝ᵥ (M *ᵥ w) = 0) : w = 0 := by
  by_contra hne
  have h := hM.dotProduct_mulVec_pos hne
  rw [star_vec] at h
  exact absurd h0 (ne_of_gt h)

/-- A unit vector is nonzero. -/
theorem ne_zero_of_unit {B : Type*} [Fintype B] {v : B → ℝ} (hv : v ⬝ᵥ v = 1) : v ≠ 0 := by
  intro h0
  rw [h0, zero_dotProduct] at hv
  exact zero_ne_one hv

/-! ## Step 1 — the spread lower bound, with its equality case

Completing the square: for `M ≻ 0` and unit `v`, the gap between the advertised variance
`⟪v, M⁻¹v⟫` and the curvature's reciprocal `⟪v, Mv⟫⁻¹` is itself a `M`-quadratic form — the
quadratic-form Cauchy–Schwarz of `NearNullDefinitional.form_cauchy_schwarz`, unfolded so that the
defect (hence the equality case) is explicit. -/

section Core

variable {B : Type*} [Fintype B] [DecidableEq B]

/-- **The defect identity.**  For `M ≻ 0` and a unit vector `v`,

    `⟪v, M⁻¹v⟫ − ⟪v, Mv⟫⁻¹ = ⟪w, Mw⟫`,   `w = M⁻¹v − ⟪v, Mv⟫⁻¹ v`.

The right side is `≥ 0` (PSD form) and vanishes iff `w = 0` iff `v` is an eigenvector of `M`. -/
theorem inv_form_defect {M : Matrix B B ℝ} (hM : M.PosDef) {v : B → ℝ} (hv : v ⬝ᵥ v = 1) :
    v ⬝ᵥ (M⁻¹ *ᵥ v) - (v ⬝ᵥ (M *ᵥ v))⁻¹
      = (M⁻¹ *ᵥ v - (v ⬝ᵥ (M *ᵥ v))⁻¹ • v) ⬝ᵥ
          (M *ᵥ (M⁻¹ *ᵥ v - (v ⬝ᵥ (M *ᵥ v))⁻¹ • v)) := by
  have hdet : IsUnit M.det := (Matrix.isUnit_iff_isUnit_det _).mp hM.isUnit
  have hne : v ≠ 0 := ne_zero_of_unit hv
  have hμpos : 0 < v ⬝ᵥ (M *ᵥ v) := by
    have h := hM.dotProduct_mulVec_pos hne
    rwa [star_vec] at h
  have hμne : (v ⬝ᵥ (M *ᵥ v)) ≠ 0 := ne_of_gt hμpos
  have hMu : M *ᵥ (M⁻¹ *ᵥ v) = v := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hdet, Matrix.one_mulVec]
  have hsym : Mᵀ = M := NearNullDefinitional.pd_transpose_eq hM
  have huMv : (M⁻¹ *ᵥ v) ⬝ᵥ (M *ᵥ v) = 1 := by
    rw [Matrix.dotProduct_mulVec]
    have hvm : (M⁻¹ *ᵥ v) ᵥ* M = M *ᵥ (M⁻¹ *ᵥ v) := by
      have h := Matrix.vecMul_transpose M (M⁻¹ *ᵥ v)
      rwa [hsym] at h
    rw [hvm, hMu, hv]
  rw [Matrix.mulVec_sub, Matrix.mulVec_smul, hMu]
  simp only [sub_dotProduct, dotProduct_sub, smul_dotProduct, dotProduct_smul, smul_eq_mul]
  rw [huMv, hv, dotProduct_comm (M⁻¹ *ᵥ v) v, inv_mul_cancel₀ hμne]
  ring

/-- **Step 1, the inequality**: the advertised variance along `v` is at least the curvature's
reciprocal, `⟪v, Mv⟫⁻¹ ≤ ⟪v, M⁻¹v⟫`.  (The defect is a PSD form.) -/
theorem inv_curv_le_inv_form {M : Matrix B B ℝ} (hM : M.PosDef) {v : B → ℝ}
    (hv : v ⬝ᵥ v = 1) : (v ⬝ᵥ (M *ᵥ v))⁻¹ ≤ v ⬝ᵥ (M⁻¹ *ᵥ v) := by
  have h := inv_form_defect hM hv
  have h0 := NearNullDefinitional.psd_form_nonneg hM.posSemidef
    (M⁻¹ *ᵥ v - (v ⬝ᵥ (M *ᵥ v))⁻¹ • v)
  linarith

/-- **The equality case**: `⟪v, M⁻¹v⟫ = ⟪v, Mv⟫⁻¹` **iff** `v` is an eigenvector of `M`
(with eigenvalue `⟪v, Mv⟫`).  In particular equality holds for a one-dimensional carrier, and
fails on any larger one as soon as `v` is not an eigendirection of the block. -/
theorem inv_form_eq_iff_eigen {M : Matrix B B ℝ} (hM : M.PosDef) {v : B → ℝ}
    (hv : v ⬝ᵥ v = 1) :
    v ⬝ᵥ (M⁻¹ *ᵥ v) = (v ⬝ᵥ (M *ᵥ v))⁻¹ ↔ M *ᵥ v = (v ⬝ᵥ (M *ᵥ v)) • v := by
  have hdet : IsUnit M.det := (Matrix.isUnit_iff_isUnit_det _).mp hM.isUnit
  have hne : v ≠ 0 := ne_zero_of_unit hv
  have hμpos : 0 < v ⬝ᵥ (M *ᵥ v) := by
    have h := hM.dotProduct_mulVec_pos hne
    rwa [star_vec] at h
  have hμne : (v ⬝ᵥ (M *ᵥ v)) ≠ 0 := ne_of_gt hμpos
  have hMu : M *ᵥ (M⁻¹ *ᵥ v) = v := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hdet, Matrix.one_mulVec]
  constructor
  · intro hEq
    have h := inv_form_defect hM hv
    rw [hEq, sub_self] at h
    have hw : M⁻¹ *ᵥ v - (v ⬝ᵥ (M *ᵥ v))⁻¹ • v = 0 := eq_zero_of_form_eq_zero hM h.symm
    have huv : M⁻¹ *ᵥ v = (v ⬝ᵥ (M *ᵥ v))⁻¹ • v := sub_eq_zero.mp hw
    have h3 : M *ᵥ (M⁻¹ *ᵥ v) = M *ᵥ ((v ⬝ᵥ (M *ᵥ v))⁻¹ • v) := by rw [huv]
    rw [hMu, Matrix.mulVec_smul] at h3
    have h4 : (v ⬝ᵥ (M *ᵥ v)) • v
        = (v ⬝ᵥ (M *ᵥ v)) • ((v ⬝ᵥ (M *ᵥ v))⁻¹ • (M *ᵥ v)) := by rw [← h3]
    rw [smul_smul, mul_inv_cancel₀ hμne, one_smul] at h4
    exact h4.symm
  · intro heig
    have h1 : M⁻¹ *ᵥ (M *ᵥ v) = v := by
      rw [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul _ hdet, Matrix.one_mulVec]
    rw [heig, Matrix.mulVec_smul] at h1
    have h2 : (v ⬝ᵥ (M *ᵥ v))⁻¹ • ((v ⬝ᵥ (M *ᵥ v)) • (M⁻¹ *ᵥ v))
        = (v ⬝ᵥ (M *ᵥ v))⁻¹ • v := by rw [h1]
    rw [smul_smul, inv_mul_cancel₀ hμne, one_smul] at h2
    rw [h2, dotProduct_smul, smul_eq_mul, hv, mul_one]

/-- On a carrier where `v` is **not** an eigendirection of `M`, the curvature strictly understates
the advertised variance: `⟪v, Mv⟫⁻¹ < ⟪v, M⁻¹v⟫`. -/
theorem inv_curv_lt_inv_form_of_not_eigen {M : Matrix B B ℝ} (hM : M.PosDef) {v : B → ℝ}
    (hv : v ⬝ᵥ v = 1) (hne : M *ᵥ v ≠ (v ⬝ᵥ (M *ᵥ v)) • v) :
    (v ⬝ᵥ (M *ᵥ v))⁻¹ < v ⬝ᵥ (M⁻¹ *ᵥ v) :=
  lt_of_le_of_ne (inv_curv_le_inv_form hM hv)
    (fun h => hne ((inv_form_eq_iff_eigen hM hv).mp h.symm))

omit [DecidableEq B] in
/-- **The one-dimensional carrier makes every unit vector an eigenvector.** -/
theorem unique_eigen [Unique B] (M : Matrix B B ℝ) {v : B → ℝ} (hv : v ⬝ᵥ v = 1) :
    M *ᵥ v = (v ⬝ᵥ (M *ᵥ v)) • v := by
  have hvd : v default * v default = 1 := by
    have h := hv
    simp only [dotProduct, Fintype.sum_unique] at h
    exact h
  funext i
  rw [Unique.eq_default i]
  simp only [Matrix.mulVec, dotProduct, Fintype.sum_unique, Pi.smul_apply, smul_eq_mul]
  linear_combination (-(M default default * v default)) * hvd

end Core

/-! ## Step 2 — the blind block: curvature `μ_v`, advertised conditional variance and spread -/

section Blind

variable {R B : Type*} [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B]

/-- Zero-padding of a blind-factor vector into the full space `R ⊕ B` (resolved ⊕ blind, the
`BlindFreeze` convention). -/
def embed (v : B → ℝ) : R ⊕ B → ℝ := Sum.elim 0 v

/-- The paper's **curvature `μ_v`** of the Gaussian regularizer along the blind direction `v`:
`μ_v = ⟪ι v, P (ι v)⟫` (`p:mapnearnull`: "a quadratic penalty of precision `P` has
`μ_v = vᵀPv`"). -/
def curvature (P : Matrix (R ⊕ B) (R ⊕ B) ℝ) (v : B → ℝ) : ℝ :=
  embed v ⬝ᵥ (P *ᵥ embed v)

/-- The **advertised fiber-conditional variance** along `v` given the resolved coordinates: the
`v`-form of the inverse blind-block precision.  (The modeling identification — see the module
docstring; its linear-algebra half is `condVar_eq_schur_form`.) -/
noncomputable def condVar (P : Matrix (R ⊕ B) (R ⊕ B) ℝ) (v : B → ℝ) : ℝ :=
  v ⬝ᵥ ((P.toBlocks₂₂)⁻¹ *ᵥ v)

/-- The **advertised conditional spread** `s_ρ` along `v`. -/
noncomputable def spread (P : Matrix (R ⊕ B) (R ⊕ B) ℝ) (v : B → ℝ) : ℝ :=
  Real.sqrt (condVar P v)

omit [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B] in
private lemma embed_comp_inl (v : B → ℝ) : (embed v : R ⊕ B → ℝ) ∘ Sum.inl = 0 := rfl

omit [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B] in
private lemma embed_comp_inr (v : B → ℝ) : (embed v : R ⊕ B → ℝ) ∘ Sum.inr = v := rfl

omit [DecidableEq R] [DecidableEq B] in
/-- `P` applied to an embedded blind vector, blockwise. -/
theorem mulVec_embed (P : Matrix (R ⊕ B) (R ⊕ B) ℝ) (v : B → ℝ) :
    P *ᵥ embed v = Sum.elim (P.toBlocks₁₂ *ᵥ v) (P.toBlocks₂₂ *ᵥ v) := by
  have h := Matrix.fromBlocks_mulVec P.toBlocks₁₁ P.toBlocks₁₂ P.toBlocks₂₁ P.toBlocks₂₂
    (embed v)
  rw [Matrix.fromBlocks_toBlocks, embed_comp_inl, embed_comp_inr, Matrix.mulVec_zero,
    Matrix.mulVec_zero, zero_add, zero_add] at h
  exact h

omit [DecidableEq R] [DecidableEq B] in
/-- **The curvature is exactly the blind-block form**: `μ_v = ⟪v, P₂₂ v⟫`.  (This is why no
variance-coarsening step is needed: the fiber conditional is conditioned on the full resolved
block, whose Gaussian conditional precision is exactly `P₂₂`.) -/
theorem curvature_eq_block (P : Matrix (R ⊕ B) (R ⊕ B) ℝ) (v : B → ℝ) :
    curvature P v = v ⬝ᵥ (P.toBlocks₂₂ *ᵥ v) := by
  have h : curvature P v = embed v ⬝ᵥ (P *ᵥ embed v) := rfl
  rw [h, mulVec_embed]
  show Sum.elim 0 v ⬝ᵥ Sum.elim (P.toBlocks₁₂ *ᵥ v) (P.toBlocks₂₂ *ᵥ v) = _
  rw [sumElim_dotProduct_sumElim, zero_dotProduct, zero_add]

omit [DecidableEq R] [DecidableEq B] in
/-- Embedding preserves the dot product. -/
theorem embed_dotProduct_embed (v : B → ℝ) :
    (embed v : R ⊕ B → ℝ) ⬝ᵥ embed v = v ⬝ᵥ v := by
  show Sum.elim 0 v ⬝ᵥ Sum.elim 0 v = _
  rw [sumElim_dotProduct_sumElim, zero_dotProduct, zero_add]

omit [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B] in
/-- Embedding preserves nonvanishing. -/
theorem embed_ne_zero {v : B → ℝ} (hv : v ≠ 0) : (embed v : R ⊕ B → ℝ) ≠ 0 := by
  intro h
  apply hv
  funext b
  exact congrFun h (Sum.inr b)

omit [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B] in
/-- The blind block is the `Sum.inr` principal submatrix. -/
theorem toBlocks₂₂_eq_submatrix (M : Matrix (R ⊕ B) (R ⊕ B) ℝ) :
    M.toBlocks₂₂ = M.submatrix Sum.inr Sum.inr := rfl

omit [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B] in
/-- The blind block of a positive-definite matrix is positive definite. -/
theorem toBlocks₂₂_posDef {P : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef) :
    (P.toBlocks₂₂).PosDef := by
  rw [toBlocks₂₂_eq_submatrix]
  exact hP.submatrix Sum.inr_injective

omit [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B] in
/-- The blind block of a positive-semidefinite matrix is positive semidefinite. -/
theorem toBlocks₂₂_posSemidef {D : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hD : D.PosSemidef) :
    (D.toBlocks₂₂).PosSemidef := by
  rw [toBlocks₂₂_eq_submatrix]
  exact hD.submatrix Sum.inr

omit [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B] in
/-- Taking the blind block commutes with addition. -/
theorem toBlocks₂₂_add (P D : Matrix (R ⊕ B) (R ⊕ B) ℝ) :
    (P + D).toBlocks₂₂ = P.toBlocks₂₂ + D.toBlocks₂₂ := by
  ext i j
  simp [Matrix.toBlocks₂₂]

omit [DecidableEq R] [DecidableEq B] in
/-- The curvature is positive along any nonzero blind direction (`P ≻ 0`). -/
theorem curvature_pos {P : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef) {v : B → ℝ}
    (hv : v ≠ 0) : 0 < curvature P v := by
  have h := hP.dotProduct_mulVec_pos (embed_ne_zero hv)
  rw [star_vec] at h
  exact h

omit [DecidableEq R] in
/-- **The spread lower bound at the blind block**: the advertised fiber-conditional variance is at
least the curvature's reciprocal, `μ_v⁻¹ ≤ condVar`. -/
theorem inv_curvature_le_condVar {P : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef) {v : B → ℝ}
    (hv : v ⬝ᵥ v = 1) : (curvature P v)⁻¹ ≤ condVar P v := by
  rw [curvature_eq_block]
  exact inv_curv_le_inv_form (toBlocks₂₂_posDef hP) hv

omit [DecidableEq R] in
/-- **"…advertises a conditional spread of at least `μ_v^{−1/2}`"**: `μ_v^{−1/2} ≤ s_ρ`. -/
theorem inv_sqrt_curvature_le_spread {P : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef)
    {v : B → ℝ} (hv : v ⬝ᵥ v = 1) :
    (Real.sqrt (curvature P v))⁻¹ ≤ spread P v := by
  have h := inv_curvature_le_condVar hP hv
  have h2 : Real.sqrt (curvature P v)⁻¹ ≤ Real.sqrt (condVar P v) := Real.sqrt_le_sqrt h
  rwa [Real.sqrt_inv] at h2

omit [DecidableEq R] in
/-- **The one-dimensional blind subspace attains the bound exactly**: `condVar = μ_v⁻¹`. -/
theorem condVar_eq_of_unique [Unique B] {P : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef)
    {v : B → ℝ} (hv : v ⬝ᵥ v = 1) : condVar P v = (curvature P v)⁻¹ := by
  rw [curvature_eq_block]
  exact (inv_form_eq_iff_eigen (toBlocks₂₂_posDef hP) hv).mpr (unique_eigen _ hv)

omit [DecidableEq R] in
/-- **"…on a larger one the curvature alone understates the fiber spread"**: strictly, whenever
`v` is not an eigendirection of the blind block. -/
theorem inv_curvature_lt_condVar_of_not_eigen {P : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef)
    {v : B → ℝ} (hv : v ⬝ᵥ v = 1) (hne : P.toBlocks₂₂ *ᵥ v ≠ curvature P v • v) :
    (curvature P v)⁻¹ < condVar P v := by
  rw [curvature_eq_block] at hne ⊢
  exact inv_curv_lt_inv_form_of_not_eigen (toBlocks₂₂_posDef hP) hv hne

/-- **The modeling identification's linear-algebra half, proved.**  For a block covariance
`Σ = [[A, B], [C, D]]` (resolved ⊕ blind, `A` and `Σ` invertible — automatic for `Σ ≻ 0`), the
advertised fiber-conditional variance built from the *precision* `Σ⁻¹` is exactly the `v`-form of
the **Schur complement** `D − C A⁻¹ B`, the Gaussian fiber-conditional covariance.  (Reuses the
machine-checked `BlindFreeze.invBlock₂₂_eq_schur_inv` of `p:blind`.) -/
theorem condVar_eq_schur_form (Am : Matrix R R ℝ) (Bm : Matrix R B ℝ) (Cm : Matrix B R ℝ)
    (Dm : Matrix B B ℝ) [Invertible Am] [Invertible (Matrix.fromBlocks Am Bm Cm Dm)]
    (v : B → ℝ) :
    condVar (Matrix.fromBlocks Am Bm Cm Dm)⁻¹ v = v ⬝ᵥ ((Dm - Cm * Am⁻¹ * Bm) *ᵥ v) := by
  haveI hS : Invertible (Dm - Cm * Am⁻¹ * Bm) := by
    have h := Matrix.invertibleOfFromBlocks₁₁Invertible Am Bm Cm Dm
    rwa [Matrix.invOf_eq_nonsing_inv] at h
  show v ⬝ᵥ (((Matrix.fromBlocks Am Bm Cm Dm)⁻¹.toBlocks₂₂)⁻¹ *ᵥ v) = _
  rw [BlindFreeze.invBlock₂₂_eq_schur_inv Am Bm Cm Dm, Matrix.inv_inv_of_invertible]

end Blind

/-! ## Step 3 — `eq:covmu`: plug the spread bound into the machine-checked coverage law -/

section Covmu

variable {R B : Type*} [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B]

/-- The aligned coverage closed form, specialized from `Coverage.coverage_eq` + `Coverage.C_zero`:
a band `m ± z·s_ρ` centered at the truth's mean covers `𝒩(m, s_⋆²)` with probability
`2Φ(z·s_ρ/s_⋆) − 1`. -/
theorem aligned_coverage_eq (m z sreg : ℝ) {vstar : ℝ≥0} (hvstar : vstar ≠ 0) (hz : 0 ≤ z)
    (hsreg : 0 ≤ sreg) :
    (gaussianReal m vstar).real (Icc (m - z * sreg) (m + z * sreg))
      = 2 * Φ (z * sreg / Real.sqrt vstar) - 1 := by
  rw [Coverage.coverage_eq m m z sreg hvstar hz hsreg, sub_self, zero_div, Coverage.C_zero]

private lemma div_mul_arg (z a b : ℝ) : z / (a * b) = z * b⁻¹ / a := by
  rw [div_mul_eq_div_div_swap, div_eq_mul_inv z b]

/-- **The plug-in core of `eq:covmu`**: any advertised spread `s_ρ ≥ μ^{-1/2}` yields aligned
coverage at least `2Φ(z/(s_⋆√μ)) − 1` (`Φ` monotone — reused, not re-derived). -/
theorem rhs_le_aligned {z sstar srho mu : ℝ} (hz : 0 ≤ z) (hs : 0 < sstar)
    (hsp : (Real.sqrt mu)⁻¹ ≤ srho) :
    2 * Φ (z / (sstar * Real.sqrt mu)) - 1 ≤ 2 * Φ (z * srho / sstar) - 1 := by
  have hkey : z / (sstar * Real.sqrt mu) ≤ z * srho / sstar := by
    rw [div_mul_arg z sstar (Real.sqrt mu)]
    exact (div_le_div_iff_of_pos_right hs).mpr (mul_le_mul_of_nonneg_left hsp hz)
  have := GaussianCDF.monotone_Φ hkey
  linarith

omit [DecidableEq R] in
/-- **`eq:covmu`.**  A Gaussian regularizer of precision `P ≻ 0` with curvature `μ_v` along a unit
blind direction `v` advertises the central band `m ± z·s_ρ`; against a truth of blind marginal
`𝒩(m, s_⋆²)` (`s_⋆ = √vstar`, aligned) the coverage it attains obeys

    `C ≥ 2Φ(z/(s_⋆ √μ_v)) − 1`. -/
theorem covmu {P : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef) {v : B → ℝ} (hv : v ⬝ᵥ v = 1)
    {z : ℝ} (hz : 0 ≤ z) {vstar : ℝ≥0} (hvstar : vstar ≠ 0) (m : ℝ) :
    2 * Φ (z / (Real.sqrt vstar * Real.sqrt (curvature P v))) - 1
      ≤ (gaussianReal m vstar).real (Icc (m - z * spread P v) (m + z * spread P v)) := by
  rw [aligned_coverage_eq m z (spread P v) hvstar hz (Real.sqrt_nonneg _)]
  exact rhs_le_aligned hz
    (Real.sqrt_pos.mpr (NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hvstar)))
    (inv_sqrt_curvature_le_spread hP hv)

omit [DecidableEq R] in
/-- **"…with equality when the blind subspace is one-dimensional."**  For `dim B = 1` the display
is an identity: the attained coverage is exactly `2Φ(z/(s_⋆√μ_v)) − 1`. -/
theorem covmu_eq_of_unique [Unique B] {P : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef)
    {v : B → ℝ} (hv : v ⬝ᵥ v = 1) {z : ℝ} (hz : 0 ≤ z) {vstar : ℝ≥0} (hvstar : vstar ≠ 0)
    (m : ℝ) :
    (gaussianReal m vstar).real (Icc (m - z * spread P v) (m + z * spread P v))
      = 2 * Φ (z / (Real.sqrt vstar * Real.sqrt (curvature P v))) - 1 := by
  rw [aligned_coverage_eq m z (spread P v) hvstar hz (Real.sqrt_nonneg _)]
  have hsp : spread P v = (Real.sqrt (curvature P v))⁻¹ := by
    unfold spread
    rw [condVar_eq_of_unique hP hv, Real.sqrt_inv]
  rw [hsp, ← div_mul_arg z (Real.sqrt vstar) (Real.sqrt (curvature P v))]

end Covmu

/-! ## Step 4 — the monotonicity reading: both sides fall as the damping grows -/

section Damping

variable {R B : Type*} [Fintype R] [DecidableEq R] [Fintype B] [DecidableEq B]

omit [DecidableEq R] [DecidableEq B] in
/-- **Damping only adds curvature**: `μ_v(P) ≤ μ_v(P + D)` for any PSD damping `D`. -/
theorem curvature_le_add_damping {D : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hD : D.PosSemidef)
    (P : Matrix (R ⊕ B) (R ⊕ B) ℝ) (v : B → ℝ) :
    curvature P v ≤ curvature (P + D) v := by
  have hD0 := NearNullDefinitional.psd_form_nonneg hD (embed v : R ⊕ B → ℝ)
  show embed v ⬝ᵥ (P *ᵥ embed v) ≤ embed v ⬝ᵥ ((P + D) *ᵥ embed v)
  rw [Matrix.add_mulVec, dotProduct_add]
  linarith

/-- **The ridge adds exactly `λ`**: `μ_v(P + λI) = μ_v(P) + λ` for unit `v`. -/
theorem curvature_add_ridge (P : Matrix (R ⊕ B) (R ⊕ B) ℝ) {v : B → ℝ} (hv : v ⬝ᵥ v = 1)
    (lam : ℝ) :
    curvature (P + lam • 1) v = curvature P v + lam := by
  show embed v ⬝ᵥ ((P + lam • 1) *ᵥ embed v) = curvature P v + lam
  rw [Matrix.add_mulVec, dotProduct_add, Matrix.smul_mulVec, Matrix.one_mulVec,
    dotProduct_smul, smul_eq_mul, embed_dotProduct_embed, hv, mul_one]
  rfl

/-- **The right side of `eq:covmu` is antitone in the curvature** (`Φ` monotone). -/
theorem rhs_antitone {z sstar mu1 mu2 : ℝ} (hz : 0 ≤ z) (hs : 0 < sstar) (h1 : 0 < mu1)
    (h12 : mu1 ≤ mu2) :
    2 * Φ (z / (sstar * Real.sqrt mu2)) - 1 ≤ 2 * Φ (z / (sstar * Real.sqrt mu1)) - 1 := by
  have hd1 : 0 < sstar * Real.sqrt mu1 := mul_pos hs (Real.sqrt_pos.mpr h1)
  have hkey : z / (sstar * Real.sqrt mu2) ≤ z / (sstar * Real.sqrt mu1) := by
    gcongr
  have := GaussianCDF.monotone_Φ hkey
  linarith

/-- …strictly, for a genuine level `z > 0` and a genuine curvature increase (`Φ` strictly
monotone). -/
theorem rhs_strictAnti {z sstar mu1 mu2 : ℝ} (hz : 0 < z) (hs : 0 < sstar) (h1 : 0 < mu1)
    (hlt : mu1 < mu2) :
    2 * Φ (z / (sstar * Real.sqrt mu2)) - 1 < 2 * Φ (z / (sstar * Real.sqrt mu1)) - 1 := by
  have hd1 : 0 < sstar * Real.sqrt mu1 := mul_pos hs (Real.sqrt_pos.mpr h1)
  have hdlt : sstar * Real.sqrt mu1 < sstar * Real.sqrt mu2 :=
    mul_lt_mul_of_pos_left (Real.sqrt_lt_sqrt h1.le hlt) hs
  have hkey : z / (sstar * Real.sqrt mu2) < z / (sstar * Real.sqrt mu1) :=
    div_lt_div_of_pos_left hz hd1 hdlt
  have := GaussianCDF.strictMono_Φ hkey
  linarith

omit [DecidableEq R] [DecidableEq B] in
/-- **"The right side is a decreasing function of the damping"**: `P ↦ P + D`, `D ⪰ 0`, lowers
the right side of `eq:covmu`. -/
theorem rhs_antitone_in_damping {P D : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef)
    (hD : D.PosSemidef) {v : B → ℝ} (hv0 : v ≠ 0) {z sstar : ℝ} (hz : 0 ≤ z)
    (hs : 0 < sstar) :
    2 * Φ (z / (sstar * Real.sqrt (curvature (P + D) v))) - 1
      ≤ 2 * Φ (z / (sstar * Real.sqrt (curvature P v))) - 1 :=
  rhs_antitone hz hs (curvature_pos hP hv0) (curvature_le_add_damping hD P v)

/-- …and strictly so for a genuine ridge `λ > 0` at a genuine level `z > 0`. -/
theorem rhs_strictAnti_ridge {P : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef) {v : B → ℝ}
    (hv : v ⬝ᵥ v = 1) {z sstar lam : ℝ} (hz : 0 < z) (hs : 0 < sstar) (hlam : 0 < lam) :
    2 * Φ (z / (sstar * Real.sqrt (curvature (P + lam • 1) v))) - 1
      < 2 * Φ (z / (sstar * Real.sqrt (curvature P v))) - 1 := by
  rw [curvature_add_ridge P hv lam]
  exact rhs_strictAnti hz hs (curvature_pos hP (ne_zero_of_unit hv))
    (lt_add_of_pos_right _ hlam)

omit [Fintype R] [DecidableEq R] in
/-- **The Loewner monotonicity of the fiber conditional in the damping.**  Adding a PSD damping
to the precision can only shrink the advertised fiber-conditional variance:
`condVar (P + D) ≤ condVar P` — via `NearNullDefinitional.inv_antitone_form`, the variational
quadratic-form Cauchy–Schwarz route (no spectral theorem). -/
theorem condVar_antitone_damping {P D : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef)
    (hD : D.PosSemidef) (v : B → ℝ) :
    condVar (P + D) v ≤ condVar P v := by
  have hX : (P.toBlocks₂₂).PosDef := toBlocks₂₂_posDef hP
  have hDb : (D.toBlocks₂₂).PosSemidef := toBlocks₂₂_posSemidef hD
  have hY : ((P + D).toBlocks₂₂).PosDef := by
    rw [toBlocks₂₂_add]
    exact hX.add_posSemidef hDb
  have hle : ∀ w : B → ℝ,
      w ⬝ᵥ (P.toBlocks₂₂ *ᵥ w) ≤ w ⬝ᵥ ((P + D).toBlocks₂₂ *ᵥ w) := by
    intro w
    rw [toBlocks₂₂_add, Matrix.add_mulVec, dotProduct_add]
    have := NearNullDefinitional.psd_form_nonneg hDb w
    linarith
  exact NearNullDefinitional.inv_antitone_form hX hY hle v

omit [Fintype R] [DecidableEq R] in
/-- Hence the advertised conditional spread is nonincreasing in the damping. -/
theorem spread_antitone_damping {P D : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef)
    (hD : D.PosSemidef) (v : B → ℝ) :
    spread (P + D) v ≤ spread P v :=
  Real.sqrt_le_sqrt (condVar_antitone_damping hP hD v)

omit [Fintype R] [DecidableEq R] in
/-- **"…and so is the coverage itself."**  The coverage the damped band attains against the same
truth is at most the undamped band's (aligned; `Φ` monotone through the coverage law). -/
theorem coverage_antitone_damping {P D : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef)
    (hD : D.PosSemidef) (v : B → ℝ) {z : ℝ} (hz : 0 ≤ z) {vstar : ℝ≥0} (hvstar : vstar ≠ 0)
    (m : ℝ) :
    (gaussianReal m vstar).real
        (Icc (m - z * spread (P + D) v) (m + z * spread (P + D) v))
      ≤ (gaussianReal m vstar).real (Icc (m - z * spread P v) (m + z * spread P v)) := by
  rw [aligned_coverage_eq m z (spread (P + D) v) hvstar hz (Real.sqrt_nonneg _),
    aligned_coverage_eq m z (spread P v) hvstar hz (Real.sqrt_nonneg _)]
  have hsstar : 0 < Real.sqrt vstar :=
    Real.sqrt_pos.mpr (NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hvstar))
  have hkey : z * spread (P + D) v / Real.sqrt vstar ≤ z * spread P v / Real.sqrt vstar :=
    (div_le_div_iff_of_pos_right hsstar).mpr
      (mul_le_mul_of_nonneg_left (spread_antitone_damping hP hD v) hz)
  have := GaussianCDF.monotone_Φ hkey
  linarith

omit [DecidableEq R] in
/-- **The capstone bundle** — `eq:covmu` and its monotonicity reading in one statement.  For
`P ≻ 0`, PSD damping `D`, unit blind `v`, level `z ≥ 0`, truth `𝒩(m, s_⋆²)` (`s_⋆ = √vstar`):
(i) the damped band still obeys `eq:covmu`; (ii) the right side of `eq:covmu` has fallen; and
(iii) the attained coverage itself has fallen — stability and blind calibration trade off along
the single damping axis. -/
theorem covmu_damping {P D : Matrix (R ⊕ B) (R ⊕ B) ℝ} (hP : P.PosDef) (hD : D.PosSemidef)
    {v : B → ℝ} (hv : v ⬝ᵥ v = 1) {z : ℝ} (hz : 0 ≤ z) {vstar : ℝ≥0} (hvstar : vstar ≠ 0)
    (m : ℝ) :
    (2 * Φ (z / (Real.sqrt vstar * Real.sqrt (curvature (P + D) v))) - 1
        ≤ (gaussianReal m vstar).real
            (Icc (m - z * spread (P + D) v) (m + z * spread (P + D) v)))
    ∧ (2 * Φ (z / (Real.sqrt vstar * Real.sqrt (curvature (P + D) v))) - 1
        ≤ 2 * Φ (z / (Real.sqrt vstar * Real.sqrt (curvature P v))) - 1)
    ∧ ((gaussianReal m vstar).real
          (Icc (m - z * spread (P + D) v) (m + z * spread (P + D) v))
        ≤ (gaussianReal m vstar).real
            (Icc (m - z * spread P v) (m + z * spread P v))) := by
  have hPD : (P + D).PosDef := hP.add_posSemidef hD
  have hsstar : 0 < Real.sqrt vstar :=
    Real.sqrt_pos.mpr (NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hvstar))
  exact ⟨covmu hPD hv hz hvstar m,
    rhs_antitone_in_damping hP hD (ne_zero_of_unit hv) hz hsstar,
    coverage_antitone_damping hP hD v hz hvstar m⟩

end Damping

end CoverageDamping

-- #print axioms audit (eq:covmu and its monotonicity reading)
#print axioms CoverageDamping.star_vec
#print axioms CoverageDamping.eq_zero_of_form_eq_zero
#print axioms CoverageDamping.ne_zero_of_unit
#print axioms CoverageDamping.inv_form_defect
#print axioms CoverageDamping.inv_curv_le_inv_form
#print axioms CoverageDamping.inv_form_eq_iff_eigen
#print axioms CoverageDamping.inv_curv_lt_inv_form_of_not_eigen
#print axioms CoverageDamping.unique_eigen
#print axioms CoverageDamping.mulVec_embed
#print axioms CoverageDamping.curvature_eq_block
#print axioms CoverageDamping.embed_dotProduct_embed
#print axioms CoverageDamping.embed_ne_zero
#print axioms CoverageDamping.toBlocks₂₂_eq_submatrix
#print axioms CoverageDamping.toBlocks₂₂_posDef
#print axioms CoverageDamping.toBlocks₂₂_posSemidef
#print axioms CoverageDamping.toBlocks₂₂_add
#print axioms CoverageDamping.curvature_pos
#print axioms CoverageDamping.inv_curvature_le_condVar
#print axioms CoverageDamping.inv_sqrt_curvature_le_spread
#print axioms CoverageDamping.condVar_eq_of_unique
#print axioms CoverageDamping.inv_curvature_lt_condVar_of_not_eigen
#print axioms CoverageDamping.condVar_eq_schur_form
#print axioms CoverageDamping.aligned_coverage_eq
#print axioms CoverageDamping.rhs_le_aligned
#print axioms CoverageDamping.covmu
#print axioms CoverageDamping.covmu_eq_of_unique
#print axioms CoverageDamping.curvature_le_add_damping
#print axioms CoverageDamping.curvature_add_ridge
#print axioms CoverageDamping.rhs_antitone
#print axioms CoverageDamping.rhs_strictAnti
#print axioms CoverageDamping.rhs_antitone_in_damping
#print axioms CoverageDamping.rhs_strictAnti_ridge
#print axioms CoverageDamping.condVar_antitone_damping
#print axioms CoverageDamping.spread_antitone_damping
#print axioms CoverageDamping.coverage_antitone_damping
#print axioms CoverageDamping.covmu_damping
