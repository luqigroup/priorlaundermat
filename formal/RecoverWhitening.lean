/-
# `p:recover` (ii) — the whitening/congruence reduction (the appendix's step)

The README records the one unchecked step of `p:recover` (ii): "the diagonalized recursion and the
matrix floor are both checked; the congruence between them is the appendix's."  This module
machine-checks that congruence — the passage from the loop's model-space covariance step to the
whitened autonomous data-space recursion that `RecoverConvergence.lean` (scalar facts) and
`MatrixFloor.lean` (variational floor) consume — plus the pseudoinverse lift back to the resolved
block.  Nothing proved in those modules is re-proved here.

The paper's chain (proof of `p:recover` (ii), `paper/v2/manuscript.tex`), each step a theorem:

1. **Woodbury rearrangement** (`woodbury_rearrangement`): with the covariance-form posterior
   `Σ_post = Σ − G A Σ`, `G = Σ Aᵀ (D+Γ)⁻¹`, `D = A Σ Aᵀ`, one has
   `A Σ_post Aᵀ = D (D+Γ)⁻¹ Γ` — the same two-Woodbury algebra as
   `NearNullDefinitional.a_spost_at`, here in the `D S⁻¹ Γ` form the recursion needs.
2. **Projection to the autonomous recursion** (`A_mul_gain`, `data_step_autonomous`,
   `mean_step`): `A G = K := D(D+Γ)⁻¹`, and for `Σ' = Σ_post + G S_y Gᵀ` the data-space image is
   `A Σ' Aᵀ = K Γ + K S_y Kᵀ` — autonomous in `D`; the mean step projects to
   `A μ' − A μ⋆ = (I − K)(A μ − A μ⋆)`, with `I − K = (D̃+I)⁻¹` in whitened form (`one_sub_K`),
   the operator whose norm bound `RecoverConvergence.one_sub_K_le` caps.
3. **The whitening congruence** (`K_whiten`, `K_whiten_symm`, `whiten_intertwines`,
   `whiten_data_cov`, capstones `whitened_autonomous_step`, `whitened_autonomous_step_std`):
   conjugation by any symmetric invertible `R` with `R R = Γ⁻¹` (a `Γ^{-1/2}`) intertwines the
   recursion: `D̃ = R D R` obeys `D̃' = K̃ + K̃ S̃ K̃`, `K̃ = D̃(D̃+I)⁻¹ = R K R⁻¹` symmetric,
   `S̃ = I + R D⋆ R`.  The concrete `R` is `stdWhitener Γ = CFC.sqrt Γ⁻¹` (the same CFC square
   root `DataCovarianceSqrt.lean` built), with its three properties proved
   (`stdWhitener_transpose`, `stdWhitener_mul_self`, `stdWhitener_isUnit_det`).  The capstone's
   only hypotheses are the proposition's own: `Γ ≻ 0` and `Σ ⪰ 0` (singular `Σ_reg` admissible,
   as the paper notes).  `whitened_step_quadratic_form` and `exists_gain_witness` expose the
   whitened iterate in exactly the shapes `MatrixFloor.floor_step` / `LowerBound_gain` consume,
   and `fixed_point_unique` closes the paper's "the interior fixed point is unique" cancellation:
   an invertible fixed point of `D ↦ K + K S̃ K` satisfies `D + I = S̃`.
4. **The pseudoinverse lift** (`pinv`, `projR`, `resolved_block`, `tendsto_resolved_covariance`,
   `tendsto_resolved_mean`): in the full-row-rank case the proposition assumes, `A⁺ = Aᵀ(AAᵀ)⁻¹`
   explicitly; `P_R = A⁺A` is the orthogonal projector onto `row(A)` (symmetric, idempotent,
   `A P_R = A`, `P_R Aᵀ = Aᵀ`, kernel-annihilating), `P_R Σ P_R = A⁺ (AΣAᵀ) (A⁺)ᵀ` with **no**
   invertibility hypothesis, and convergence of `D_t` (resp. `A μ_t`) lifts to convergence of
   `P_R Σ_t P_R` (resp. `P_R μ_t`) by continuity.

**Hypothesis encoding.**  Full row rank is carried as `IsUnit (A * Aᵀ).det` — equivalent for real
matrices and exactly what the explicit `A⁺` needs.  No named (unproved) hypotheses remain: every
theorem's assumptions are the proposition's standing ones (`Γ ≻ 0`, `Σ ⪰ 0`, full row rank) or
definitional relations proved elsewhere in the file.  What is *not* in scope here (and stays the
paper's): the EM/NPMLE descent-and-compactness argument that upgrades the floor + unique fixed
point to convergence of the matrix iterates, and the norm bound `‖D̃'‖ ≤ 1 + ‖S̃‖`.

No `sorry`/`admit`/`native_decide`; `#print axioms` on every public result at the end.
-/
import Mathlib
import NearNullDefinitional

namespace RecoverWhitening

open Matrix Filter
open scoped MatrixOrder Topology

variable {p n : Type*} [Fintype p] [DecidableEq p] [Fintype n] [DecidableEq n]

/-! ## The loop's covariance step and its data-space projection (paper steps 1–2) -/

/-- The data-space image `D = A Σ Aᵀ` of a model-space covariance. -/
def Dmat (A : Matrix p n ℝ) (Sig : Matrix n n ℝ) : Matrix p p ℝ := A * Sig * Aᵀ

/-- The data-space gain `K = D (D + Γ)⁻¹` of one loop round. -/
noncomputable def Kmat (D Gam : Matrix p p ℝ) : Matrix p p ℝ := D * (D + Gam)⁻¹

/-- The loop's model-space gain `G = Σ Aᵀ (D + Γ)⁻¹` — the covariance form, defined whenever
`D + Γ` is invertible, so a singular `Σ` is admissible (the paper's remark). -/
noncomputable def gainCov (A : Matrix p n ℝ) (Sig : Matrix n n ℝ) (Gam : Matrix p p ℝ) :
    Matrix n p ℝ := Sig * Aᵀ * (Dmat A Sig + Gam)⁻¹

/-- The posterior covariance in covariance form, `Σ_post = Σ − G A Σ`. -/
noncomputable def postCov (A : Matrix p n ℝ) (Sig : Matrix n n ℝ) (Gam : Matrix p p ℝ) :
    Matrix n n ℝ := Sig - gainCov A Sig Gam * A * Sig

/-- One covariance step of the loop: `Σ' = Σ_post + G S_y Gᵀ`. -/
noncomputable def nextCov (A : Matrix p n ℝ) (Sig : Matrix n n ℝ) (Gam Sy : Matrix p p ℝ) :
    Matrix n n ℝ := postCov A Sig Gam + gainCov A Sig Gam * Sy * (gainCov A Sig Gam)ᵀ

/-- **Step 1, the Woodbury rearrangement.**  `A Σ_post Aᵀ = D (D+Γ)⁻¹ Γ` for `D = A Σ Aᵀ`:
the same `D − D S⁻¹ D = D S⁻¹ (S − D)` cancellation as `NearNullDefinitional.a_spost_at`, in
the gain form the recursion needs. -/
theorem woodbury_rearrangement (A : Matrix p n ℝ) (Sig : Matrix n n ℝ) (Gam : Matrix p p ℝ)
    (hS : IsUnit (Dmat A Sig + Gam).det) :
    Dmat A (postCov A Sig Gam) = Kmat (Dmat A Sig) Gam * Gam := by
  have hexp : Dmat A (postCov A Sig Gam)
      = Dmat A Sig - Dmat A Sig * ((Dmat A Sig + Gam)⁻¹ * Dmat A Sig) := by
    simp only [Dmat, postCov, gainCov, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_assoc]
  rw [hexp]
  calc Dmat A Sig - Dmat A Sig * ((Dmat A Sig + Gam)⁻¹ * Dmat A Sig)
      = Dmat A Sig * ((Dmat A Sig + Gam)⁻¹ * (Dmat A Sig + Gam))
        - Dmat A Sig * ((Dmat A Sig + Gam)⁻¹ * Dmat A Sig) := by
        rw [Matrix.nonsing_inv_mul _ hS, Matrix.mul_one]
    _ = Dmat A Sig * ((Dmat A Sig + Gam)⁻¹ * Gam) := by
        rw [← Matrix.mul_sub, ← Matrix.mul_sub, add_sub_cancel_left]
    _ = Kmat (Dmat A Sig) Gam * Gam := by
        simp only [Kmat, Matrix.mul_assoc]

/-- **`A G = K`**: the model-space gain projects to the data-space gain (pure associativity). -/
theorem A_mul_gain (A : Matrix p n ℝ) (Sig : Matrix n n ℝ) (Gam : Matrix p p ℝ) :
    A * gainCov A Sig Gam = Kmat (Dmat A Sig) Gam := by
  simp only [gainCov, Kmat, Dmat, Matrix.mul_assoc]

/-- **Step 2, the projection to the autonomous recursion.**  The data-space image of one
covariance step depends on `Σ` only through `D = A Σ Aᵀ`:
`A Σ' Aᵀ = K Γ + K S_y Kᵀ` with `K = D (D+Γ)⁻¹` — the paper's `D_{t+1} = Ξ(D_t)`. -/
theorem data_step_autonomous (A : Matrix p n ℝ) (Sig : Matrix n n ℝ) (Gam Sy : Matrix p p ℝ)
    (hS : IsUnit (Dmat A Sig + Gam).det) :
    Dmat A (nextCov A Sig Gam Sy)
      = Kmat (Dmat A Sig) Gam * Gam
        + Kmat (Dmat A Sig) Gam * Sy * (Kmat (Dmat A Sig) Gam)ᵀ := by
  have hsplit : Dmat A (nextCov A Sig Gam Sy)
      = Dmat A (postCov A Sig Gam)
        + A * (gainCov A Sig Gam * Sy * (gainCov A Sig Gam)ᵀ) * Aᵀ := by
    simp only [Dmat, nextCov, Matrix.mul_add, Matrix.add_mul]
  have hsecond : A * (gainCov A Sig Gam * Sy * (gainCov A Sig Gam)ᵀ) * Aᵀ
      = Kmat (Dmat A Sig) Gam * Sy * (Kmat (Dmat A Sig) Gam)ᵀ := by
    rw [← A_mul_gain A Sig Gam, Matrix.transpose_mul]
    simp only [Matrix.mul_assoc]
  rw [hsplit, hsecond, woodbury_rearrangement A Sig Gam hS]

/-- **The mean step projects.**  For `μ' = μ + G (A μ⋆ − A μ)`,
`A μ' − A μ⋆ = (I − K)(A μ − A μ⋆)` — the paper's mean recursion, autonomous in data space. -/
theorem mean_step (A : Matrix p n ℝ) (Sig : Matrix n n ℝ) (Gam : Matrix p p ℝ)
    (mu mustar : n → ℝ) :
    A *ᵥ (mu + gainCov A Sig Gam *ᵥ (A *ᵥ mustar - A *ᵥ mu)) - A *ᵥ mustar
      = (1 - Kmat (Dmat A Sig) Gam) *ᵥ (A *ᵥ mu - A *ᵥ mustar) := by
  rw [Matrix.mulVec_add, Matrix.mulVec_mulVec, A_mul_gain A Sig Gam]
  simp only [Matrix.sub_mulVec, Matrix.one_mulVec, Matrix.mulVec_sub]
  abel

/-- **`I − K̃ = (D̃ + I)⁻¹`** — the contraction factor of the whitened mean step, whose scalar
norm bound is `RecoverConvergence.one_sub_K_le`. -/
theorem one_sub_K (D : Matrix p p ℝ) (hD1 : IsUnit (D + 1).det) :
    1 - Kmat D 1 = (D + 1)⁻¹ := by
  calc 1 - Kmat D 1 = (D + 1) * (D + 1)⁻¹ - D * (D + 1)⁻¹ := by
        rw [Matrix.mul_nonsing_inv _ hD1]
        simp only [Kmat]
    _ = ((D + 1) - D) * (D + 1)⁻¹ := by rw [Matrix.sub_mul]
    _ = (D + 1)⁻¹ := by rw [add_sub_cancel_left, Matrix.one_mul]

/-! ## Step 3: the whitening congruence

Everything is parameterized over any symmetric invertible `R` with `R R = Γ⁻¹` (a `Γ^{-1/2}`);
the concrete `stdWhitener` below instantiates it via `CFC.sqrt`, as `DataCovarianceSqrt.lean`
did for the data covariance. -/

/-- Congruence by a whitener: `M ↦ R M R`. -/
def whiten (R M : Matrix p p ℝ) : Matrix p p ℝ := R * M * R

/-- The defining cancellation `R Γ R = I` of a whitener. -/
theorem R_mul_Gam_mul_R (R Gam : Matrix p p ℝ) (hRdet : IsUnit R.det)
    (hRR : R * R = Gam⁻¹) (hGdet : IsUnit Gam.det) : R * Gam * R = 1 := by
  have hGamEq : Gam = R⁻¹ * R⁻¹ := by
    have h : Gam⁻¹⁻¹ = (R * R)⁻¹ := by rw [hRR]
    rw [Matrix.nonsing_inv_nonsing_inv _ hGdet, Matrix.mul_inv_rev] at h
    exact h
  rw [hGamEq]
  calc R * (R⁻¹ * R⁻¹) * R = R * R⁻¹ * (R⁻¹ * R) := by simp only [Matrix.mul_assoc]
    _ = 1 := by
        rw [Matrix.mul_nonsing_inv _ hRdet, Matrix.nonsing_inv_mul _ hRdet, Matrix.one_mul]

/-- Whitening shifts the noise to the identity: `D̃ + I = R (D + Γ) R`. -/
theorem whiten_add_one (R Gam : Matrix p p ℝ) (hRdet : IsUnit R.det)
    (hRR : R * R = Gam⁻¹) (hGdet : IsUnit Gam.det) (D : Matrix p p ℝ) :
    whiten R D + 1 = R * (D + Gam) * R := by
  simp only [whiten]
  rw [Matrix.mul_add, Matrix.add_mul, R_mul_Gam_mul_R R Gam hRdet hRR hGdet]

/-- Invertibility transfers through the congruence: `D + Γ` invertible makes `D̃ + I`
invertible. -/
theorem isUnit_whiten_shift_det (R Gam : Matrix p p ℝ) (hRdet : IsUnit R.det)
    (hRR : R * R = Gam⁻¹) (hGdet : IsUnit Gam.det) (D : Matrix p p ℝ)
    (hS : IsUnit (D + Gam).det) : IsUnit (whiten R D + 1).det := by
  rw [whiten_add_one R Gam hRdet hRR hGdet D, Matrix.det_mul, Matrix.det_mul]
  exact (hRdet.mul hS).mul hRdet

/-- The inverse of the whitened shift: `(D̃ + I)⁻¹ = R⁻¹ (D + Γ)⁻¹ R⁻¹`. -/
theorem whiten_shift_inv (R Gam : Matrix p p ℝ) (hRdet : IsUnit R.det)
    (hRR : R * R = Gam⁻¹) (hGdet : IsUnit Gam.det) (D : Matrix p p ℝ) :
    (whiten R D + 1)⁻¹ = R⁻¹ * (D + Gam)⁻¹ * R⁻¹ := by
  rw [whiten_add_one R Gam hRdet hRR hGdet D, Matrix.mul_inv_rev, Matrix.mul_inv_rev]
  simp only [Matrix.mul_assoc]

/-- **The gain conjugates**: `K̃ = D̃ (D̃ + I)⁻¹ = R K R⁻¹` — the whitened gain is the similarity
image of the raw gain. -/
theorem K_whiten (R Gam : Matrix p p ℝ) (hRdet : IsUnit R.det)
    (hRR : R * R = Gam⁻¹) (hGdet : IsUnit Gam.det) (D : Matrix p p ℝ) :
    Kmat (whiten R D) 1 = R * Kmat D Gam * R⁻¹ := by
  rw [Kmat, whiten_shift_inv R Gam hRdet hRR hGdet D]
  calc whiten R D * (R⁻¹ * (D + Gam)⁻¹ * R⁻¹)
      = R * D * (R * (R⁻¹ * ((D + Gam)⁻¹ * R⁻¹))) := by
        simp only [whiten, Matrix.mul_assoc]
    _ = R * D * ((D + Gam)⁻¹ * R⁻¹) := by
        rw [Matrix.mul_nonsing_inv_cancel_left _ _ hRdet]
    _ = R * Kmat D Gam * R⁻¹ := by simp only [Kmat, Matrix.mul_assoc]

private lemma mul_nonsing_inv_comm (X Y : Matrix p p ℝ) (h : X * Y = Y * X)
    (hY : IsUnit Y.det) : X * Y⁻¹ = Y⁻¹ * X := by
  calc X * Y⁻¹ = Y⁻¹ * (Y * (X * Y⁻¹)) :=
        (Matrix.nonsing_inv_mul_cancel_left Y (X * Y⁻¹) hY).symm
    _ = Y⁻¹ * (Y * X * Y⁻¹) := by rw [← Matrix.mul_assoc Y X Y⁻¹]
    _ = Y⁻¹ * (X * Y * Y⁻¹) := by rw [h]
    _ = Y⁻¹ * X := by rw [Matrix.mul_nonsing_inv_cancel_right _ _ hY]

/-- A symmetric `D` has a symmetric identity-noise gain: `(D (D+I)⁻¹)ᵀ = D (D+I)⁻¹`, since `D`
commutes with `D + I`. -/
theorem K_symm_of_symm (D : Matrix p p ℝ) (hD : Dᵀ = D) (hD1 : IsUnit (D + 1).det) :
    (Kmat D 1)ᵀ = Kmat D 1 := by
  have hcomm : D * (D + 1) = (D + 1) * D := by
    rw [Matrix.mul_add, Matrix.add_mul, Matrix.mul_one, Matrix.one_mul]
  have hswap : D * (D + 1)⁻¹ = (D + 1)⁻¹ * D :=
    mul_nonsing_inv_comm D (D + 1) hcomm hD1
  rw [Kmat, Matrix.transpose_mul, Matrix.transpose_nonsing_inv, Matrix.transpose_add,
    Matrix.transpose_one, hD, hswap]

/-- Whitening preserves symmetry. -/
theorem whiten_symm (R : Matrix p p ℝ) (hRsym : Rᵀ = R) (D : Matrix p p ℝ) (hD : Dᵀ = D) :
    (whiten R D)ᵀ = whiten R D := by
  simp only [whiten, Matrix.transpose_mul, hRsym, hD, Matrix.mul_assoc]

/-- **The paper's "`K̃_t` symmetric".** -/
theorem K_whiten_symm (R Gam : Matrix p p ℝ) (hRsym : Rᵀ = R) (hRdet : IsUnit R.det)
    (hRR : R * R = Gam⁻¹) (hGdet : IsUnit Gam.det) (D : Matrix p p ℝ) (hD : Dᵀ = D)
    (hS : IsUnit (D + Gam).det) :
    (Kmat (whiten R D) 1)ᵀ = Kmat (whiten R D) 1 :=
  K_symm_of_symm _ (whiten_symm R hRsym D hD)
    (isUnit_whiten_shift_det R Gam hRdet hRR hGdet D hS)

private lemma conj_split₂ (R : Matrix p p ℝ) (hRdet : IsUnit R.det) (X Y : Matrix p p ℝ) :
    R * (X * Y) * R = R * X * R⁻¹ * (R * Y * R) := by
  calc R * (X * Y) * R
      = R * X * (Y * R) := by simp only [Matrix.mul_assoc]
    _ = R * X * (R⁻¹ * (R * (Y * R))) := by
        rw [Matrix.nonsing_inv_mul_cancel_left R (Y * R) hRdet]
    _ = R * X * R⁻¹ * (R * Y * R) := by simp only [Matrix.mul_assoc]

private lemma conj_split₃ (R : Matrix p p ℝ) (hRdet : IsUnit R.det) (X Y Z : Matrix p p ℝ) :
    R * (X * Y * Z) * R = R * X * R⁻¹ * (R * Y * R) * (R⁻¹ * Z * R) := by
  calc R * (X * Y * Z) * R
      = R * X * (Y * (Z * R)) := by simp only [Matrix.mul_assoc]
    _ = R * X * (R⁻¹ * (R * (Y * (R * (R⁻¹ * (Z * R)))))) := by
        rw [Matrix.mul_nonsing_inv_cancel_left R (Z * R) hRdet,
          Matrix.nonsing_inv_mul_cancel_left R (Y * (Z * R)) hRdet]
    _ = R * X * R⁻¹ * (R * Y * R) * (R⁻¹ * Z * R) := by simp only [Matrix.mul_assoc]

/-- **Step 3, the whitening intertwines the recursions.**  Conjugating one round of the
autonomous recursion by `R` (`R R = Γ⁻¹`, symmetric, invertible) gives exactly the whitened
round: `R (K Γ + K S_y Kᵀ) R = K̃ + K̃ S̃ K̃` with `K̃ = D̃(D̃+I)⁻¹` symmetric and
`S̃ = R S_y R`.  This is the reduction that lets `MatrixFloor.floor_step` and the scalar facts of
`RecoverConvergence` apply to the deployed matrix recursion. -/
theorem whiten_intertwines (R Gam : Matrix p p ℝ) (hRsym : Rᵀ = R) (hRdet : IsUnit R.det)
    (hRR : R * R = Gam⁻¹) (hGdet : IsUnit Gam.det) (D Sy : Matrix p p ℝ) (hD : Dᵀ = D)
    (hS : IsUnit (D + Gam).det) :
    whiten R (Kmat D Gam * Gam + Kmat D Gam * Sy * (Kmat D Gam)ᵀ)
      = Kmat (whiten R D) 1
        + Kmat (whiten R D) 1 * whiten R Sy * Kmat (whiten R D) 1 := by
  have hK := K_whiten R Gam hRdet hRR hGdet D
  have hKt : R⁻¹ * (Kmat D Gam)ᵀ * R = (Kmat (whiten R D) 1)ᵀ := by
    rw [hK, Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_nonsing_inv, hRsym]
    simp only [Matrix.mul_assoc]
  have hKsym := K_whiten_symm R Gam hRsym hRdet hRR hGdet D hD hS
  calc whiten R (Kmat D Gam * Gam + Kmat D Gam * Sy * (Kmat D Gam)ᵀ)
      = R * (Kmat D Gam * Gam) * R
        + R * (Kmat D Gam * Sy * (Kmat D Gam)ᵀ) * R := by
        simp only [whiten, Matrix.mul_add, Matrix.add_mul]
    _ = R * Kmat D Gam * R⁻¹ * (R * Gam * R)
        + R * Kmat D Gam * R⁻¹ * (R * Sy * R) * (R⁻¹ * (Kmat D Gam)ᵀ * R) := by
        rw [conj_split₂ R hRdet, conj_split₃ R hRdet]
    _ = Kmat (whiten R D) 1
        + Kmat (whiten R D) 1 * whiten R Sy * Kmat (whiten R D) 1 := by
        rw [R_mul_Gam_mul_R R Gam hRdet hRR hGdet, Matrix.mul_one, ← hK, hKt, hKsym]
        simp only [whiten]

/-- The whitened data covariance: `R (D⋆ + Γ) R = I + R D⋆ R` — the paper's
`S̃ = I + Γ^{-1/2} D⋆ Γ^{-1/2}`. -/
theorem whiten_data_cov (R Gam : Matrix p p ℝ) (hRdet : IsUnit R.det)
    (hRR : R * R = Gam⁻¹) (hGdet : IsUnit Gam.det) (Dstar : Matrix p p ℝ) :
    whiten R (Dstar + Gam) = 1 + whiten R Dstar := by
  simp only [whiten]
  rw [Matrix.mul_add, Matrix.add_mul, R_mul_Gam_mul_R R Gam hRdet hRR hGdet]
  exact add_comm _ _

/-- **Capstone of steps 1–3.**  Under the proposition's hypotheses alone — `Γ ≻ 0`, `Σ ⪰ 0`
(singular admissible), `S_y = A Σ⋆ Aᵀ + Γ` — one full covariance step of the loop, seen in the
whitened data space, is exactly the recursion `D̃' = K̃ + K̃ (I + D̃⋆) K̃` that the checked
scalar/floor modules analyze. -/
theorem whitened_autonomous_step (A : Matrix p n ℝ) (Gam : Matrix p p ℝ)
    (Sig Sigstar : Matrix n n ℝ) (hGam : Gam.PosDef) (hSig : Sig.PosSemidef)
    (R : Matrix p p ℝ) (hRsym : Rᵀ = R) (hRdet : IsUnit R.det) (hRR : R * R = Gam⁻¹) :
    whiten R (Dmat A (nextCov A Sig Gam (Dmat A Sigstar + Gam)))
      = Kmat (whiten R (Dmat A Sig)) 1
        + Kmat (whiten R (Dmat A Sig)) 1 * (1 + whiten R (Dmat A Sigstar))
          * Kmat (whiten R (Dmat A Sig)) 1 := by
  have hGdet : IsUnit Gam.det := (Matrix.isUnit_iff_isUnit_det _).mp hGam.isUnit
  have hSigT : Sigᵀ = Sig := by
    have h := hSig.isHermitian.eq
    rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at h
  have hD : (Dmat A Sig)ᵀ = Dmat A Sig := by
    simp only [Dmat, Matrix.transpose_mul, Matrix.transpose_transpose, hSigT, Matrix.mul_assoc]
  have hDpsd : (Dmat A Sig).PosSemidef :=
    NearNullDefinitional.psd_mul_mul_transpose hSig A
  have hSpd : (Dmat A Sig + Gam).PosDef := Matrix.PosDef.posSemidef_add hDpsd hGam
  have hS : IsUnit (Dmat A Sig + Gam).det := (Matrix.isUnit_iff_isUnit_det _).mp hSpd.isUnit
  rw [data_step_autonomous A Sig Gam _ hS,
    whiten_intertwines R Gam hRsym hRdet hRR hGdet (Dmat A Sig) _ hD hS,
    whiten_data_cov R Gam hRdet hRR hGdet (Dmat A Sigstar)]

/-! ### The concrete whitener `Γ^{-1/2} = CFC.sqrt Γ⁻¹` (the route `DataCovarianceSqrt` built) -/

/-- The standard whitener `Γ^{-1/2}`: the CFC square root of `Γ⁻¹`. -/
noncomputable def stdWhitener (Gam : Matrix p p ℝ) : Matrix p p ℝ := CFC.sqrt Gam⁻¹

/-- The whitener is symmetric (CFC roots are nonnegative, hence self-adjoint; over `ℝ` the
conjugate transpose is the transpose). -/
theorem stdWhitener_transpose (Gam : Matrix p p ℝ) :
    (stdWhitener Gam)ᵀ = stdWhitener Gam := by
  have h : (stdWhitener Gam).IsHermitian :=
    Matrix.isHermitian_iff_isSelfAdjoint.mpr
      (CFC.sqrt_nonneg (Gam⁻¹ : Matrix p p ℝ)).isSelfAdjoint
  have heq := h.eq
  rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at heq

/-- The whitener squares to `Γ⁻¹`. -/
theorem stdWhitener_mul_self (Gam : Matrix p p ℝ) (hGam : Gam.PosDef) :
    stdWhitener Gam * stdWhitener Gam = Gam⁻¹ :=
  CFC.sqrt_mul_sqrt_self (Gam⁻¹ : Matrix p p ℝ) hGam.inv.posSemidef.nonneg

/-- The whitener is invertible: `det(R)² = det(Γ⁻¹)` is a unit. -/
theorem stdWhitener_isUnit_det (Gam : Matrix p p ℝ) (hGam : Gam.PosDef) :
    IsUnit (stdWhitener Gam).det := by
  have h := congrArg Matrix.det (stdWhitener_mul_self Gam hGam)
  rw [Matrix.det_mul] at h
  have hinv : IsUnit (Gam⁻¹).det :=
    Matrix.isUnit_nonsing_inv_det _ ((Matrix.isUnit_iff_isUnit_det _).mp hGam.isUnit)
  rw [← h] at hinv
  exact isUnit_of_mul_isUnit_left hinv

/-- **The capstone at the constructed whitener** — hypotheses ONLY `Γ ≻ 0` and `Σ ⪰ 0`: the
whitening reduction of `p:recover` (ii), fully discharged. -/
theorem whitened_autonomous_step_std (A : Matrix p n ℝ) (Gam : Matrix p p ℝ)
    (Sig Sigstar : Matrix n n ℝ) (hGam : Gam.PosDef) (hSig : Sig.PosSemidef) :
    whiten (stdWhitener Gam) (Dmat A (nextCov A Sig Gam (Dmat A Sigstar + Gam)))
      = Kmat (whiten (stdWhitener Gam) (Dmat A Sig)) 1
        + Kmat (whiten (stdWhitener Gam) (Dmat A Sig)) 1
          * (1 + whiten (stdWhitener Gam) (Dmat A Sigstar))
          * Kmat (whiten (stdWhitener Gam) (Dmat A Sig)) 1 :=
  whitened_autonomous_step A Gam Sig Sigstar hGam hSig (stdWhitener Gam)
    (stdWhitener_transpose Gam) (stdWhitener_isUnit_det Gam hGam)
    (stdWhitener_mul_self Gam hGam)

/-! ### The whitened iterate in the shapes the checked modules consume -/

/-- The whitened round's quadratic form splits as the sum-plus-congruence that
`MatrixFloor.LowerBound_add` and `MatrixFloor.LowerBound_congruence` bound:
`⟪v, (K̃ + K̃ S̃ K̃) v⟫ = ⟪v, K̃ v⟫ + ⟪K̃ v, S̃ (K̃ v)⟫` for symmetric `K̃`. -/
theorem whitened_step_quadratic_form (Kt S : Matrix p p ℝ) (hK : Ktᵀ = Kt) (v : p → ℝ) :
    v ⬝ᵥ ((Kt + Kt * S * Kt) *ᵥ v)
      = v ⬝ᵥ (Kt *ᵥ v) + (Kt *ᵥ v) ⬝ᵥ (S *ᵥ (Kt *ᵥ v)) := by
  have hvm : v ᵥ* Kt = Kt *ᵥ v := by
    conv_lhs => rw [← hK]
    rw [Matrix.vecMul_transpose]
  rw [Matrix.add_mulVec, dotProduct_add]
  have h2 : v ⬝ᵥ ((Kt * S * Kt) *ᵥ v) = (Kt *ᵥ v) ⬝ᵥ (S *ᵥ (Kt *ᵥ v)) := by
    rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, dotProduct_mulVec, hvm]
  rw [h2]

/-- The gain's defining witness in the exact shape of `MatrixFloor.LowerBound_gain`'s `hinv`
input: for every `v` there is `w` with `D w + w = v` and `K v = v − w` (namely
`w = (D+I)⁻¹ v`). -/
theorem exists_gain_witness (D : Matrix p p ℝ) (hD1 : IsUnit (D + 1).det) (v : p → ℝ) :
    ∃ w : p → ℝ, D *ᵥ w + w = v ∧ Kmat D 1 *ᵥ v = v - w := by
  refine ⟨(D + 1)⁻¹ *ᵥ v, ?_, ?_⟩
  · calc D *ᵥ ((D + 1)⁻¹ *ᵥ v) + (D + 1)⁻¹ *ᵥ v
        = (D + 1) *ᵥ ((D + 1)⁻¹ *ᵥ v) := by
          rw [Matrix.add_mulVec, Matrix.one_mulVec]
      _ = v := by
          rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hD1, Matrix.one_mulVec]
  · rw [← one_sub_K D hD1, Matrix.sub_mulVec, Matrix.one_mulVec]
    abel

/-- **The interior fixed point is unique** — the paper's `E = S̃` cancellation: any invertible
fixed point of the whitened recursion `D ↦ K + K S̃ K`, `K = D(D+I)⁻¹`, satisfies
`D + I = S̃` (so with `S̃ = I + D̃⋆` it is `D̃⋆`). -/
theorem fixed_point_unique (S Dt : Matrix p p ℝ) (hDt : IsUnit Dt.det)
    (hD1 : IsUnit (Dt + 1).det)
    (hfix : Dt = Kmat Dt 1 + Kmat Dt 1 * S * Kmat Dt 1) :
    Dt + 1 = S := by
  have e1 : Kmat Dt 1 * (Dt + 1) = Dt := by
    simp only [Kmat]
    exact Matrix.nonsing_inv_mul_cancel_right _ _ hD1
  have e3 : Dt = Kmat Dt 1 * Dt + Kmat Dt 1 := by
    have hdist : Kmat Dt 1 * (Dt + 1) = Kmat Dt 1 * Dt + Kmat Dt 1 := by
      rw [Matrix.mul_add, Matrix.mul_one]
    rw [← hdist, e1]
  have h1 : Dt - Kmat Dt 1 = Kmat Dt 1 * Dt := sub_eq_iff_eq_add.mpr e3
  have hKS : Dt - Kmat Dt 1 = Kmat Dt 1 * S * Kmat Dt 1 := sub_eq_iff_eq_add'.mpr hfix
  have hKD : Kmat Dt 1 * Dt = Kmat Dt 1 * (S * Kmat Dt 1) := by
    rw [← h1, hKS, Matrix.mul_assoc]
  have hKdet : IsUnit (Kmat Dt 1).det := by
    have hright : Kmat Dt 1 * ((Dt + 1) * Dt⁻¹) = 1 := by
      rw [← Matrix.mul_assoc, e1, Matrix.mul_nonsing_inv _ hDt]
    exact Matrix.isUnit_det_of_right_inverse hright
  have hSK : Dt = S * Kmat Dt 1 :=
    calc Dt = (Kmat Dt 1)⁻¹ * (Kmat Dt 1 * Dt) :=
          (Matrix.nonsing_inv_mul_cancel_left _ _ hKdet).symm
      _ = (Kmat Dt 1)⁻¹ * (Kmat Dt 1 * (S * Kmat Dt 1)) := by rw [hKD]
      _ = S * Kmat Dt 1 := Matrix.nonsing_inv_mul_cancel_left _ _ hKdet
  have hDD : Dt * (Dt + 1) = S * Dt := by
    nth_rewrite 1 [hSK]
    rw [Matrix.mul_assoc, e1]
  have hcomm : (Dt + 1) * Dt = S * Dt := by
    rw [Matrix.add_mul, Matrix.one_mul, ← hDD, Matrix.mul_add, Matrix.mul_one]
  calc Dt + 1 = (Dt + 1) * Dt * Dt⁻¹ := (Matrix.mul_nonsing_inv_cancel_right _ _ hDt).symm
    _ = S * Dt * Dt⁻¹ := by rw [hcomm]
    _ = S := Matrix.mul_nonsing_inv_cancel_right _ _ hDt

/-! ## Step 4: the pseudoinverse lift to the resolved block -/

/-- The Moore–Penrose pseudoinverse in the full-row-rank case, `A⁺ = Aᵀ (A Aᵀ)⁻¹` — faithful to
the proposition's hypothesis: full row rank of a real `A` is exactly invertibility of `A Aᵀ`. -/
noncomputable def pinv (A : Matrix p n ℝ) : Matrix n p ℝ := Aᵀ * (A * Aᵀ)⁻¹

/-- The resolved projector `P_R = A⁺ A`. -/
noncomputable def projR (A : Matrix p n ℝ) : Matrix n n ℝ := pinv A * A

/-- `A A⁺ = I` in the full-row-rank case. -/
theorem pinv_right_inverse (A : Matrix p n ℝ) (hA : IsUnit (A * Aᵀ).det) :
    A * pinv A = 1 := by
  simp only [pinv, ← Matrix.mul_assoc]
  exact Matrix.mul_nonsing_inv _ hA

/-- `P_R` is symmetric (no invertibility needed). -/
theorem projR_transpose (A : Matrix p n ℝ) : (projR A)ᵀ = projR A := by
  have hsym : ((A * Aᵀ)⁻¹)ᵀ = (A * Aᵀ)⁻¹ := by
    rw [Matrix.transpose_nonsing_inv, Matrix.transpose_mul, Matrix.transpose_transpose]
  simp only [projR, pinv, Matrix.transpose_mul, Matrix.transpose_transpose, hsym,
    Matrix.mul_assoc]

/-- `P_R` is idempotent. -/
theorem projR_idem (A : Matrix p n ℝ) (hA : IsUnit (A * Aᵀ).det) :
    projR A * projR A = projR A := by
  have h : projR A * projR A = pinv A * (A * pinv A) * A := by
    simp only [projR, Matrix.mul_assoc]
  rw [h, pinv_right_inverse A hA, Matrix.mul_one, projR]

/-- `A P_R = A`: the projector fixes the operator. -/
theorem A_mul_projR (A : Matrix p n ℝ) (hA : IsUnit (A * Aᵀ).det) :
    A * projR A = A := by
  have h : A * projR A = A * pinv A * A := by
    simp only [projR, Matrix.mul_assoc]
  rw [h, pinv_right_inverse A hA, Matrix.one_mul]

/-- `P_R Aᵀ = Aᵀ`: the projector fixes the row space. -/
theorem projR_mul_transpose (A : Matrix p n ℝ) (hA : IsUnit (A * Aᵀ).det) :
    projR A * Aᵀ = Aᵀ := by
  have h : projR A * Aᵀ = Aᵀ * ((A * Aᵀ)⁻¹ * (A * Aᵀ)) := by
    simp only [projR, pinv, Matrix.mul_assoc]
  rw [h, Matrix.nonsing_inv_mul _ hA, Matrix.mul_one]

/-- `P_R` annihilates the kernel: `A v = 0 → P_R v = 0`. -/
theorem projR_mulVec_kernel (A : Matrix p n ℝ) {v : n → ℝ} (hv : A *ᵥ v = 0) :
    projR A *ᵥ v = 0 := by
  rw [projR, ← Matrix.mulVec_mulVec, hv, Matrix.mulVec_zero]

/-- **The pseudoinverse lift of the covariance** — the paper's
`P_R Σ P_R = A⁺ D (A⁺)ᵀ` for `D = A Σ Aᵀ`: pure algebra, no invertibility hypothesis. -/
theorem resolved_block (A : Matrix p n ℝ) (Sig : Matrix n n ℝ) :
    projR A * Sig * projR A = pinv A * Dmat A Sig * (pinv A)ᵀ := by
  have hT : (pinv A)ᵀ = (A * Aᵀ)⁻¹ * A := by
    rw [pinv, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_nonsing_inv,
      Matrix.transpose_mul, Matrix.transpose_transpose]
  rw [hT]
  simp only [projR, pinv, Dmat, Matrix.mul_assoc]

/-- The mean's lift: `P_R μ = A⁺ (A μ)`. -/
theorem projR_mulVec (A : Matrix p n ℝ) (mu : n → ℝ) :
    projR A *ᵥ mu = pinv A *ᵥ (A *ᵥ mu) := by
  rw [projR, ← Matrix.mulVec_mulVec]

/-- Conjugation is continuous, so limits pass through `D ↦ B D Bᵀ`. -/
theorem tendsto_conj {B : Matrix n p ℝ} {Dseq : ℕ → Matrix p p ℝ} {Dlim : Matrix p p ℝ}
    (h : Tendsto Dseq atTop (𝓝 Dlim)) :
    Tendsto (fun t => B * Dseq t * Bᵀ) atTop (𝓝 (B * Dlim * Bᵀ)) := by
  have hc : Continuous fun M : Matrix p p ℝ => B * M * Bᵀ := by fun_prop
  exact (hc.tendsto Dlim).comp h

/-- **Step 4, the covariance lift.**  If the data-space covariance converges,
`D_t = A Σ_t Aᵀ → D⋆ = A Σ⋆ Aᵀ` — which is what the checked recursion results deliver after
whitening — then the resolved block converges: `P_R Σ_t P_R → P_R Σ⋆ P_R`. -/
theorem tendsto_resolved_covariance (A : Matrix p n ℝ)
    (Sigs : ℕ → Matrix n n ℝ) (Sigstar : Matrix n n ℝ)
    (hD : Tendsto (fun t => Dmat A (Sigs t)) atTop (𝓝 (Dmat A Sigstar))) :
    Tendsto (fun t => projR A * Sigs t * projR A) atTop
      (𝓝 (projR A * Sigstar * projR A)) := by
  simp only [resolved_block]
  exact tendsto_conj hD

/-- **Step 4, the mean lift.**  If `A μ_t → A μ⋆` (the geometric decay checked in
`RecoverConvergence.tendsto_mean_error`), then `P_R μ_t → P_R μ⋆`. -/
theorem tendsto_resolved_mean (A : Matrix p n ℝ) (mus : ℕ → n → ℝ) (mustar : n → ℝ)
    (hm : Tendsto (fun t => A *ᵥ mus t) atTop (𝓝 (A *ᵥ mustar))) :
    Tendsto (fun t => projR A *ᵥ mus t) atTop (𝓝 (projR A *ᵥ mustar)) := by
  simp only [projR_mulVec]
  have hc : Continuous fun w : p → ℝ => pinv A *ᵥ w := by fun_prop
  exact (hc.tendsto (A *ᵥ mustar)).comp hm

end RecoverWhitening

-- #print axioms audit (whitening/congruence reduction for p:recover (ii))
#print axioms RecoverWhitening.woodbury_rearrangement
#print axioms RecoverWhitening.A_mul_gain
#print axioms RecoverWhitening.data_step_autonomous
#print axioms RecoverWhitening.mean_step
#print axioms RecoverWhitening.one_sub_K
#print axioms RecoverWhitening.R_mul_Gam_mul_R
#print axioms RecoverWhitening.whiten_add_one
#print axioms RecoverWhitening.isUnit_whiten_shift_det
#print axioms RecoverWhitening.whiten_shift_inv
#print axioms RecoverWhitening.K_whiten
#print axioms RecoverWhitening.K_symm_of_symm
#print axioms RecoverWhitening.whiten_symm
#print axioms RecoverWhitening.K_whiten_symm
#print axioms RecoverWhitening.whiten_intertwines
#print axioms RecoverWhitening.whiten_data_cov
#print axioms RecoverWhitening.whitened_autonomous_step
#print axioms RecoverWhitening.stdWhitener_transpose
#print axioms RecoverWhitening.stdWhitener_mul_self
#print axioms RecoverWhitening.stdWhitener_isUnit_det
#print axioms RecoverWhitening.whitened_autonomous_step_std
#print axioms RecoverWhitening.whitened_step_quadratic_form
#print axioms RecoverWhitening.exists_gain_witness
#print axioms RecoverWhitening.fixed_point_unique
#print axioms RecoverWhitening.pinv_right_inverse
#print axioms RecoverWhitening.projR_transpose
#print axioms RecoverWhitening.projR_idem
#print axioms RecoverWhitening.A_mul_projR
#print axioms RecoverWhitening.projR_mul_transpose
#print axioms RecoverWhitening.projR_mulVec_kernel
#print axioms RecoverWhitening.resolved_block
#print axioms RecoverWhitening.projR_mulVec
#print axioms RecoverWhitening.tendsto_conj
#print axioms RecoverWhitening.tendsto_resolved_covariance
#print axioms RecoverWhitening.tendsto_resolved_mean
