/-
# `eq:nearnull-freeze`: the precision difference is a factored form

The last step of `p:nearnull` (i).  `NearNullFreeze.lean` shows that a precision difference of the
form `Aᵀ B A` is moved by at most `‖B‖ ‖A v‖²` along `v` and by exactly nothing on `ker A`.  What was
still done on paper is the rearrangement that puts `Σ_q⁻¹ − Σ_ρ⁻¹` into that form and names `B`.

The appendix's chain is: the curated covariance is `Σ_q = Σ_post + G S_y Gᵀ`; Woodbury inverts it;
`Σ_post⁻¹ G = Aᵀ Γ⁻¹` and `Gᵀ Σ_post⁻¹ G = Γ⁻¹ − S_ρ⁻¹` come straight from the definitions; and
`Σ_post⁻¹ = Σ_ρ⁻¹ + Aᵀ Γ⁻¹ A`.  Substituting gives

    Σ_q⁻¹ − Σ_ρ⁻¹ = Aᵀ (Γ⁻¹ − Γ⁻¹ Q⁻¹ Γ⁻¹) A,    Q = S_y⁻¹ + Γ⁻¹ − S_ρ⁻¹,

which is `eq:nearnull-freeze`, and hence `B = Γ⁻¹ − Γ⁻¹ Q⁻¹ Γ⁻¹`.

`precision_diff_factored` below is that derivation, over `Matrix`, on mathlib's Woodbury identity
(`Matrix.add_mul_mul_inv_eq_sub`).  The definitional relations enter as hypotheses, which is what
they are: each is one line from the definitions of `G`, `Σ_post` and `Q`, and `hGT` is `hG`
transposed under symmetry of `Γ` and `Σ_post`.

`bound_of_psd_sub` closes the constant: the appendix argues that `Γ⁻¹` and `Γ⁻¹ Q⁻¹ Γ⁻¹` are each
positive semidefinite, so their difference has spectral norm at most the larger of theirs.  That is
proved here variationally, in the same style as the rest of the floor argument.

No `sorry`/`admit`; `#print axioms` on each result at the end.
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Analysis.InnerProductSpace.Basic

namespace NearNullFactored

open Matrix

variable {n m : Type*} [Fintype n] [DecidableEq n] [Fintype m] [DecidableEq m]

/-- **`eq:nearnull-freeze`.**  With the curated covariance `Σ_q = Σ_post + G S_y Gᵀ` and the
definitional relations of the preamble, the one step moves the precision by a form factored through
the operator, `Aᵀ (Γ⁻¹ − Γ⁻¹ Q⁻¹ Γ⁻¹) A`.  This names the `B` that `NearNullFreeze.precision_move_le`
bounds, so `p:nearnull` (i) is complete. -/
theorem precision_diff_factored
    (A : Matrix m n ℝ) (Sig_post Sig_rho Sq : Matrix n n ℝ) (G : Matrix n m ℝ)
    (Gam Sy Sdata Q : Matrix m m ℝ)
    (hSq : Sq = Sig_post + G * Sy * Gᵀ)
    (hG : Sig_post⁻¹ * G = Aᵀ * Gam⁻¹)
    (hGT : Gᵀ * Sig_post⁻¹ = Gam⁻¹ * A)
    (hGG : Gᵀ * Sig_post⁻¹ * G = Gam⁻¹ - Sdata⁻¹)
    (hpost : Sig_post⁻¹ = Sig_rho⁻¹ + Aᵀ * Gam⁻¹ * A)
    (hQ : Q = Sy⁻¹ + (Gam⁻¹ - Sdata⁻¹))
    (hpu : IsUnit Sig_post) (hsu : IsUnit Sy) (hqu : IsUnit (Sy⁻¹ + Gᵀ * Sig_post⁻¹ * G)) :
    Sq⁻¹ - Sig_rho⁻¹ = Aᵀ * (Gam⁻¹ - Gam⁻¹ * Q⁻¹ * Gam⁻¹) * A := by
  have hwood : Sq⁻¹
      = Sig_post⁻¹ - Sig_post⁻¹ * G * (Sy⁻¹ + Gᵀ * Sig_post⁻¹ * G)⁻¹ * Gᵀ * Sig_post⁻¹ := by
    rw [hSq]; exact Matrix.add_mul_mul_inv_eq_sub Sig_post G Sy Gᵀ hpu hsu hqu
  have hmid : (Sy⁻¹ + Gᵀ * Sig_post⁻¹ * G)⁻¹ = Q⁻¹ := by rw [hGG, ← hQ]
  rw [hwood, hmid, hG]
  simp only [Matrix.mul_assoc]
  rw [hGT, hpost]
  simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_assoc]
  abel

/-- **The constant.**  If `X` and `Y` are positive semidefinite with `X ⪯ a I` and `Y ⪯ b I` in the
variational sense, then their difference's quadratic form is bounded by `max a b`.  This is the
appendix's step "each positive semidefinite, so their difference has spectral norm at most the larger
of theirs", giving `c ≤ λ_min(Γ)⁻¹ max{1, λ_max(S_y)/λ_min(Γ)}` once `Q ⪰ S_y⁻¹` supplies `b`. -/
theorem bound_of_psd_sub {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {a b : ℝ} {X Y : E →L[ℝ] E}
    (hX0 : ∀ v : E, 0 ≤ @inner ℝ _ _ v (X v)) (hXa : ∀ v : E, @inner ℝ _ _ v (X v) ≤ a * ‖v‖ ^ 2)
    (hY0 : ∀ v : E, 0 ≤ @inner ℝ _ _ v (Y v)) (hYb : ∀ v : E, @inner ℝ _ _ v (Y v) ≤ b * ‖v‖ ^ 2)
    (v : E) :
    |@inner ℝ _ _ v ((X - Y) v)| ≤ max a b * ‖v‖ ^ 2 := by
  have hsub : @inner ℝ _ _ v ((X - Y) v)
      = @inner ℝ _ _ v (X v) - @inner ℝ _ _ v (Y v) := by
    simp [ContinuousLinearMap.sub_apply, inner_sub_right]
  have hnn : (0:ℝ) ≤ ‖v‖ ^ 2 := by positivity
  have ha : a * ‖v‖ ^ 2 ≤ max a b * ‖v‖ ^ 2 :=
    mul_le_mul_of_nonneg_right (le_max_left a b) hnn
  have hb : b * ‖v‖ ^ 2 ≤ max a b * ‖v‖ ^ 2 :=
    mul_le_mul_of_nonneg_right (le_max_right a b) hnn
  rw [hsub, abs_le]
  constructor
  · linarith [hX0 v, hYb v]
  · linarith [hY0 v, hXa v]

end NearNullFactored

-- #print axioms audit (eq:nearnull-freeze and the constant)
#print axioms NearNullFactored.precision_diff_factored
#print axioms NearNullFactored.bound_of_psd_sub
