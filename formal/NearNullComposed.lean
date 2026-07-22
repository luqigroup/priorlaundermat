/-
# `p:nearnull` (i), composed: the near-null freeze bound in one statement

The two halves of `p:nearnull` (i) were machine-checked in separate modules that never met:
`NearNullFactored.precision_diff_factored` reduces `Σ_q⁻¹ − Σ_ρ⁻¹` to the factored form
`Aᵀ B A` with `B = Γ⁻¹ − Γ⁻¹ Q⁻¹ Γ⁻¹` (over `Matrix`, the preamble's definitional relations as
hypotheses), while `NearNullFreeze.precision_move_le` bounds a factored quadratic form — but over
continuous linear *endo*morphisms of one space, a carrier that cannot even hold a rectangular `A`.
The identification of the one's `B` with the other's factored form was made on paper.

This module makes it inside Lean.  The sandwich is restated over matrices — the carrier bridge
through `Matrix.toEuclideanLin` is heavier than the eight-line dot-product identity
`quadratic_form_factored` below, which is the entire content of the sandwich once the bound on `B`
is variational — and composed with the imported factorization into `precision_bound`:

    |⟪v, (Σ_q⁻¹ − Σ_ρ⁻¹) v⟫| ≤ c ⟪A v, A v⟫,

for every `v` (the paper states it for unit `v`; both sides are quadratic, so this is the same
statement), with the kernel endpoint `A v = 0 → the form moves by exactly 0` as
`precision_zero_of_kernel`.  Here `⟪·,·⟫` is the dot product and `⟪w, w⟫ = ∑ wᵢ²` is the squared
Euclidean norm (`dotProduct_self_eq_sum_sq` pins this), so the conclusion is the paper's
`|vᵀ(Σ_q⁻¹ − Σ_ρ⁻¹)v| ≤ c ‖A v‖²` at ~line 640 of the manuscript.

**What `c` is here.**  The constant enters as the honest variational hypothesis
`∀ w, |⟪w, B w⟫| ≤ c ⟪w, w⟫` — for a self-adjoint `B` this says exactly `c ≥ ‖B‖₂`, the paper's
`c = ‖Γ⁻¹ − Γ⁻¹ Q⁻¹ Γ⁻¹‖₂`, but the spectral characterization itself is not re-proved here.
`psd_sub_form_le` discharges that hypothesis the way the appendix does: `Γ⁻¹` and `Γ⁻¹ Q⁻¹ Γ⁻¹`
are each positive semidefinite, so their difference's form is bounded by the larger of their
bounds, giving `c = max a b` (`precision_bound_of_psd`) — the PSD-ness and the bounds `a`, `b`
of the two factors are spectral facts about `Γ`, `S_y` and remain hypotheses, again variational.

The definitional-relation hypotheses are exactly those `NearNullFactored` already takes (each one
line from the preamble's definitions of `G`, `Σ_post`, `Q`, plus the three invertibility
assumptions); nothing new is assumed beyond them and the stated bound on `B`.

No `sorry`/`admit`; `#print axioms` on each result at the end.
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import NearNullFactored

namespace NearNullComposed

open Matrix

variable {n m : Type*} [Fintype n] [Fintype m]

/-- The dot product of a real vector with itself is nonnegative — the `0 ≤ ‖w‖²` the difference
bound needs, in the carrier the composition lives in. -/
theorem dotProduct_self_nonneg (w : m → ℝ) : 0 ≤ w ⬝ᵥ w := by
  unfold dotProduct
  exact Finset.sum_nonneg fun i _ => mul_self_nonneg (w i)

/-- `⟪w, w⟫ = ∑ wᵢ²`: the dot product with itself *is* the squared Euclidean norm, stated
concretely so the main bound reads as the paper's `c ‖A v‖²`. -/
theorem dotProduct_self_eq_sum_sq (w : m → ℝ) : w ⬝ᵥ w = ∑ i, w i ^ 2 := by
  unfold dotProduct
  exact Finset.sum_congr rfl fun i _ => (pow_two (w i)).symm

/-- **The sandwich reads through the transpose, over matrices.**  The quadratic form of `Aᵀ B A`
at `v` is the form of `B` at `A v` — the matrix restatement of `NearNullFreeze.sandwich_factors`,
whose continuous-linear-map carrier cannot hold a rectangular `A`. -/
theorem quadratic_form_factored (A : Matrix m n ℝ) (B : Matrix m m ℝ) (v : n → ℝ) :
    v ⬝ᵥ ((Aᵀ * B * A) *ᵥ v) = (A *ᵥ v) ⬝ᵥ (B *ᵥ (A *ᵥ v)) :=
  calc v ⬝ᵥ ((Aᵀ * B * A) *ᵥ v)
      = v ⬝ᵥ (Aᵀ *ᵥ ((B * A) *ᵥ v)) := by rw [Matrix.mul_assoc, ← Matrix.mulVec_mulVec]
    _ = (v ᵥ* Aᵀ) ⬝ᵥ ((B * A) *ᵥ v) := dotProduct_mulVec _ _ _
    _ = (A *ᵥ v) ⬝ᵥ ((B * A) *ᵥ v) := by rw [Matrix.vecMul_transpose]
    _ = (A *ᵥ v) ⬝ᵥ (B *ᵥ (A *ᵥ v)) := by rw [← Matrix.mulVec_mulVec]

/-- **The constant, in the composition's carrier.**  If `X` and `Y` are positive semidefinite with
`X ⪯ a I` and `Y ⪯ b I` in the variational sense, the difference's quadratic form is bounded by
`max a b` — the appendix's "each positive semidefinite, so their difference has spectral norm at
most the larger of theirs", restated from `NearNullFactored.bound_of_psd_sub` over matrices so it
can feed `precision_bound` directly. -/
theorem psd_sub_form_le {a b : ℝ} {X Y : Matrix m m ℝ}
    (hX0 : ∀ w : m → ℝ, 0 ≤ w ⬝ᵥ (X *ᵥ w)) (hXa : ∀ w : m → ℝ, w ⬝ᵥ (X *ᵥ w) ≤ a * (w ⬝ᵥ w))
    (hY0 : ∀ w : m → ℝ, 0 ≤ w ⬝ᵥ (Y *ᵥ w)) (hYb : ∀ w : m → ℝ, w ⬝ᵥ (Y *ᵥ w) ≤ b * (w ⬝ᵥ w))
    (w : m → ℝ) :
    |w ⬝ᵥ ((X - Y) *ᵥ w)| ≤ max a b * (w ⬝ᵥ w) := by
  have hsub : w ⬝ᵥ ((X - Y) *ᵥ w) = w ⬝ᵥ (X *ᵥ w) - w ⬝ᵥ (Y *ᵥ w) := by
    rw [Matrix.sub_mulVec, dotProduct_sub]
  have hnn : (0:ℝ) ≤ w ⬝ᵥ w := dotProduct_self_nonneg w
  have ha : a * (w ⬝ᵥ w) ≤ max a b * (w ⬝ᵥ w) :=
    mul_le_mul_of_nonneg_right (le_max_left a b) hnn
  have hb : b * (w ⬝ᵥ w) ≤ max a b * (w ⬝ᵥ w) :=
    mul_le_mul_of_nonneg_right (le_max_right a b) hnn
  rw [hsub, abs_le]
  constructor
  · linarith [hX0 w, hYb w]
  · linarith [hY0 w, hXa w]

/-- **`p:nearnull` (i), composed.**  Under the preamble's definitional relations (exactly the
hypotheses `NearNullFactored.precision_diff_factored` takes) and a variational bound `c` on the
middle factor `B = Γ⁻¹ − Γ⁻¹ Q⁻¹ Γ⁻¹` — for self-adjoint `B` this is `c ≥ ‖B‖₂`, the paper's
constant — the one step moves the prior's precision along any `v` by at most `c ⟪A v, A v⟫`, i.e.
`c ‖A v‖²`:  what `eq:nearnull-freeze` and the sandwich said separately, now one Lean theorem. -/
theorem precision_bound [DecidableEq n] [DecidableEq m]
    (A : Matrix m n ℝ) (Sig_post Sig_rho Sq : Matrix n n ℝ) (G : Matrix n m ℝ)
    (Gam Sy Sdata Q : Matrix m m ℝ) {c : ℝ}
    (hSq : Sq = Sig_post + G * Sy * Gᵀ)
    (hG : Sig_post⁻¹ * G = Aᵀ * Gam⁻¹)
    (hGT : Gᵀ * Sig_post⁻¹ = Gam⁻¹ * A)
    (hGG : Gᵀ * Sig_post⁻¹ * G = Gam⁻¹ - Sdata⁻¹)
    (hpost : Sig_post⁻¹ = Sig_rho⁻¹ + Aᵀ * Gam⁻¹ * A)
    (hQ : Q = Sy⁻¹ + (Gam⁻¹ - Sdata⁻¹))
    (hpu : IsUnit Sig_post) (hsu : IsUnit Sy) (hqu : IsUnit (Sy⁻¹ + Gᵀ * Sig_post⁻¹ * G))
    (hB : ∀ w : m → ℝ,
      |w ⬝ᵥ ((Gam⁻¹ - Gam⁻¹ * Q⁻¹ * Gam⁻¹) *ᵥ w)| ≤ c * (w ⬝ᵥ w))
    (v : n → ℝ) :
    |v ⬝ᵥ ((Sq⁻¹ - Sig_rho⁻¹) *ᵥ v)| ≤ c * ((A *ᵥ v) ⬝ᵥ (A *ᵥ v)) := by
  have hfact := NearNullFactored.precision_diff_factored A Sig_post Sig_rho Sq G Gam Sy Sdata Q
    hSq hG hGT hGG hpost hQ hpu hsu hqu
  rw [hfact, quadratic_form_factored]
  exact hB (A *ᵥ v)

/-- **The exact freeze is the endpoint, composed.**  On a genuine kernel direction (`A v = 0`) the
quadratic form of `Σ_q⁻¹ − Σ_ρ⁻¹` moves by exactly `0` — `eq:freeze` recovered from the factored
form, for the precision difference itself rather than an abstract `Aᵀ B A`. -/
theorem precision_zero_of_kernel [DecidableEq n] [DecidableEq m]
    (A : Matrix m n ℝ) (Sig_post Sig_rho Sq : Matrix n n ℝ) (G : Matrix n m ℝ)
    (Gam Sy Sdata Q : Matrix m m ℝ)
    (hSq : Sq = Sig_post + G * Sy * Gᵀ)
    (hG : Sig_post⁻¹ * G = Aᵀ * Gam⁻¹)
    (hGT : Gᵀ * Sig_post⁻¹ = Gam⁻¹ * A)
    (hGG : Gᵀ * Sig_post⁻¹ * G = Gam⁻¹ - Sdata⁻¹)
    (hpost : Sig_post⁻¹ = Sig_rho⁻¹ + Aᵀ * Gam⁻¹ * A)
    (hQ : Q = Sy⁻¹ + (Gam⁻¹ - Sdata⁻¹))
    (hpu : IsUnit Sig_post) (hsu : IsUnit Sy) (hqu : IsUnit (Sy⁻¹ + Gᵀ * Sig_post⁻¹ * G))
    {v : n → ℝ} (hv : A *ᵥ v = 0) :
    v ⬝ᵥ ((Sq⁻¹ - Sig_rho⁻¹) *ᵥ v) = 0 := by
  have hfact := NearNullFactored.precision_diff_factored A Sig_post Sig_rho Sq G Gam Sy Sdata Q
    hSq hG hGT hGG hpost hQ hpu hsu hqu
  rw [hfact, quadratic_form_factored, hv]
  simp

/-- **The paper's `c`, supplied by the PSD structure.**  With `Γ⁻¹ ⪯ a I` and
`Γ⁻¹ Q⁻¹ Γ⁻¹ ⪯ b I` both positive semidefinite (variationally — these are the appendix's spectral
facts, hypotheses here), the bound holds with `c = max a b`; instantiating
`a = λ_min(Γ)⁻¹`, `b = λ_max(S_y)/λ_min(Γ)²` (the appendix's `Q ⪰ S_y⁻¹` step) gives the stated
`c ≤ λ_min(Γ)⁻¹ max{1, λ_max(S_y)/λ_min(Γ)}`. -/
theorem precision_bound_of_psd [DecidableEq n] [DecidableEq m]
    (A : Matrix m n ℝ) (Sig_post Sig_rho Sq : Matrix n n ℝ) (G : Matrix n m ℝ)
    (Gam Sy Sdata Q : Matrix m m ℝ) {a b : ℝ}
    (hSq : Sq = Sig_post + G * Sy * Gᵀ)
    (hG : Sig_post⁻¹ * G = Aᵀ * Gam⁻¹)
    (hGT : Gᵀ * Sig_post⁻¹ = Gam⁻¹ * A)
    (hGG : Gᵀ * Sig_post⁻¹ * G = Gam⁻¹ - Sdata⁻¹)
    (hpost : Sig_post⁻¹ = Sig_rho⁻¹ + Aᵀ * Gam⁻¹ * A)
    (hQ : Q = Sy⁻¹ + (Gam⁻¹ - Sdata⁻¹))
    (hpu : IsUnit Sig_post) (hsu : IsUnit Sy) (hqu : IsUnit (Sy⁻¹ + Gᵀ * Sig_post⁻¹ * G))
    (hX0 : ∀ w : m → ℝ, 0 ≤ w ⬝ᵥ (Gam⁻¹ *ᵥ w))
    (hXa : ∀ w : m → ℝ, w ⬝ᵥ (Gam⁻¹ *ᵥ w) ≤ a * (w ⬝ᵥ w))
    (hY0 : ∀ w : m → ℝ, 0 ≤ w ⬝ᵥ ((Gam⁻¹ * Q⁻¹ * Gam⁻¹) *ᵥ w))
    (hYb : ∀ w : m → ℝ, w ⬝ᵥ ((Gam⁻¹ * Q⁻¹ * Gam⁻¹) *ᵥ w) ≤ b * (w ⬝ᵥ w))
    (v : n → ℝ) :
    |v ⬝ᵥ ((Sq⁻¹ - Sig_rho⁻¹) *ᵥ v)| ≤ max a b * ((A *ᵥ v) ⬝ᵥ (A *ᵥ v)) :=
  precision_bound A Sig_post Sig_rho Sq G Gam Sy Sdata Q
    hSq hG hGT hGG hpost hQ hpu hsu hqu
    (fun w => psd_sub_form_le hX0 hXa hY0 hYb w) v

end NearNullComposed

-- #print axioms audit (p:nearnull (i), composed)
#print axioms NearNullComposed.dotProduct_self_nonneg
#print axioms NearNullComposed.dotProduct_self_eq_sum_sq
#print axioms NearNullComposed.quadratic_form_factored
#print axioms NearNullComposed.psd_sub_form_le
#print axioms NearNullComposed.precision_bound
#print axioms NearNullComposed.precision_zero_of_kernel
#print axioms NearNullComposed.precision_bound_of_psd
