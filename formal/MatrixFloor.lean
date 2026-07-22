/-
# The matrix reduction behind `p:recover` (ii)'s interior floor

`RecoverConvergence.lean` proves the scalar recursion's behaviour.  The appendix reaches the matrix
case not by diagonalizing — the iterate and `S̃` do not commute, so the eigenvectors rotate every
round — but by pushing a *lower bound* through the recursion:

    D̃ₜ₊₁ = K̃ₜ + K̃ₜ S̃ K̃ₜ,    K̃ₜ = D̃ₜ (D̃ₜ + I)⁻¹  self-adjoint,

    λₜ₊₁ ≥ λ_min(K̃ₜ) + s_min · λ_min(K̃ₜ²) = g(λₜ),
    g(λ) = λ/(1+λ) + s_min λ²/(1+λ)²,   g(λ) − λ = λ²(s_min − 1 − λ)/(1+λ)².

Two operator facts carry that step, and both are formalized here **without** mathlib's eigenvalue
API, using instead the variational characterization `m ≤ λ_min(M) ↔ ∀ v, m‖v‖² ≤ ⟪v, M v⟫`:

* `LowerBound_add`        — lower bounds add across a sum (the Weyl inequality the appendix cites);
* `LowerBound_congruence` — if `s` bounds `S` below and `K` is self-adjoint, then `K S K` is bounded
                            below by `s K²`, i.e. `s‖K v‖²` (the congruence step).

Composing them gives `floor_step`: one round's lower bound is at least `g` of the previous one.  The
scalar half — that `g` fixes `s_min − 1`, is increasing, and satisfies `g(λ) ≥ λ` below the fixed
point, hence that the floor `min(λ₀, s_min − 1)` persists — is already machine-checked in
`RecoverConvergence.lean`, since `g` is *literally* that file's `Ψ` with `d⋆ = s_min − 1`:
`Recover.psi_sub_self_sign` is the identity displayed above.  `floor_persists` below makes that
reuse explicit.

* `LowerBound_gain`       — the spectral-mapping step `λ_min(K̃ₜ) ≥ f(λₜ)` for `f(λ) = λ/(1+λ)`,
                            also variational: no functional calculus is needed, only
                            self-adjointness and Cauchy–Schwarz applied to `w = (D+I)⁻¹ v`.

No `sorry`/`admit`; `#print axioms` on each result at the end.
-/
import Mathlib.Analysis.InnerProductSpace.Adjoint
import RecoverConvergence

namespace MatrixFloor

open RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `m` is a variational lower bound for `M`: `m ‖v‖² ≤ ⟪v, M v⟫` for every `v`.  For a self-adjoint
operator this says exactly `m ≤ λ_min(M)`, and it is the only form the floor argument needs. -/
def LowerBound (m : ℝ) (M : E →L[ℝ] E) : Prop := ∀ v : E, m * ‖v‖ ^ 2 ≤ ⟪v, M v⟫

/-- **Lower bounds add** — the Weyl inequality the appendix invokes for `λ_min(A + B)`. -/
theorem LowerBound_add {a b : ℝ} {A B : E →L[ℝ] E}
    (hA : LowerBound a A) (hB : LowerBound b B) : LowerBound (a + b) (A + B) := by
  intro v
  have h := add_le_add (hA v) (hB v)
  calc (a + b) * ‖v‖ ^ 2 = a * ‖v‖ ^ 2 + b * ‖v‖ ^ 2 := by ring
    _ ≤ ⟪v, A v⟫ + ⟪v, B v⟫ := h
    _ = ⟪v, (A + B) v⟫ := by simp [inner_add_right]

/-- **The congruence step.**  If `s` bounds `S` below, then for self-adjoint `K` the congruence
`K S K` is bounded below by `s ‖K v‖²` — the appendix's `K̃ S̃ K̃ ⪰ s_min K̃²`. -/
theorem LowerBound_congruence {s : ℝ} {S K : E →L[ℝ] E} (hS : LowerBound s S)
    (hK : ∀ x y : E, ⟪K x, y⟫ = ⟪x, K y⟫) (v : E) :
    s * ‖K v‖ ^ 2 ≤ ⟪v, (K.comp (S.comp K)) v⟫ := by
  have h := hS (K v)
  have hmove : ⟪v, (K.comp (S.comp K)) v⟫ = ⟪K v, S (K v)⟫ := by
    simp only [ContinuousLinearMap.comp_apply]
    exact (hK v (S (K v))).symm
  rw [hmove]; exact h

/-- **The spectral-mapping step, variationally.**  For `K = I − (D+I)⁻¹` — the gain `D(D+I)⁻¹` — a
lower bound `m` on `D` gives the lower bound `f m = m/(1+m)` on `K`, with `f` the increasing map of
the floor recursion.  No functional calculus: writing `w = (D+I)⁻¹ v`, self-adjointness gives
`⟪v, w⟫ = ⟪D w + w, w⟫ ≥ (1+m)‖w‖²`, and Cauchy–Schwarz turns that into `‖w‖ ≤ ‖v‖/(1+m)`, whence
`⟪v, K v⟫ = ‖v‖² − ⟪v, w⟫ ≥ ‖v‖² m/(1+m)`. -/
theorem LowerBound_gain {m : ℝ} (hm : 0 ≤ m) {D K : E →L[ℝ] E}
    (hD : LowerBound m D) (hDsa : ∀ x y : E, ⟪D x, y⟫ = ⟪x, D y⟫)
    (hinv : ∀ v : E, ∃ w : E, D w + w = v ∧ K v = v - w) :
    LowerBound (m / (1 + m)) K := by
  intro v
  obtain ⟨w, hw, hKv⟩ := hinv v
  have h1m : (0:ℝ) < 1 + m := by linarith
  have hCS : ⟪v, w⟫ ≤ ‖v‖ * ‖w‖ := real_inner_le_norm v w
  have hvw : (1 + m) * ‖w‖ ^ 2 ≤ ⟪v, w⟫ := by
    have hsa : ⟪D w, w⟫ = ⟪w, D w⟫ := hDsa w w
    have hDw : m * ‖w‖ ^ 2 ≤ ⟪w, D w⟫ := hD w
    have hexp : ⟪v, w⟫ = ⟪D w, w⟫ + ‖w‖ ^ 2 := by
      rw [← hw, inner_add_left, real_inner_self_eq_norm_sq]
    rw [hexp, hsa]; nlinarith
  have hcap : ⟪v, w⟫ ≤ ‖v‖ ^ 2 / (1 + m) := by
    rcases eq_or_lt_of_le (norm_nonneg w) with h | h
    · have hw0 : ‖w‖ = 0 := h.symm
      have hle0 : ⟪v, w⟫ ≤ 0 := by
        calc ⟪v, w⟫ ≤ ‖v‖ * ‖w‖ := hCS
          _ = 0 := by rw [hw0]; ring
      have hnn : (0:ℝ) ≤ ‖v‖ ^ 2 / (1 + m) := by positivity
      linarith
    · have hnorm : (1 + m) * ‖w‖ ≤ ‖v‖ := by nlinarith [le_trans hvw hCS]
      calc ⟪v, w⟫ ≤ ‖v‖ * ‖w‖ := hCS
        _ ≤ ‖v‖ ^ 2 / (1 + m) := by
              rw [le_div_iff₀ h1m]; nlinarith [norm_nonneg v]
  have hKval : ⟪v, K v⟫ = ‖v‖ ^ 2 - ⟪v, w⟫ := by
    rw [hKv, inner_sub_right, real_inner_self_eq_norm_sq]
  have hrw : m / (1 + m) * ‖v‖ ^ 2 = ‖v‖ ^ 2 - ‖v‖ ^ 2 / (1 + m) := by field_simp; ring
  rw [hKval, hrw]
  linarith [hcap]

/-- **One round of the floor recursion.**  Given the spectral-mapping input `λ_min(K) ≥ f` and a
lower bound `s` on `S`, the next iterate `K + K S K` is bounded below by `f + s · f²` — the
appendix's `g(λ)`, since `f = λ/(1+λ)`. -/
theorem floor_step {f s : ℝ} {S K : E →L[ℝ] E} (hf : LowerBound f K) (hfnn : 0 ≤ f)
    (hS : LowerBound s S) (hsnn : 0 ≤ s)
    (hK : ∀ x y : E, ⟪K x, y⟫ = ⟪x, K y⟫)
    (hKsq : ∀ v : E, f ^ 2 * ‖v‖ ^ 2 ≤ ‖K v‖ ^ 2) :
    LowerBound (f + s * f ^ 2) (K + K.comp (S.comp K)) := by
  intro v
  have h1 : f * ‖v‖ ^ 2 ≤ ⟪v, K v⟫ := hf v
  have h2 : s * ‖K v‖ ^ 2 ≤ ⟪v, (K.comp (S.comp K)) v⟫ := LowerBound_congruence hS hK v
  have h3 : s * (f ^ 2 * ‖v‖ ^ 2) ≤ s * ‖K v‖ ^ 2 := mul_le_mul_of_nonneg_left (hKsq v) hsnn
  calc (f + s * f ^ 2) * ‖v‖ ^ 2 = f * ‖v‖ ^ 2 + s * (f ^ 2 * ‖v‖ ^ 2) := by ring
    _ ≤ ⟪v, K v⟫ + ⟪v, (K.comp (S.comp K)) v⟫ := by linarith
    _ = ⟪v, (K + K.comp (S.comp K)) v⟫ := by simp [inner_add_right]

/-- **The floor persists**, by the scalar facts already checked.  The appendix's `g` is exactly
`RecoverConvergence`'s `Ψ` at `d⋆ = s_min − 1`: writing `f = λ/(1+λ)`, one has
`g(λ) = f + (1 + d⋆) f²  = Ψ(1 + d⋆) λ`, so `g(λ) ≥ λ` for `0 < λ ≤ d⋆` is `Recover.psi_sub_self_sign`,
and the invariant interval of `Recover.psi_mem_interval` gives the permanent floor
`λ_flat = min(λ₀, d⋆)`. -/
theorem floor_persists {dstar lam : ℝ} (hstar : 0 < dstar) (hlam : 0 < lam) (hle : lam ≤ dstar) :
    lam ≤ Recover.Psi (1 + dstar) lam := by
  have h := Recover.psi_sub_self_sign hstar hlam
  rcases eq_or_lt_of_le hle with h' | h'
  · subst h'; rw [Recover.psi_fixed hlam]
  · nlinarith [h, sub_pos.mpr h']

/-- And the floor is uniform in the round: starting from `λ₀ > 0`, every iterate of the scalar floor
recursion stays at or above `min(λ₀, d⋆)`.  (Restated from `Recover.psi_mem_interval`, which proves
the interval invariant.) -/
theorem floor_uniform {dstar lam0 : ℝ} (hstar : 0 < dstar) (h0 : 0 < lam0)
    (lam : ℕ → ℝ) (hl0 : lam 0 = lam0)
    (hrec : ∀ t, lam (t + 1) = Recover.Psi (1 + dstar) (lam t)) :
    ∀ t, min lam0 dstar ≤ lam t := by
  have key : ∀ t, min lam0 dstar ≤ lam t ∧ lam t ≤ max lam0 dstar := by
    intro t; induction t with
    | zero => rw [hl0]; exact ⟨min_le_left _ _, le_max_left _ _⟩
    | succ n ih => rw [hrec n]; exact Recover.psi_mem_interval hstar h0 ih.1 ih.2
  exact fun t => (key t).1

end MatrixFloor

-- #print axioms audit (matrix reduction for p:recover (ii))
#print axioms MatrixFloor.LowerBound_add
#print axioms MatrixFloor.LowerBound_congruence
#print axioms MatrixFloor.floor_step
#print axioms MatrixFloor.floor_persists
#print axioms MatrixFloor.floor_uniform
#print axioms MatrixFloor.LowerBound_gain
