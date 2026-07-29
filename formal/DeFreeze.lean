import Mathlib.Analysis.InnerProductSpace.Projection.Submodule
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.LinearAlgebra.Prod
import SingleBestCollapse

/-!
# Two structural results of the paper: `c:augment` (de-freezing) and `r:hilbert`

## `c:augment` (general half) — an added channel de-freezes exactly the directions it resolves

Let `A` be the operator with blind subspace `ker A`, and let a new channel resolve a subspace
`𝔅₁ ⊆ ker A` — i.e. the channel `C` (the paper's `B₁ᵀ`) has `ker C = 𝔅₁ᗮ`.  Curating jointly against
`(y, z)` is the loop map for the **stacked** operator `A.prod C`, whose kernel — the new frozen set —
is `ker A ⊓ 𝔅₁ᗮ`; and since `ker A = 𝔅₁ ⊔ (ker A ⊓ 𝔅₁ᗮ)` with `𝔅₁ ⊓ (ker A ⊓ 𝔅₁ᗮ) = ⊥`, **exactly**
`𝔅₁` leaves the frozen set and nothing else.

* `defreeze_new_frozen` — `ker (A.prod C) = ker A ⊓ 𝔅₁ᗮ`.
* `defreeze_decomposition` — `ker A = 𝔅₁ ⊔ ker (A.prod C)` (the orthomodular split).
* `defreeze_disjoint` — `𝔅₁ ⊓ ker (A.prod C) = ⊥` (internal direct sum: exactly `𝔅₁` de-freezes).

## `r:hilbert` — the freeze/collapse in function space (bounded operator, closed kernel)

The same cancellation as `p:map`/`p:blind` runs for a bounded linear operator `A : H →L[ℝ] K` on
Hilbert spaces with a (possibly nontrivial) **closed** kernel:

* `hilbert_fiber_invariance` — `A (x + n) = A x` for `n ∈ ker A` (the invariance, reused).
* `hilbert_ker_isClosed` — `ker A` is closed (continuity).
* `hilbert_collapse` — every global minimizer of `D(y, A·) + R` has a blind component minimizing the
  penalty over its own fiber (`SingleBestCollapse.blind_minimizes_penalty_on_fiber_coset` for `⇑A`).

## Honesty

No `sorry`/`admit`.  `#print axioms` on each main theorem lists only `propext`, `Classical.choice`,
`Quot.sound` (no `sorryAx`).  Reconciliation at the end.
-/

open LinearMap Submodule

namespace DeFreeze

/-! ## `c:augment` — de-freezing (general) -/

section Augment

variable {E W Z : Type*}
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [AddCommGroup W] [Module ℝ W] [AddCommGroup Z] [Module ℝ Z]
  (A : E →ₗ[ℝ] W) (C : E →ₗ[ℝ] Z) (𝔅₁ : Submodule ℝ E)

omit [FiniteDimensional ℝ E] in
/-- **The new frozen set is the stacked kernel `ker A ⊓ 𝔅₁ᗮ`.** Curating jointly against `(y, z)` is
the loop for the stacked operator `A.prod C`; its kernel — the blind subspace after augmentation — is
`ker A ⊓ ker C = ker A ⊓ 𝔅₁ᗮ` when the new channel resolves exactly `𝔅₁` (`ker C = 𝔅₁ᗮ`). -/
theorem defreeze_new_frozen (hC : ker C = 𝔅₁ᗮ) :
    ker (A.prod C) = ker A ⊓ 𝔅₁ᗮ := by
  rw [ker_prod, hC]

/-- **Exactly `𝔅₁` leaves the frozen set** (the orthomodular decomposition): for `𝔅₁ ≤ ker A`,
`ker A = 𝔅₁ ⊔ ker (A.prod C)`. -/
theorem defreeze_decomposition (h₁ : 𝔅₁ ≤ ker A) (hC : ker C = 𝔅₁ᗮ) :
    ker A = 𝔅₁ ⊔ ker (A.prod C) := by
  rw [defreeze_new_frozen A C 𝔅₁ hC, inf_comm]
  exact (Submodule.sup_orthogonal_inf_of_hasOrthogonalProjection h₁).symm

omit [FiniteDimensional ℝ E] in
/-- **…and nothing else** (internal direct sum): `𝔅₁ ⊓ ker (A.prod C) = ⊥`, so the de-frozen part is
exactly `𝔅₁`. -/
theorem defreeze_disjoint (hC : ker C = 𝔅₁ᗮ) :
    𝔅₁ ⊓ ker (A.prod C) = ⊥ := by
  rw [defreeze_new_frozen A C 𝔅₁ hC]
  refine le_antisymm ?_ bot_le
  calc 𝔅₁ ⊓ (ker A ⊓ 𝔅₁ᗮ) ≤ 𝔅₁ ⊓ 𝔅₁ᗮ :=
        inf_le_inf_left _ inf_le_right
    _ = ⊥ := Submodule.inf_orthogonal_eq_bot 𝔅₁

end Augment

/-! ## `r:hilbert` — the freeze/collapse in function space -/

section Hilbert

variable {H K : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [NormedAddCommGroup K] [InnerProductSpace ℝ K]

/-- **The blind-fiber invariance for a bounded operator** (`r:hilbert`): `A (x + n) = A x` for
`n ∈ ker A` — the same cancellation as the finite-dimensional case. -/
theorem hilbert_fiber_invariance (A : H →L[ℝ] K) :
    ∀ (x n : H), n ∈ ker (A : H →ₗ[ℝ] K) → A (x + n) = A x :=
  SingleBestCollapse.linear_blind_invariance (A : H →ₗ[ℝ] K)

/-- **The blind subspace of a bounded operator is closed** (continuity) — the "closed kernel" of
`r:hilbert`. -/
theorem hilbert_ker_isClosed (A : H →L[ℝ] K) :
    IsClosed (ker (A : H →ₗ[ℝ] K) : Set H) :=
  A.isClosed_ker

/-- **`r:hilbert` — the single-best collapse in function space.** For a bounded linear operator `A`
with blind subspace `N = ker A`, a data term reaching `x` only through `A x`, and any penalty `R`,
every global minimizer's blind component minimizes the penalty over its own fiber `x̂ + N`:
`R x̂ ≤ R (x̂ + n)` for all `n ∈ ker A` — `p:map`'s cancellation, verbatim, in a Hilbert space. -/
theorem hilbert_collapse (A : H →L[ℝ] K) {Y : Type*} (D : Y → K → ℝ) (R : H → ℝ)
    {y : Y} {xhat : H} (hmin : ∀ x, D y (A xhat) + R xhat ≤ D y (A x) + R x) :
    ∀ n ∈ ker (A : H →ₗ[ℝ] K), R xhat ≤ R (xhat + n) :=
  SingleBestCollapse.blind_minimizes_penalty_on_fiber_coset (⇑A) D R
    (ker (A : H →ₗ[ℝ] K)) (hilbert_fiber_invariance A) hmin

end Hilbert

/-!
## Reconciliation — statement + proof level

**(a)** already in the text; **(b)** load-bearing but not stated; **(c)** automatic.

### `c:augment`

* the stacked operator is `A.prod C` and the new frozen set is its kernel — **(a)**: "curating
  jointly against `(y, z)` is the map `loopop` for the stacked operator `[A; B₁ᵀ]`, whose kernel is
  …".  The `p:blind` freeze acts on the kernel; augmentation replaces `ker A` by `ker (A.prod C)`.
* `ker C = 𝔅₁ᗮ` (the new channel resolves exactly `𝔅₁`) — **(a)**: the paper's `B₁` has orthonormal
  columns spanning `𝔅₁`, so its adjoint `B₁ᵀ` has kernel `(range B₁)ᗮ = 𝔅₁ᗮ`.  Taken here as the
  channel's defining property (the orthonormality/adjoint step is the standard `ker Bᵀ = (range B)ᗮ`).
* `𝔅₁ ≤ ker A`, `FiniteDimensional ℝ E` — **(a)/(c)**: `𝔅₁ ⊆ ker A` is stated; finite dimension gives
  every subspace an orthogonal projection (`HasOrthogonalProjection`), which is what the orthomodular
  split `ker A = 𝔅₁ ⊔ (ker A ⊓ 𝔅₁ᗮ)` needs.  **Proof level:** the paper's one line "since
  `ker A = 𝔅₁ ⊕ (ker A ∩ 𝔅₁⊥)`" is exactly `Submodule.sup_orthogonal_inf_of_hasOrthogonalProjection`
  (`⊔` + `⊓ 𝔅₁ᗮ`) plus `𝔅₁ ⊓ 𝔅₁ᗮ = ⊥`; Lean pins that the direct-sum claim needs the orthogonal
  projection (finite dimension or completeness of `𝔅₁`), which the prose leaves implicit.

### `r:hilbert`

* `A : H →L[ℝ] K` bounded, `ker A` closed and possibly nontrivial — **(a)**: "a bounded linear
  operator with nontrivial closed kernel".  **Proof level:** the collapse is entirely reused from the
  finite-dimensional `SingleBestCollapse` — the cancellation `A (x+n) = A x` and the fiber-optimality
  are **dimension-agnostic** (pure linearity), so `r:hilbert`'s "same cancellation in function space"
  is literally the same theorem instantiated at `⇑A`; the only genuinely Hilbert facts are that
  `ker A` is closed (`hilbert_ker_isClosed`, continuity) and nontrivial.  The paper devotes a remark
  to this; the formalization shows it needs no new argument beyond continuity of `A`.
-/

end DeFreeze

/-! ### Axiom audit

The doc comment above claims these rest only on Lean's three standard axioms. These commands
are what make that checkable rather than asserted: each must print `propext, Classical.choice,
Quot.sound` and never `sorryAx`. -/

#print axioms DeFreeze.defreeze_decomposition
#print axioms DeFreeze.defreeze_disjoint
#print axioms DeFreeze.hilbert_collapse
