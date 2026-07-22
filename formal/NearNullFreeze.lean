/-
# The near-null refinements: `p:nearnull` (i) and `p:mapnearnull` (i)

Both results say the *exact* blind-subspace statements degrade gracefully rather than switching off
when the operator presents a near-null direction instead of a genuine kernel.  Each rests on one
inequality, isolated and machine-checked here.

## `p:nearnull` (i) — the precision freeze bound

One loop step moves the prior's precision by `Σ_q⁻¹ − Σ_ρ⁻¹ = Aᵀ B A` with
`B = Γ⁻¹ − Γ⁻¹ Q⁻¹ Γ⁻¹`.  Sandwiching with a unit `v` gives

    |vᵀ (Σ_q⁻¹ − Σ_ρ⁻¹) v| = |⟪A v, B (A v)⟫| ≤ ‖B‖ ‖A v‖²,

so a direction with singular value `σ_v` has its precision moved by at most `c σ_v²`, and the bound
is *exactly zero* on the kernel (`A v = 0`) — the exact freeze recovered as the endpoint.  That is
`quadratic_form_abs_le` and `freeze_bound`/`freeze_exact_of_kernel` below.

## `p:mapnearnull` (i) — the single-best pinning bound

With a penalty whose sections along `v` are `μ_v`-strongly convex, the archived coordinate sits
within `|⟪A v, d(y)⟫| / μ_v` of the penalty's conditional minimizer `φ`, hence within
`‖A v‖ ‖d(y)‖ / μ_v` — a tube that closes at rate `‖A v‖` as the illumination vanishes, and closes
*exactly* on the kernel.  That is `strongly_convex_pin` and `pin_le_norm_mul`.

Strong convexity enters through its derivative form — `(g' a − g' b)(a − b) ≥ μ (a − b)²` — which is
what the appendix uses and what makes the bound one line; it is taken as the hypothesis rather than
re-derived from a convexity predicate.

No `sorry`/`admit`; `#print axioms` on each result at the end.
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Normed.Operator.ContinuousLinearMap
import Mathlib.Analysis.InnerProductSpace.LinearMap
import Mathlib.Analysis.Calculus.LocalExtr.Basic

namespace NearNullFreeze

open RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **The sandwich bound.**  For a bounded operator `B` and any vector `u`, the quadratic form is
controlled by the operator norm: `|⟪u, B u⟫| ≤ ‖B‖ ‖u‖²`.  (Cauchy–Schwarz, then `‖B u‖ ≤ ‖B‖ ‖u‖`.) -/
theorem quadratic_form_abs_le (B : E →L[ℝ] E) (u : E) :
    |⟪u, B u⟫| ≤ ‖B‖ * ‖u‖ ^ 2 := by
  have hCS : |⟪u, B u⟫| ≤ ‖u‖ * ‖B u‖ := abs_real_inner_le_norm u (B u)
  have hop : ‖B u‖ ≤ ‖B‖ * ‖u‖ := B.le_opNorm u
  have hu : (0:ℝ) ≤ ‖u‖ := norm_nonneg u
  calc |⟪u, B u⟫| ≤ ‖u‖ * ‖B u‖ := hCS
    _ ≤ ‖u‖ * (‖B‖ * ‖u‖) := by exact mul_le_mul_of_nonneg_left hop hu
    _ = ‖B‖ * ‖u‖ ^ 2 := by ring

/-- **`p:nearnull` (i) — the precision freeze bound.**  The precision moved along `v` by one step is
`⟪A v, B (A v)⟫`, and it is bounded by `‖B‖ ‖A v‖²`: a direction is frozen to the extent the operator
fails to illuminate it, the movement vanishing quadratically in the singular value. -/
theorem freeze_bound (A : E →L[ℝ] E) (B : E →L[ℝ] E) (v : E) :
    |⟪A v, B (A v)⟫| ≤ ‖B‖ * ‖A v‖ ^ 2 :=
  quadratic_form_abs_le B (A v)

/-- **The exact freeze is the endpoint.**  On a genuine kernel direction (`A v = 0`) the bound is
zero, so the precision does not move at all — `eq:freeze` recovered from the graded bound. -/
theorem freeze_exact_of_kernel (A : E →L[ℝ] E) (B : E →L[ℝ] E) {v : E} (hv : A v = 0) :
    ⟪A v, B (A v)⟫ = 0 := by
  rw [hv]
  simp

/-- **The sandwich reads through the adjoint.**  If an operator factors through `A` as `A* ∘ B ∘ A`,
its quadratic form at `v` is the form of `B` at `A v`.  This is the link between
\eqref{eq:nearnull-freeze} — which states the precision difference in the factored form `Aᵀ B A` —
and `freeze_bound`, which bounds it. -/
theorem sandwich_factors (A B : E →L[ℝ] E) (Astar : E →L[ℝ] E)
    (hadj : ∀ x y : E, ⟪A x, y⟫ = ⟪x, Astar y⟫) (v : E) :
    ⟪v, (Astar.comp (B.comp A)) v⟫ = ⟪A v, B (A v)⟫ := by
  simp only [ContinuousLinearMap.comp_apply]
  exact (hadj v (B (A v))).symm

/-- **`p:nearnull` (i) in the form the paper states it.**  A precision difference that factors
through the operator is moved by at most `‖B‖ ‖A v‖²` along `v`, and not at all on the kernel.
Given \eqref{eq:nearnull-freeze}'s factorization, this is the proposition's bound. -/
theorem precision_move_le (A B Astar : E →L[ℝ] E)
    (hadj : ∀ x y : E, ⟪A x, y⟫ = ⟪x, Astar y⟫) (v : E) :
    |⟪v, (Astar.comp (B.comp A)) v⟫| ≤ ‖B‖ * ‖A v‖ ^ 2 := by
  rw [sandwich_factors A B Astar hadj v]
  exact freeze_bound A B v

/-- …and vanishes exactly on the kernel: the exact freeze, recovered from the factored form. -/
theorem precision_move_zero_of_kernel (A B Astar : E →L[ℝ] E)
    (hadj : ∀ x y : E, ⟪A x, y⟫ = ⟪x, Astar y⟫) {v : E} (hv : A v = 0) :
    ⟪v, (Astar.comp (B.comp A)) v⟫ = 0 := by
  rw [sandwich_factors A B Astar hadj v, hv]
  simp

/-- **`p:mapnearnull` (i) — strong convexity pins the archived coordinate.**  If `g` has a
`μ`-monotone derivative (the derivative form of `μ`-strong convexity) and `t⋆` is a critical point,
then any `t` sits within `|g' t| / μ` of `t⋆`.  Applied along `v` with `g' t = ⟪A v, d y⟫`, this is
the appendix's tube. -/
theorem strongly_convex_pin {g' : ℝ → ℝ} {mu tstar t : ℝ} (hmu : 0 < mu)
    (hmono : ∀ a b : ℝ, mu * (a - b) ^ 2 ≤ (g' a - g' b) * (a - b))
    (hcrit : g' tstar = 0) :
    |t - tstar| ≤ |g' t| / mu := by
  have hkey : mu * (t - tstar) ^ 2 ≤ (g' t - g' tstar) * (t - tstar) := hmono t tstar
  rw [hcrit, sub_zero] at hkey
  have habs : g' t * (t - tstar) ≤ |g' t| * |t - tstar| := by
    calc g' t * (t - tstar) ≤ |g' t * (t - tstar)| := le_abs_self _
      _ = |g' t| * |t - tstar| := abs_mul _ _
  have hsq : mu * |t - tstar| ^ 2 ≤ |g' t| * |t - tstar| := by
    have : (t - tstar) ^ 2 = |t - tstar| ^ 2 := (sq_abs _).symm
    rw [← this]; linarith
  rcases eq_or_lt_of_le (abs_nonneg (t - tstar)) with h | h
  · rw [← h]; positivity
  · rw [le_div_iff₀ hmu]
    nlinarith [hsq, h]

/-- The pinning tube in the form the paper states it: `‖A v‖ ‖d‖ / μ`, by Cauchy–Schwarz on the
directional derivative.  The tube closes at rate `‖A v‖` as the illumination vanishes. -/
theorem pin_le_norm_mul {mu : ℝ} (hmu : 0 < mu) (Av d : E) {gap : ℝ}
    (hgap : gap ≤ |⟪Av, d⟫| / mu) :
    gap ≤ ‖Av‖ * ‖d‖ / mu := by
  refine le_trans hgap ?_
  gcongr
  exact abs_real_inner_le_norm Av d

/-- **The variational identification** (`p:mapnearnull`'s missing link).  If `x̂` minimizes the
objective and the section along `v` is differentiable with derivative `⟪A v, d⟫ + rv` at `t = 0`
(the data term's chain rule plus the penalty's directional derivative `rv`), then stationarity forces

    rv = −⟪A v, d⟫,

so the penalty's directional derivative at the archived point is exactly the quantity the pinning
bound is stated in.  Composing with `strongly_convex_pin` and `pin_le_norm_mul` closes
`p:mapnearnull` (i): the archived coordinate lies within `‖A v‖ ‖d‖ / μ_v` of the penalty's
conditional minimizer. -/
theorem penalty_deriv_eq_neg_inner {g : ℝ → ℝ} {Av d : E} {rv : ℝ}
    (hderiv : HasDerivAt g (⟪Av, d⟫ + rv) 0) (hmin : ∀ t : ℝ, g 0 ≤ g t) :
    rv = -⟪Av, d⟫ := by
  have hlocal : IsLocalMin g 0 := Filter.Eventually.of_forall hmin
  have := hlocal.hasDerivAt_eq_zero hderiv
  linarith

/-- Hence the penalty's directional derivative is bounded by the illumination: `|rv| ≤ ‖A v‖ ‖d‖`,
vanishing on the kernel.  This is the input `strongly_convex_pin` consumes. -/
theorem abs_penalty_deriv_le {g : ℝ → ℝ} {Av d : E} {rv : ℝ}
    (hderiv : HasDerivAt g (⟪Av, d⟫ + rv) 0) (hmin : ∀ t : ℝ, g 0 ≤ g t) :
    |rv| ≤ ‖Av‖ * ‖d‖ := by
  rw [penalty_deriv_eq_neg_inner hderiv hmin, abs_neg]
  exact abs_real_inner_le_norm Av d

end NearNullFreeze

-- #print axioms audit (p:nearnull (i), p:mapnearnull (i))
#print axioms NearNullFreeze.quadratic_form_abs_le
#print axioms NearNullFreeze.freeze_bound
#print axioms NearNullFreeze.freeze_exact_of_kernel
#print axioms NearNullFreeze.strongly_convex_pin
#print axioms NearNullFreeze.pin_le_norm_mul
#print axioms NearNullFreeze.penalty_deriv_eq_neg_inner
#print axioms NearNullFreeze.abs_penalty_deriv_le
#print axioms NearNullFreeze.sandwich_factors
#print axioms NearNullFreeze.precision_move_le
#print axioms NearNullFreeze.precision_move_zero_of_kernel
