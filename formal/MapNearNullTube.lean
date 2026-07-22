/-
# `p:mapnearnull` (i) as ONE theorem: the near-null pinning tube

`NearNullFreeze` machine-checks the three steps of the near-null single-best pinning as separate
lemmas sharing a module but not a statement: `strongly_convex_pin` (a `μ`-monotone derivative pins
any point within `|g' t|/μ` of a critical point), `pin_le_norm_mul` (its Cauchy–Schwarz form), and
`penalty_deriv_eq_neg_inner` (stationarity of the minimized objective identifies the penalty
section's derivative with `−⟪A v, d⟫`).  Its docstring promises that composing them closes
`p:mapnearnull` (i); this module performs that composition as a single theorem, `pinning_tube`:

    |⟪v, x̂⟫ − φ| ≤ |⟪A v, d⟫| / μ_v ≤ ‖A v‖ ‖d‖ / μ_v,

the two-inequality display of `p:mapnearnull` (i) — the archived coordinate `vᵀx̂` sits within a
tube of the penalty's conditional minimizer `φ(Π x̂)` whose width closes at rate `‖A v‖` as the
illumination vanishes, and closes exactly on the kernel.

## Setting and hypotheses (vs the paper's)

The archive stores `x̂ ∈ argmin D(y, A x) + R(x)` (eq:varest with linear `F = A`).  Fix `v` and
write `Π x̂ := x̂ − ⟪v, x̂⟫ v` for the remaining coordinates, so `x̂ = Π x̂ + ⟪v, x̂⟫ v` holds by
construction (the paper's unit normalization of `v` is not needed for part (i); it only calibrates
the parametrization of `μ_v`).  The hypotheses are those the three lemmas jointly consume:

* `hsec` — the 1-D penalty section `u ↦ R(Π x̂ + u v)` is differentiable with derivative `f'`.
  The paper assumes only convexity and runs the proof through subgradients; the machine-checked
  route works with a genuine derivative of the section, as `NearNullFreeze`'s docstring already
  records for `strongly_convex_pin`.
* `hmono` — the derivative form of `μ_v`-strong convexity of that section,
  `(f' a − f' b)(a − b) ≥ μ_v (a − b)²` (what the appendix uses; implied by the paper's
  "`t ↦ R(x + t v) − μ_v t²/2` convex" under `hsec`).
* `hphi` — `φ` minimizes the section: the penalty's conditional minimizer along `v` given the
  remaining coordinates, exactly the paper's `φ(Π x̂)`.  Its criticality `f' φ = 0` is *derived*
  (interior first-order condition), not assumed.
* `hd` — the data term is differentiable at `A x̂` with gradient `d` there (the paper's `d(y)`),
  taken in Fréchet form `HasFDerivAt Ddat (innerSL ℝ d) (A x̂)`.
* `hmin` — `x̂` minimizes the objective along the line `x̂ + t v` (implied by, and weaker than,
  the paper's global minimality of `x̂`).

## The glue, proved rather than hypothesized

The composition needs three facts the `NearNullFreeze` lemmas do not supply, all proved here:
`line_decomp` (the affine reparametrization `x̂ + t v = Π x̂ + (⟪v, x̂⟫ + t) v`, definitional
algebra), `dataSection_hasDerivAt` (the chain rule giving the data term's section the derivative
`⟪A v, d⟫` at `t = 0`), and `penaltySection_hasDerivAt` (the shift rule giving the penalty's
section the derivative `f' ⟪v, x̂⟫` at `t = 0`).  Together they let
`penalty_deriv_eq_neg_inner` identify the abstract `g'` of `strongly_convex_pin` with
`−⟪A v, d⟫` — the link the module docstring left on paper.

No `sorry`/`admit`; `#print axioms` on every public result at the end.
-/
import NearNullFreeze
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Comp

namespace MapNearNullTube

open RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- **The affine reparametrization** of the line through `x̂` along `v` by the archived coordinate:
`x̂ + t v = Π x̂ + (⟪v, x̂⟫ + t) v` with `Π x̂ = x̂ − ⟪v, x̂⟫ v`.  Pure module algebra — the glue
between the objective's section (parametrized at `x̂`) and the penalty's section (parametrized at
the remaining coordinates `Π x̂`). -/
theorem line_decomp (xhat v : E) (t : ℝ) :
    xhat + t • v = (xhat - ⟪v, xhat⟫ • v) + (⟪v, xhat⟫ + t) • v := by
  rw [add_smul]
  abel

/-- **The data term's section has derivative `⟪A v, d⟫` at the minimizer** — the chain rule for
`t ↦ D(y, A(x̂ + t v))` at `t = 0`, given the paper's hypothesis that the data term is
differentiable at `A x̂` with gradient `d` there. -/
theorem dataSection_hasDerivAt (Ddat : F → ℝ) (A : E →L[ℝ] F) (xhat v : E) (d : F)
    (hd : HasFDerivAt Ddat (innerSL ℝ d) (A xhat)) :
    HasDerivAt (fun t : ℝ => Ddat (A (xhat + t • v))) ⟪A v, d⟫ 0 := by
  -- the image line `t ↦ A x̂ + t (A v)` and its velocity
  have h1 : HasDerivAt (fun t : ℝ => t • (A v)) ((1 : ℝ) • (A v)) 0 :=
    (hasDerivAt_id (0 : ℝ)).smul_const (A v)
  rw [one_smul] at h1
  have hline : HasDerivAt (fun t : ℝ => A xhat + t • (A v)) (A v) 0 := h1.const_add (A xhat)
  -- chain rule through the gradient of the data term
  have hcomp : HasDerivAt (Ddat ∘ fun t : ℝ => A xhat + t • (A v)) (innerSL ℝ d (A v)) 0 :=
    hd.comp_hasDerivAt_of_eq (0 : ℝ) hline (by simp)
  -- `A` is linear, so the image line is the image of the line
  have hfun : (fun t : ℝ => Ddat (A (xhat + t • v)))
      = (Ddat ∘ fun t : ℝ => A xhat + t • (A v)) := by
    funext t
    simp [Function.comp_apply, map_add, map_smul]
  rw [hfun, real_inner_comm]
  exact hcomp

/-- **The penalty's section has derivative `f' ⟪v, x̂⟫` at the minimizer** — the shift rule:
`t ↦ R(x̂ + t v)` is the section `u ↦ R(Π x̂ + u v)` reparametrized by `u = ⟪v, x̂⟫ + t`
(`line_decomp`), so at `t = 0` its derivative is the section's derivative at the archived
coordinate. -/
theorem penaltySection_hasDerivAt (R : E → ℝ) (xhat v : E) (f' : ℝ → ℝ)
    (hsec : ∀ s : ℝ, HasDerivAt (fun u : ℝ => R ((xhat - ⟪v, xhat⟫ • v) + u • v)) (f' s) s) :
    HasDerivAt (fun t : ℝ => R (xhat + t • v)) (f' ⟪v, xhat⟫) 0 := by
  have hshift : HasDerivAt (fun t : ℝ => ⟪v, xhat⟫ + t) 1 0 :=
    (hasDerivAt_id (0 : ℝ)).const_add ⟪v, xhat⟫
  have hc := (hsec (⟪v, xhat⟫ + 0)).comp (0 : ℝ) hshift
  rw [add_zero, mul_one] at hc
  have hfun : (fun t : ℝ => R (xhat + t • v))
      = ((fun u : ℝ => R ((xhat - ⟪v, xhat⟫ • v) + u • v)) ∘ fun t : ℝ => ⟪v, xhat⟫ + t) := by
    funext t
    simp only [Function.comp_apply]
    rw [line_decomp]
  rw [hfun]
  exact hc

/-- **`p:mapnearnull` (i) — the pinning tube, composed.**  For a minimizer `x̂` of the variational
archive with linear forward operator `A`, data term differentiable at `A x̂` with gradient `d`, and
penalty whose section along `v` given the remaining coordinates `Π x̂ = x̂ − ⟪v, x̂⟫ v` is
differentiable with `μ_v`-strongly monotone derivative, the archived coordinate is pinned to the
penalty's conditional minimizer `φ`:

    |⟪v, x̂⟫ − φ| ≤ |⟪A v, d⟫| / μ_v  ≤  ‖A v‖ ‖d‖ / μ_v.

The tube closes at rate `‖A v‖` as the illumination vanishes, and exactly on the kernel — the
single-best collapse of `p:map`, recovered gracefully on the near-null.  Composes
`NearNullFreeze.penalty_deriv_eq_neg_inner` (stationarity), `NearNullFreeze.strongly_convex_pin`
(the pinning), and `NearNullFreeze.pin_le_norm_mul` (Cauchy–Schwarz) through this module's glue. -/
theorem pinning_tube (A : E →L[ℝ] F) (v xhat : E) (d : F)
    (R : E → ℝ) (Ddat : F → ℝ) (f' : ℝ → ℝ) (phi mu : ℝ)
    (hmu : 0 < mu)
    -- the penalty's 1-D section along `v` given `Π x̂`, differentiable with derivative `f'`
    (hsec : ∀ s : ℝ, HasDerivAt (fun u : ℝ => R ((xhat - ⟪v, xhat⟫ • v) + u • v)) (f' s) s)
    -- derivative form of `μ_v`-strong convexity of that section
    (hmono : ∀ a b : ℝ, mu * (a - b) ^ 2 ≤ (f' a - f' b) * (a - b))
    -- `φ`: the penalty's conditional minimizer along `v` given the remaining coordinates
    (hphi : ∀ s : ℝ, R ((xhat - ⟪v, xhat⟫ • v) + phi • v)
      ≤ R ((xhat - ⟪v, xhat⟫ • v) + s • v))
    -- the data term is differentiable at `A x̂` with gradient `d` there
    (hd : HasFDerivAt Ddat (innerSL ℝ d) (A xhat))
    -- `x̂` minimizes the objective along the line through `v`
    (hmin : ∀ t : ℝ, Ddat (A xhat) + R xhat ≤ Ddat (A (xhat + t • v)) + R (xhat + t • v)) :
    |⟪v, xhat⟫ - phi| ≤ |⟪A v, d⟫| / mu ∧ |⟪A v, d⟫| / mu ≤ ‖A v‖ * ‖d‖ / mu := by
  -- the objective's section along `v` has derivative `⟪A v, d⟫ + f' ⟪v, x̂⟫` at the minimizer
  have hderiv : HasDerivAt (fun t : ℝ => Ddat (A (xhat + t • v)) + R (xhat + t • v))
      (⟪A v, d⟫ + f' ⟪v, xhat⟫) 0 :=
    (dataSection_hasDerivAt Ddat A xhat v d hd).add (penaltySection_hasDerivAt R xhat v f' hsec)
  -- minimality along the line, at the section's origin
  have hgmin : ∀ t : ℝ, Ddat (A (xhat + (0 : ℝ) • v)) + R (xhat + (0 : ℝ) • v)
      ≤ Ddat (A (xhat + t • v)) + R (xhat + t • v) := by
    intro t
    have h0 : xhat + (0 : ℝ) • v = xhat := by simp
    rw [h0]
    exact hmin t
  -- stationarity identifies the penalty section's derivative with `−⟪A v, d⟫`
  have hstat : f' ⟪v, xhat⟫ = -⟪A v, d⟫ :=
    NearNullFreeze.penalty_deriv_eq_neg_inner hderiv hgmin
  -- the conditional minimizer is a critical point of the section
  have hcrit : f' phi = 0 := by
    have hloc : IsLocalMin (fun u : ℝ => R ((xhat - ⟪v, xhat⟫ • v) + u • v)) phi :=
      Filter.Eventually.of_forall hphi
    exact hloc.hasDerivAt_eq_zero (hsec phi)
  -- strong convexity pins the archived coordinate; Cauchy–Schwarz widens the tube to `‖A v‖ ‖d‖`
  have hpin : |⟪v, xhat⟫ - phi| ≤ |f' ⟪v, xhat⟫| / mu :=
    NearNullFreeze.strongly_convex_pin hmu hmono hcrit
  rw [hstat, abs_neg] at hpin
  exact ⟨hpin, NearNullFreeze.pin_le_norm_mul hmu (A v) d le_rfl⟩

/-- **The kernel endpoint.**  On a genuine kernel direction (`A v = 0`) the tube has zero width:
the archived coordinate *is* the penalty's conditional minimizer, `p:map`'s exact collapse
recovered as the endpoint of the graded bound. -/
theorem pinning_exact_of_kernel (A : E →L[ℝ] F) (v xhat : E) (d : F)
    (R : E → ℝ) (Ddat : F → ℝ) (f' : ℝ → ℝ) (phi mu : ℝ)
    (hmu : 0 < mu)
    (hsec : ∀ s : ℝ, HasDerivAt (fun u : ℝ => R ((xhat - ⟪v, xhat⟫ • v) + u • v)) (f' s) s)
    (hmono : ∀ a b : ℝ, mu * (a - b) ^ 2 ≤ (f' a - f' b) * (a - b))
    (hphi : ∀ s : ℝ, R ((xhat - ⟪v, xhat⟫ • v) + phi • v)
      ≤ R ((xhat - ⟪v, xhat⟫ • v) + s • v))
    (hd : HasFDerivAt Ddat (innerSL ℝ d) (A xhat))
    (hmin : ∀ t : ℝ, Ddat (A xhat) + R xhat ≤ Ddat (A (xhat + t • v)) + R (xhat + t • v))
    (hker : A v = 0) :
    ⟪v, xhat⟫ = phi := by
  have h := (pinning_tube A v xhat d R Ddat f' phi mu hmu hsec hmono hphi hd hmin).1
  rw [hker] at h
  simp only [inner_zero_left, abs_zero, zero_div] at h
  have := abs_nonneg (⟪v, xhat⟫ - phi)
  have hz : |⟪v, xhat⟫ - phi| = 0 := le_antisymm h this
  have := abs_eq_zero.mp hz
  linarith

end MapNearNullTube

-- #print axioms audit (p:mapnearnull (i), composed)
#print axioms MapNearNullTube.line_decomp
#print axioms MapNearNullTube.dataSection_hasDerivAt
#print axioms MapNearNullTube.penaltySection_hasDerivAt
#print axioms MapNearNullTube.pinning_tube
#print axioms MapNearNullTube.pinning_exact_of_kernel
