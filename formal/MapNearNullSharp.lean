/-
# `p:mapnearnull`, sharpened: the kinked tube, the flat-direction divergence, and the
conditional-variance bound

`MapNearNullTube` closes `p:mapnearnull` (i) for penalties whose 1-D sections along the archived
direction are *differentiable*.  The paper assumes only convexity and runs the proof through
subgradients; this module removes the differentiability, and adds the two remaining pieces of
`p:mapnearnull` the machine had not reached.

## (2a) The tube without differentiability — `strongly_convex_pin_subgradient`, `pinning_tube_kinked`

For a 1-D penalty section `f` that is `μ`-strongly convex in the paper's sense — `t ↦ f t − μ/2 t²`
convex, no derivative assumed — and a data section `g` differentiable at the minimizer `t⋆` of
`f + g`, the minimizer is pinned to the penalty's minimizer `φ` within `|g'|/μ`.  The proof is the
subgradient argument rebuilt from secant slopes alone: convexity of the shifted section makes its
slopes monotone (`ConvexOn.slope_mono_adjacent`); reading the monotone chain between the mirrored
points `φ + t⋆ − t ≤ t` and letting `t ↑ t⋆` recovers the *full* strong-monotonicity constant `μ`
(a one-point secant comparison alone would lose a factor 2); minimality of `f + g` at `t⋆` bounds
the left secant of `f` by the data slope, whose limit is `g'` (`hasDerivAt_iff_tendsto_slope`).
No subdifferential API is needed — left/right secants suffice.  `pinning_tube_kinked` then
assembles the paper's display exactly as `MapNearNullTube.pinning_tube` did, with the convexity
hypothesis in place of the differentiable-section hypothesis: kinked penalties (ℓ¹ terms plus
damping, total variation plus damping) are now inside the theorem, and the kernel endpoint
`pinning_exact_of_kernel_kinked` recovers `p:map`'s exact collapse.

## (2b) The exactly-flat direction — `flat_argmin_eq`, `flat_selection_law`, `flat_coordinate_variance`

The sharpness witness at `μ_v = 0` (manuscript, the "penalty exactly flat along `v`" paragraph):
in the separable witness the flat coordinate's section is the pure data term `t ↦ (y − εt)²/2`,
any minimizer of it is *exactly* `y/ε` (`flat_argmin_eq`), so every measurable minimizer selection
pushes `y ∼ N(0, σ²)` to `N(0, σ²/ε²)` (`flat_selection_law`, via mathlib's
`gaussianReal_map_div_const`), with variance `σ²/ε² = σ²‖Av‖⁻²` (`flat_coordinate_variance`,
`flat_coordinate_variance_norm`) — the classical ill-posed noise amplification, diverging as the
illumination `ε = ‖A v‖` vanishes: opposite in kind to the freeze, and separated from it exactly
by `μ_v`.

## (2c) The conditional-variance bound — `p:mapnearnull` (ii)'s first display

`integral_sq_sub_condExp_le`: the conditional expectation is the `L²`-optimal `m`-measurable
predictor — for square-integrable `X` and any `m`-measurable square-integrable `Y`,
`∫ (X − E[X|m])² ≤ ∫ (X − Y)²`.  Proved by identifying `E[X|m]` with the orthogonal projection
onto `lpMeas` (`condExpL2`, mathlib's `MemLp.condExpL2_ae_eq_condExp`) and invoking the
projection's minimality (`Submodule.starProjection_minimal`).  `integral_condVar_eq` is the tower
property `E[Var(X|m)] = E[(X − E[X|m])²]` (mathlib's `condVar` + `integral_condExp`), and
`integral_condVar_le_tube` / `condVar_tube_paper` chain them with the pathwise tube of (2a):

    E[Var(vᵀx̂ | Π x̂)] ≤ ‖A v‖² E‖d‖² / μ_v²,

with `X = vᵀx̂`, `Y = φ(Π x̂)` (measurable w.r.t. the conditioning σ-algebra), `D = ‖d‖`.
The square-integrability of `D` is an explicit hypothesis — the paper derives it for the Gaussian
data term by objective comparison, a step since closed in `RidgeMarginal.lean`
(`misfit_comparison → expected_grad_sq_lt_top`).

No `sorry`/`admit`; `#print axioms` on every public result at the end.
-/
import MapNearNullTube
import Mathlib.Analysis.Convex.Slope
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Probability.CondVar
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Basic
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Analysis.InnerProductSpace.Projection.Basic

namespace MapNearNullSharp

open RealInnerProductSpace MeasureTheory ProbabilityTheory Filter Set
open scoped Topology NNReal

/-! ### (2a) The kinked tube: 1-D convex analysis from secant slopes -/

/-- Reflecting a convex function through the origin preserves convexity — the change of variables
`t ↦ −t` used to transport the right-hand pinning lemma to the left of `φ`. -/
theorem convexOn_comp_neg {h : ℝ → ℝ} (hh : ConvexOn ℝ Set.univ h) :
    ConvexOn ℝ Set.univ fun t => h (-t) := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  have hkey := hh.2 (Set.mem_univ (-x)) (Set.mem_univ (-y)) ha hb hab
  have hpt : a • (-x) + b • (-y) = -(a • x + b • y) := by
    rw [smul_neg, smul_neg]
    exact (neg_add _ _).symm
  rw [hpt] at hkey
  simpa using hkey

/-- **Right-hand pinning from secant slopes.**  If `t ↦ f t − μ/2 t²` is convex (the paper's
`μ`-strong convexity of the section, kinks allowed), `φ` minimizes `f`, `g` has derivative `g'` at
`t⋆`, and `t⋆` minimizes `f + g`, then for `φ < t⋆` the data tilt must carry the full restoring
force: `μ (t⋆ − φ) ≤ −g'`.  The mirrored secant chain between `φ + t⋆ − t ≤ t < t⋆`, in the limit
`t ↑ t⋆`, recovers the sharp constant `μ`. -/
theorem pin_right {f g : ℝ → ℝ} {mu phi tstar g' : ℝ}
    (hconv : ConvexOn ℝ Set.univ fun t => f t - mu / 2 * t ^ 2)
    (hphi : ∀ s, f phi ≤ f s)
    (hg : HasDerivAt g g' tstar)
    (hmin : ∀ t, f tstar + g tstar ≤ f t + g t)
    (hlt : phi < tstar) :
    mu * (tstar - phi) ≤ -g' := by
  -- the data secant from the left tends to `g'`
  have hslope : Tendsto (fun t => (g t - g tstar) / (t - tstar)) (𝓝[<] tstar) (𝓝 g') := by
    have h1 : Tendsto (slope g tstar) (𝓝[≠] tstar) (𝓝 g') := hasDerivAt_iff_tendsto_slope.mp hg
    have h2 : (𝓝[<] tstar : Filter ℝ) ≤ 𝓝[≠] tstar :=
      nhdsWithin_mono tstar fun y hy => Set.mem_compl_singleton_iff.mpr (ne_of_lt hy)
    have h3 := h1.mono_left h2
    have heq : slope g tstar = fun t => (g t - g tstar) / (t - tstar) := by
      funext t
      rw [slope_def_field]
    rwa [heq] at h3
  have hneg : Tendsto (fun t => -((g t - g tstar) / (t - tstar))) (𝓝[<] tstar) (𝓝 (-g')) :=
    hslope.neg
  have hlin : Tendsto (fun t : ℝ => mu * (t - phi)) (𝓝[<] tstar) (𝓝 (mu * (tstar - phi))) :=
    ((continuous_const.mul (continuous_id.sub continuous_const)).tendsto tstar).mono_left
      nhdsWithin_le_nhds
  refine le_of_tendsto_of_tendsto hlin hneg ?_
  have hmem : Set.Ioo ((phi + tstar) / 2) tstar ∈ 𝓝[<] tstar :=
    Ioo_mem_nhdsLT (by linarith)
  filter_upwards [hmem] with t ht
  obtain ⟨ht1, ht2⟩ := ht
  have hps : phi < phi + tstar - t := by linarith
  have hst : phi + tstar - t < t := by linarith
  -- secants of the shifted section relate to secants of `f`
  have key : ∀ a b : ℝ, a < b →
      ((f b - mu / 2 * b ^ 2) - (f a - mu / 2 * a ^ 2)) / (b - a)
        = (f b - f a) / (b - a) - mu / 2 * (a + b) := by
    intro a b hab
    have h : b - a ≠ 0 := sub_ne_zero.2 hab.ne'
    field_simp
    ring
  -- monotone secants of the convex shifted section along `φ < φ+t⋆−t < t < t⋆`
  have c1 : ((f (phi + tstar - t) - mu / 2 * (phi + tstar - t) ^ 2)
        - (f phi - mu / 2 * phi ^ 2)) / ((phi + tstar - t) - phi)
      ≤ ((f t - mu / 2 * t ^ 2)
        - (f (phi + tstar - t) - mu / 2 * (phi + tstar - t) ^ 2)) / (t - (phi + tstar - t)) :=
    hconv.slope_mono_adjacent (Set.mem_univ _) (Set.mem_univ _) hps hst
  have c2 : ((f t - mu / 2 * t ^ 2)
        - (f (phi + tstar - t) - mu / 2 * (phi + tstar - t) ^ 2)) / (t - (phi + tstar - t))
      ≤ ((f tstar - mu / 2 * tstar ^ 2) - (f t - mu / 2 * t ^ 2)) / (tstar - t) :=
    hconv.slope_mono_adjacent (Set.mem_univ _) (Set.mem_univ _) hst ht2
  have e1 := key phi (phi + tstar - t) hps
  have e2 := key (phi + tstar - t) t hst
  have e3 := key t tstar ht2
  -- `φ` minimizes `f`: its right secant is nonnegative
  have h2 : (0 : ℝ) ≤ (f (phi + tstar - t) - f phi) / ((phi + tstar - t) - phi) :=
    div_nonneg (by have := hphi (phi + tstar - t); linarith) (by linarith)
  -- minimality of `f + g` at `t⋆`: the left secant of `f` is dominated by the data secant
  have h1 : (f tstar - f t) / (tstar - t) ≤ -((g t - g tstar) / (t - tstar)) := by
    have e0 : -((g t - g tstar) / (t - tstar)) = (g t - g tstar) / (tstar - t) := by
      rw [show t - tstar = -(tstar - t) by ring, div_neg, neg_neg]
    rw [e0]
    exact div_le_div_of_nonneg_right (by have := hmin t; linarith) (by linarith)
  linarith [c1, c2, e1, e2, e3, h1, h2]

/-- **Left-hand pinning**, by reflecting `pin_right` through the origin:
for `t⋆ < φ`, `μ (φ − t⋆) ≤ g'`. -/
theorem pin_left {f g : ℝ → ℝ} {mu phi tstar g' : ℝ}
    (hconv : ConvexOn ℝ Set.univ fun t => f t - mu / 2 * t ^ 2)
    (hphi : ∀ s, f phi ≤ f s)
    (hg : HasDerivAt g g' tstar)
    (hmin : ∀ t, f tstar + g tstar ≤ f t + g t)
    (hlt : tstar < phi) :
    mu * (phi - tstar) ≤ g' := by
  -- the reflected shifted section is convex
  have hconvR : ConvexOn ℝ Set.univ fun t => f (-t) - mu / 2 * t ^ 2 := by
    have hcomp := convexOn_comp_neg hconv
    have h2 : (fun t => (fun u => f u - mu / 2 * u ^ 2) (-t))
        = fun t => f (-t) - mu / 2 * t ^ 2 := by
      funext t
      simp
    rwa [h2] at hcomp
  -- reflected minimizer of the penalty
  have hphiR : ∀ s, f (- -phi) ≤ f (-s) := fun s => by
    rw [neg_neg]; exact hphi (-s)
  -- reflected data derivative
  have hgR : HasDerivAt (fun t : ℝ => g (-t)) (-g') (-tstar) := by
    have hneg : HasDerivAt (fun t : ℝ => -t) (-1 : ℝ) (-tstar) := hasDerivAt_neg (-tstar)
    have hg2 : HasDerivAt g g' (- -tstar) := by rw [neg_neg]; exact hg
    have h3 := hg2.comp (-tstar) hneg
    have h4 : g' * -1 = -g' := by ring
    rw [h4] at h3
    exact h3
  -- reflected minimizer of the objective
  have hminR : ∀ u, f (- -tstar) + g (- -tstar) ≤ f (-u) + g (-u) := fun u => by
    simp only [neg_neg]; exact hmin (-u)
  have hltR : -phi < -tstar := neg_lt_neg hlt
  have hkey := pin_right (f := fun t => f (-t)) (g := fun t => g (-t))
    hconvR hphiR hgR hminR hltR
  linarith [hkey]

/-- **`p:mapnearnull` (i)'s 1-D core, differentiability-free.**  If the penalty section `f` is
`μ`-strongly convex in the paper's sense (`t ↦ f t − μ/2 t²` convex — kinks allowed), `φ`
minimizes `f`, the data section `g` is differentiable at the minimizer `t⋆` of `f + g`, then

    |t⋆ − φ| ≤ |g'(t⋆)| / μ.

This is `NearNullFreeze.strongly_convex_pin` with the differentiable-derivative hypothesis
replaced by the subgradient-strength hypothesis the paper actually assumes; the proof runs through
one-sided secant slopes rather than a subdifferential API. -/
theorem strongly_convex_pin_subgradient {f g : ℝ → ℝ} {mu phi tstar g' : ℝ} (hmu : 0 < mu)
    (hconv : ConvexOn ℝ Set.univ fun t => f t - mu / 2 * t ^ 2)
    (hphi : ∀ s, f phi ≤ f s)
    (hg : HasDerivAt g g' tstar)
    (hmin : ∀ t, f tstar + g tstar ≤ f t + g t) :
    |tstar - phi| ≤ |g'| / mu := by
  rcases lt_trichotomy phi tstar with h | h | h
  · have hp := pin_right hconv hphi hg hmin h
    have habs : mu * (tstar - phi) ≤ |g'| := hp.trans (neg_le_abs g')
    rw [abs_of_pos (by linarith : (0 : ℝ) < tstar - phi), le_div_iff₀ hmu]
    linarith
  · rw [h, sub_self, abs_zero]
    positivity
  · have hp := pin_left hconv hphi hg hmin h
    have habs : mu * (phi - tstar) ≤ |g'| := hp.trans (le_abs_self g')
    rw [abs_of_neg (by linarith : tstar - phi < 0), le_div_iff₀ hmu]
    linarith

section Tube

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- **`p:mapnearnull` (i) for kinked penalties — the pinning tube, subgradient form.**
Identical to `MapNearNullTube.pinning_tube` except that the penalty's 1-D section along `v` given
the remaining coordinates `Π x̂ = x̂ − ⟪v, x̂⟫ v` is only assumed `μ_v`-strongly *convex* in the
paper's sense (`u ↦ R(Π x̂ + u v) − μ_v u²/2` convex), not differentiable:

    |⟪v, x̂⟫ − φ| ≤ |⟪A v, d⟫| / μ_v ≤ ‖A v‖ ‖d‖ / μ_v.

This is the hypothesis the paper states — an ℓ¹ or total-variation penalty plus damping
`λ‖x‖²/2` qualifies with `μ_v ≥ λ`, kinks and all.  The archived coordinate is pinned to the
penalty's conditional minimizer within a tube that closes at rate `‖A v‖`. -/
theorem pinning_tube_kinked (A : E →L[ℝ] F) (v xhat : E) (d : F)
    (R : E → ℝ) (Ddat : F → ℝ) (phi mu : ℝ) (hmu : 0 < mu)
    -- `μ_v`-strong convexity of the penalty's section along `v` given `Π x̂` (no differentiability)
    (hconv : ConvexOn ℝ Set.univ
      fun u => R ((xhat - ⟪v, xhat⟫ • v) + u • v) - mu / 2 * u ^ 2)
    -- `φ`: the penalty's conditional minimizer along `v` given the remaining coordinates
    (hphi : ∀ s : ℝ, R ((xhat - ⟪v, xhat⟫ • v) + phi • v)
      ≤ R ((xhat - ⟪v, xhat⟫ • v) + s • v))
    -- the data term is differentiable at `A x̂` with gradient `d` there
    (hd : HasFDerivAt Ddat (innerSL ℝ d) (A xhat))
    -- `x̂` minimizes the objective along the line through `v`
    (hmin : ∀ t : ℝ, Ddat (A xhat) + R xhat ≤ Ddat (A (xhat + t • v)) + R (xhat + t • v)) :
    |⟪v, xhat⟫ - phi| ≤ |⟪A v, d⟫| / mu ∧ |⟪A v, d⟫| / mu ≤ ‖A v‖ * ‖d‖ / mu := by
  constructor
  · -- the data section, reparametrized by the archived coordinate `u = ⟪v, x̂⟫ + t`
    have hdataG : HasDerivAt (fun u : ℝ => Ddat (A (xhat + (u - ⟪v, xhat⟫) • v)))
        ⟪A v, d⟫ ⟪v, xhat⟫ := by
      have h0 : HasDerivAt (fun t : ℝ => Ddat (A (xhat + t • v))) ⟪A v, d⟫ 0 :=
        MapNearNullTube.dataSection_hasDerivAt Ddat A xhat v d hd
      have hshift : HasDerivAt (fun u : ℝ => u - ⟪v, xhat⟫) 1 ⟪v, xhat⟫ :=
        (hasDerivAt_id _).sub_const _
      have h1 : HasDerivAt ((fun t : ℝ => Ddat (A (xhat + t • v))) ∘ fun u : ℝ => u - ⟪v, xhat⟫)
          (⟪A v, d⟫ * 1) ⟪v, xhat⟫ := by
        refine HasDerivAt.comp _ ?_ hshift
        simpa using h0
      rw [mul_one] at h1
      exact h1
    -- minimality of the objective in the `u` coordinate
    have hminG : ∀ u : ℝ,
        R ((xhat - ⟪v, xhat⟫ • v) + ⟪v, xhat⟫ • v)
            + Ddat (A (xhat + (⟪v, xhat⟫ - ⟪v, xhat⟫) • v))
          ≤ R ((xhat - ⟪v, xhat⟫ • v) + u • v) + Ddat (A (xhat + (u - ⟪v, xhat⟫) • v)) := by
      intro u
      have e1 : (xhat - ⟪v, xhat⟫ • v) + ⟪v, xhat⟫ • v = xhat := by abel
      have e2 : xhat + (⟪v, xhat⟫ - ⟪v, xhat⟫) • v = xhat := by
        rw [sub_self, zero_smul, add_zero]
      have e3 : (xhat - ⟪v, xhat⟫ • v) + u • v = xhat + (u - ⟪v, xhat⟫) • v := by
        rw [sub_smul]; abel
      rw [e1, e2, e3]
      have h := hmin (u - ⟪v, xhat⟫)
      linarith
    exact strongly_convex_pin_subgradient
      (f := fun u => R ((xhat - ⟪v, xhat⟫ • v) + u • v))
      (g := fun u => Ddat (A (xhat + (u - ⟪v, xhat⟫) • v)))
      hmu hconv hphi hdataG hminG
  · exact NearNullFreeze.pin_le_norm_mul hmu (A v) d le_rfl

/-- **The kernel endpoint, kinked case.**  On a genuine kernel direction (`A v = 0`) the tube has
zero width even for a kinked penalty: the archived coordinate *is* the penalty's conditional
minimizer — `p:map`'s exact collapse recovered as the endpoint of the graded bound. -/
theorem pinning_exact_of_kernel_kinked (A : E →L[ℝ] F) (v xhat : E) (d : F)
    (R : E → ℝ) (Ddat : F → ℝ) (phi mu : ℝ) (hmu : 0 < mu)
    (hconv : ConvexOn ℝ Set.univ
      fun u => R ((xhat - ⟪v, xhat⟫ • v) + u • v) - mu / 2 * u ^ 2)
    (hphi : ∀ s : ℝ, R ((xhat - ⟪v, xhat⟫ • v) + phi • v)
      ≤ R ((xhat - ⟪v, xhat⟫ • v) + s • v))
    (hd : HasFDerivAt Ddat (innerSL ℝ d) (A xhat))
    (hmin : ∀ t : ℝ, Ddat (A xhat) + R xhat ≤ Ddat (A (xhat + t • v)) + R (xhat + t • v))
    (hker : A v = 0) :
    ⟪v, xhat⟫ = phi := by
  have h := (pinning_tube_kinked A v xhat d R Ddat phi mu hmu hconv hphi hd hmin).1
  rw [hker] at h
  simp only [inner_zero_left, abs_zero, zero_div] at h
  have hnn := abs_nonneg (⟪v, xhat⟫ - phi)
  have hz : |⟪v, xhat⟫ - phi| = 0 := le_antisymm h hnn
  have := abs_eq_zero.mp hz
  linarith

end Tube

/-! ### (2b) The exactly-flat direction: exact minimizer and diverging law -/

/-- **The flat-direction minimizer is exact.**  When the penalty is constant along the direction,
the coordinate's section is the pure data term `t ↦ (y − εt)²/2`, and *any* minimizer of it equals
`y/ε`: the data term alone decides the coordinate, at gain `1/ε`. -/
theorem flat_argmin_eq {eps : ℝ} (heps : eps ≠ 0) (y t : ℝ)
    (hmin : ∀ s : ℝ, (y - eps * t) ^ 2 / 2 ≤ (y - eps * s) ^ 2 / 2) :
    t = y / eps := by
  have hcancel : eps * (y / eps) = y := by field_simp
  have h1 := hmin (y / eps)
  rw [hcancel, sub_self] at h1
  have h3 : (y - eps * t) ^ 2 ≤ 0 := by nlinarith [h1]
  have h4 : y - eps * t = 0 := by
    have h5 : (y - eps * t) ^ 2 = 0 := le_antisymm h3 (sq_nonneg _)
    exact sq_eq_zero_iff.mp h5
  field_simp
  linarith

/-- **The flat coordinate's law.**  The map `y ↦ y/ε` pushes `N(0, σ²)` to `N(0, σ²/ε²)` —
mathlib's `gaussianReal_map_div_const`, recorded at the witness's parameters. -/
theorem flat_coordinate_law (eps : ℝ) (s : ℝ≥0) :
    (gaussianReal 0 s).map (· / eps)
      = gaussianReal 0 (s / NNReal.mk (eps ^ 2) (sq_nonneg eps)) := by
  rw [gaussianReal_map_div_const eps, zero_div]

/-- **Every minimizer selection inherits the diverging law.**  If `T` selects, for each datum `y`,
a minimizer of the flat-direction section `t ↦ (y − εt)²/2`, then `T` *is* `y ↦ y/ε`
(`flat_argmin_eq`), so under `y ∼ N(0, σ²)` the archived flat coordinate has law `N(0, σ²/ε²)`:
the noise amplification is not an artifact of a particular tie-break — it is forced. -/
theorem flat_selection_law {eps : ℝ} (heps : eps ≠ 0) (s : ℝ≥0) (T : ℝ → ℝ)
    (hT : ∀ y t : ℝ, (y - eps * T y) ^ 2 / 2 ≤ (y - eps * t) ^ 2 / 2) :
    (gaussianReal 0 s).map T = gaussianReal 0 (s / NNReal.mk (eps ^ 2) (sq_nonneg eps)) := by
  have hTeq : T = fun y => y / eps := funext fun y => flat_argmin_eq heps y (T y) (hT y)
  rw [hTeq]
  exact flat_coordinate_law eps s

/-- **The noise-amplification variance.**  The archived flat coordinate has variance `σ²/ε²`,
diverging as the illumination `ε` vanishes — the manuscript's
`Var(vᵀx̂) ∼ ‖A v‖⁻²` display, exactly. -/
theorem flat_coordinate_variance (eps : ℝ) (s : ℝ≥0) :
    Var[fun x => x; (gaussianReal 0 s).map (· / eps)] = (s : ℝ) / eps ^ 2 := by
  rw [flat_coordinate_law eps s, variance_fun_id_gaussianReal, NNReal.coe_div, NNReal.coe_mk]

/-- The same variance in operator-norm form: `σ² ‖ε‖⁻²`, with `‖ε‖ = ‖A v‖` for the 1-D witness
operator `A v = ε`. -/
theorem flat_coordinate_variance_norm (eps : ℝ) (s : ℝ≥0) :
    Var[fun x => x; (gaussianReal 0 s).map (· / eps)] = (s : ℝ) * (‖eps‖ ^ 2)⁻¹ := by
  rw [flat_coordinate_variance, Real.norm_eq_abs, sq_abs, div_eq_mul_inv]

/-! ### (2c) The conditional-variance bound: `p:mapnearnull` (ii) -/

section ConditionalVariance

variable {Ω : Type*} {m m₀ : MeasurableSpace Ω} {μ : Measure[m₀] Ω}

set_option maxHeartbeats 1000000 in
/-- **The conditional expectation is the `L²`-optimal measurable predictor.**  For
square-integrable `X` and any `m`-measurable square-integrable predictor `Y`,

    ∫ (X − E[X|m])² dμ ≤ ∫ (X − Y)² dμ.

Proof: in `L²`, `E[X|m]` is the orthogonal projection of `X` onto the subspace `lpMeas` of
`m`-measurable classes (`MemLp.condExpL2_ae_eq_condExp`), and the orthogonal projection minimizes
the distance to the subspace (`Submodule.starProjection_minimal`). -/
theorem integral_sq_sub_condExp_le (hm : m ≤ m₀) [IsFiniteMeasure μ] {X Y : Ω → ℝ}
    (hX : MemLp X 2 μ) (hY : MemLp Y 2 μ) (hYm : AEStronglyMeasurable[m] Y μ) :
    ∫ ω, (X ω - (μ[X | m]) ω) ^ 2 ∂μ ≤ ∫ ω, (X ω - Y ω) ^ 2 ∂μ := by
  haveI : Fact (m ≤ m₀) := ⟨hm⟩
  -- `Y` lands in the subspace of `m`-measurable `L²` classes
  have hyU : MemLp.toLp Y hY ∈ lpMeas ℝ ℝ m 2 μ := by
    rw [mem_lpMeas_iff_aestronglyMeasurable]
    exact hYm.congr hY.coeFn_toLp.symm
  -- the orthogonal projection onto that subspace: distance-minimizing, and a.e. the condExp.
  -- All projection/`condExpL2` definitional work is confined to this one step; `P` is opaque after.
  obtain ⟨P, hPle, hPae⟩ : ∃ P : Lp ℝ 2 μ,
      ‖MemLp.toLp X hX - P‖ ≤ ‖MemLp.toLp X hX - MemLp.toLp Y hY‖ ∧
        (P : Ω → ℝ) =ᵐ[μ] μ[X | m] := by
    refine ⟨(lpMeas ℝ ℝ m 2 μ).starProjection (MemLp.toLp X hX), ?_, ?_⟩
    · rw [Submodule.starProjection_minimal]
      exact ciInf_le ⟨0, Set.forall_mem_range.mpr fun _ => norm_nonneg _⟩
        (⟨MemLp.toLp Y hY, hyU⟩ : lpMeas ℝ ℝ m 2 μ)
    · have h1 : (lpMeas ℝ ℝ m 2 μ).starProjection (MemLp.toLp X hX)
          = ((condExpL2 ℝ ℝ hm (MemLp.toLp X hX) : lpMeas ℝ ℝ m 2 μ) : Lp ℝ 2 μ) := rfl
      rw [h1]
      exact hX.condExpL2_ae_eq_condExp hm
  -- `L²` norms as integrals
  have hnorm : ∀ w : Lp ℝ 2 μ, ∫ ω, (w ω) ^ 2 ∂μ = ‖w‖ ^ 2 := by
    intro w
    calc ∫ ω, (w ω) ^ 2 ∂μ
        = ∫ ω, inner ℝ (w ω) (w ω) ∂μ := by
          refine integral_congr_ae (Eventually.of_forall fun ω => ?_)
          simp [Real.norm_eq_abs, sq_abs]
      _ = inner ℝ w w := (L2.inner_def w w).symm
      _ = ‖w‖ ^ 2 := real_inner_self_eq_norm_sq w
  -- rewrite both integrals through `L²`
  have e1 : ∫ ω, (X ω - (μ[X | m]) ω) ^ 2 ∂μ
      = ∫ ω, ((MemLp.toLp X hX - P : Lp ℝ 2 μ) ω) ^ 2 ∂μ := by
    refine integral_congr_ae ?_
    filter_upwards [Lp.coeFn_sub (MemLp.toLp X hX) P,
      hX.coeFn_toLp, hPae] with ω h1 h2 h3
    rw [h1, Pi.sub_apply, h2, h3]
  have e2 : ∫ ω, (X ω - Y ω) ^ 2 ∂μ
      = ∫ ω, ((MemLp.toLp X hX - MemLp.toLp Y hY : Lp ℝ 2 μ) ω) ^ 2 ∂μ := by
    refine integral_congr_ae ?_
    filter_upwards [Lp.coeFn_sub (MemLp.toLp X hX) (MemLp.toLp Y hY),
      hX.coeFn_toLp, hY.coeFn_toLp] with ω h1 h2 h3
    rw [h1, Pi.sub_apply, h2, h3]
  rw [e1, e2, hnorm, hnorm]
  nlinarith [hPle, norm_nonneg (MemLp.toLp X hX - P),
    norm_nonneg (MemLp.toLp X hX - MemLp.toLp Y hY)]

/-- **The tower property for the conditional variance.**  The expected conditional variance is the
mean squared residual of the conditional mean:
`∫ Var(X|m) dμ = ∫ (X − E[X|m])² dμ` (`integral_condExp` applied to `condVar`'s defining
conditional expectation). -/
theorem integral_condVar_eq (hm : m ≤ m₀) [IsFiniteMeasure μ] (X : Ω → ℝ) :
    ∫ ω, Var[X ; μ | m] ω ∂μ = ∫ ω, (X ω - (μ[X | m]) ω) ^ 2 ∂μ := by
  calc ∫ ω, Var[X ; μ | m] ω ∂μ
      = ∫ ω, (μ[(X - μ[X | m]) ^ 2 | m]) ω ∂μ := rfl
    _ = ∫ ω, ((X - μ[X | m]) ^ 2) ω ∂μ := integral_condExp hm
    _ = ∫ ω, (X ω - (μ[X | m]) ω) ^ 2 ∂μ := by
        refine integral_congr_ae (Eventually.of_forall fun ω => ?_)
        simp [Pi.pow_apply, Pi.sub_apply]

/-- **`p:mapnearnull` (ii), abstract form.**  A pathwise tube `|X − Y| ≤ c·D` around *any*
`m`-measurable predictor `Y` bounds the expected conditional variance:

    ∫ Var(X|m) dμ ≤ c² ∫ D² dμ.

Chains the tower property, the `L²`-optimality of the conditional mean among `m`-measurable
predictors, and the tube.  Square-integrability of `Y` is *derived* from the tube. -/
theorem integral_condVar_le_tube (hm : m ≤ m₀) [IsFiniteMeasure μ]
    {X Y D : Ω → ℝ} {c : ℝ}
    (hX : MemLp X 2 μ) (hYm : AEStronglyMeasurable[m] Y μ) (hD : MemLp D 2 μ)
    (htube : ∀ᵐ ω ∂μ, |X ω - Y ω| ≤ c * D ω) :
    ∫ ω, Var[X ; μ | m] ω ∂μ ≤ c ^ 2 * ∫ ω, (D ω) ^ 2 ∂μ := by
  -- `X − Y` inherits square-integrability from the tube
  have hXY : MemLp (fun ω => X ω - Y ω) 2 μ := by
    refine MemLp.of_le (hD.const_mul c) (hX.aestronglyMeasurable.sub (hYm.mono hm)) ?_
    filter_upwards [htube] with ω hω
    rw [Real.norm_eq_abs, Real.norm_eq_abs]
    exact hω.trans (le_abs_self _)
  have hY : MemLp Y 2 μ := by
    have h1 := hX.sub hXY
    have h2 : (X - fun ω => X ω - Y ω) = Y := by funext ω; simp
    rwa [h2] at h1
  calc ∫ ω, Var[X ; μ | m] ω ∂μ
      = ∫ ω, (X ω - (μ[X | m]) ω) ^ 2 ∂μ := integral_condVar_eq hm X
    _ ≤ ∫ ω, (X ω - Y ω) ^ 2 ∂μ := integral_sq_sub_condExp_le hm hX hY hYm
    _ ≤ ∫ ω, c ^ 2 * (D ω) ^ 2 ∂μ := by
        refine integral_mono_ae hXY.integrable_sq (hD.integrable_sq.const_mul _) ?_
        filter_upwards [htube] with ω hω
        calc (X ω - Y ω) ^ 2 = |X ω - Y ω| ^ 2 := (sq_abs _).symm
          _ ≤ (c * D ω) ^ 2 := by nlinarith [hω, abs_nonneg (X ω - Y ω)]
          _ = c ^ 2 * (D ω) ^ 2 := by ring
    _ = c ^ 2 * ∫ ω, (D ω) ^ 2 ∂μ := integral_const_mul _ _

variable {G : Type*} [NormedAddCommGroup G] [InnerProductSpace ℝ G]

omit [InnerProductSpace ℝ G] in
/-- **`p:mapnearnull` (ii) in the paper's display.**  With `X = vᵀx̂`, `Y = φ(Π x̂)`
(`m` = the σ-algebra of the remaining coordinates `Π x̂`), `D = ‖d‖` the data-gradient norm, and
the pathwise tube of `pinning_tube_kinked`, the expected conditional variance of the archived
blind coordinate obeys

    E[Var(vᵀx̂ | Π x̂)] ≤ ‖A v‖² E‖d‖² / μ_v²,

vanishing quadratically with the illumination.  `E‖d‖² < ∞` is the explicit hypothesis `hD`; the
paper's objective-comparison argument for it (Gaussian data term, penalty bounded below) is not
formalized. -/
theorem condVar_tube_paper (hm : m ≤ m₀) [IsFiniteMeasure μ]
    {X Y D : Ω → ℝ} (Av : G) {mu : ℝ} (_hmu : 0 < mu)
    (hX : MemLp X 2 μ) (hYm : AEStronglyMeasurable[m] Y μ) (hD : MemLp D 2 μ)
    (htube : ∀ᵐ ω ∂μ, |X ω - Y ω| ≤ ‖Av‖ * D ω / mu) :
    ∫ ω, Var[X ; μ | m] ω ∂μ ≤ ‖Av‖ ^ 2 * (∫ ω, (D ω) ^ 2 ∂μ) / mu ^ 2 := by
  have h1 : ∀ᵐ ω ∂μ, |X ω - Y ω| ≤ ‖Av‖ / mu * D ω := by
    filter_upwards [htube] with ω hω
    calc |X ω - Y ω| ≤ ‖Av‖ * D ω / mu := hω
      _ = ‖Av‖ / mu * D ω := by ring
  calc ∫ ω, Var[X ; μ | m] ω ∂μ
      ≤ (‖Av‖ / mu) ^ 2 * ∫ ω, (D ω) ^ 2 ∂μ := integral_condVar_le_tube hm hX hYm hD h1
    _ = ‖Av‖ ^ 2 * (∫ ω, (D ω) ^ 2 ∂μ) / mu ^ 2 := by
        rw [div_pow]; ring

end ConditionalVariance

end MapNearNullSharp

-- #print axioms audit (p:mapnearnull: kinked tube, flat witness, conditional-variance bound)
#print axioms MapNearNullSharp.convexOn_comp_neg
#print axioms MapNearNullSharp.pin_right
#print axioms MapNearNullSharp.pin_left
#print axioms MapNearNullSharp.strongly_convex_pin_subgradient
#print axioms MapNearNullSharp.pinning_tube_kinked
#print axioms MapNearNullSharp.pinning_exact_of_kernel_kinked
#print axioms MapNearNullSharp.flat_argmin_eq
#print axioms MapNearNullSharp.flat_coordinate_law
#print axioms MapNearNullSharp.flat_selection_law
#print axioms MapNearNullSharp.flat_coordinate_variance
#print axioms MapNearNullSharp.flat_coordinate_variance_norm
#print axioms MapNearNullSharp.integral_sq_sub_condExp_le
#print axioms MapNearNullSharp.integral_condVar_eq
#print axioms MapNearNullSharp.integral_condVar_le_tube
#print axioms MapNearNullSharp.condVar_tube_paper
