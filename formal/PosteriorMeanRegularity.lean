import Mathlib
import PosteriorMean

/-!
# Exp-family regularity for `p:mean`, machine-checked: a strictly convex log-partition has an
injective gradient

`PosteriorMean.posterior_mean_collapse` takes the injectivity of the statistic→resolved map
`ρ = ∇Λ` as an explicit hypothesis — the one analytic input the paper *derives* from the strict
convexity of the exponential-family log-partition `Λ` (`∇²Λ = Cov(x_R ∣ τ) ≻ 0`, so `Λ` is strictly
convex). This file closes that gap on any real inner product space: the gradient of a strictly convex
function is strictly monotone, hence injective. `p:mean`'s injectivity then follows from the honest
primitive — strict convexity of `Λ` — with no `sorry`.

## Results
* `injective_of_strictMono_gradient` — a strictly monotone gradient (`0 < ⟪g y - g x, y - x⟫` for
  `x ≠ y`, the operational form of `∇²Λ ≻ 0`) is injective.
* `strictMono_inner_gradient_of_strictConvexOn` — the gradient of a strictly convex function is
  strictly monotone (restriction to the segment + mathlib's 1D `StrictConvexOn.strictMonoOn_deriv`).
* `gradient_injective_of_strictConvexOn` — **the missing piece**: strict convexity ⟹ injective
  gradient.
* `posterior_mean_collapse_of_strictConvex` — `p:mean`'s structural collapse with the injectivity
  hypothesis replaced by strict convexity of the log-partition `Λ`.

## Honesty
No `sorry`/`admit`; `#print axioms` on each result lists only `propext`, `Classical.choice`,
`Quot.sound`.
-/

open scoped InnerProductSpace

namespace PosteriorMean

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **A strictly monotone gradient is injective.** If `⟪g y - g x, y - x⟫ > 0` whenever `x ≠ y`
(the operational content of a positive-definite Hessian `∇²Λ = Cov ≻ 0`), then `g` is injective:
`g x = g y` forces the inner product to vanish, contradicting strict positivity. -/
theorem injective_of_strictMono_gradient {g : E → E}
    (h : ∀ ⦃x y : E⦄, x ≠ y → 0 < ⟪g y - g x, y - x⟫_ℝ) : Function.Injective g := by
  intro x y hxy
  by_contra hne
  have hpos := h hne
  rw [hxy, sub_self, inner_zero_left] at hpos
  exact lt_irrefl 0 hpos

/-- **The gradient of a strictly convex function is strictly monotone.**
For `x ≠ y`, `0 < ⟪g y - g x, y - x⟫`, where `g` is a gradient of the strictly convex `f`.
Proof: restrict `f` to the segment `t ↦ x + t • (y - x)`; the restriction is a strictly convex `ℝ → ℝ`
whose derivative at `t` is `⟪g(x + t•(y-x)), y - x⟫`; by mathlib's 1D `strictMonoOn_deriv` this
derivative is strictly increasing, so its value at `0` (`⟪g x, y-x⟫`) is below its value at `1`
(`⟪g y, y-x⟫`). -/
theorem strictMono_inner_gradient_of_strictConvexOn [CompleteSpace E] {f : E → ℝ} {g : E → E}
    (hconv : StrictConvexOn ℝ Set.univ f) (hg : ∀ x, HasGradientAt f (g x) x)
    ⦃x y : E⦄ (hxy : x ≠ y) : 0 < ⟪g y - g x, y - x⟫_ℝ := by
  -- the segment γ t = x + t • (y - x), a plain function
  set v : E := y - x with hv
  have hv0 : v ≠ 0 := sub_ne_zero.mpr (Ne.symm hxy)
  set γ : ℝ → E := fun t => x + t • v with hγ
  have hγ0 : γ 0 = x := by simp [hγ]
  have hγ1 : γ 1 = y := by simp [hγ, hv]
  -- γ is injective (v ≠ 0)
  have hγinj : Function.Injective γ := by
    intro a b hab
    simp only [hγ, add_right_inj] at hab
    exact smul_left_injective ℝ hv0 hab
  -- affine-combination identity: γ (p•a+q•b) = p•γa+q•γb when p+q=1
  have hγaff : ∀ (a b p q : ℝ), p + q = 1 → γ (p • a + q • b) = p • γ a + q • γ b := by
    intro a b p q hpq
    have hq : q = 1 - p := by linarith
    subst hq
    simp only [hγ, smul_eq_mul]
    match_scalars <;> ring
  -- the restriction φ = f ∘ γ is strictly convex on ℝ
  have hφconv : StrictConvexOn ℝ Set.univ (fun t => f (γ t)) := by
    refine ⟨convex_univ, ?_⟩
    intro a _ b _ hab p q hp hq hpq
    have hne : γ a ≠ γ b := fun h => hab (hγinj h)
    have := hconv.2 (Set.mem_univ (γ a)) (Set.mem_univ (γ b)) hne hp hq hpq
    calc f (γ (p • a + q • b)) = f (p • γ a + q • γ b) := by rw [hγaff a b p q hpq]
      _ < p • f (γ a) + q • f (γ b) := this
  -- derivative of γ is v
  have hγderiv : ∀ t, HasDerivAt γ v t := by
    intro t
    simpa [hγ] using (((hasDerivAt_id t).smul_const v).const_add x)
  -- φ has derivative ⟪g (γ t), v⟫ at t
  have hφderiv : ∀ t, HasDerivAt (fun s => f (γ s)) (⟪g (γ t), v⟫_ℝ) t := by
    intro t
    have h2 := ((hg (γ t)).hasFDerivAt).comp_hasDerivAt t (hγderiv t)
    simpa [Function.comp_def, InnerProductSpace.toDual_apply_apply] using h2
  have hφdiff : ∀ t, DifferentiableAt ℝ (fun s => f (γ s)) t := fun t => (hφderiv t).differentiableAt
  -- 1D: strictly convex derivative is strictly monotone
  have hmono : StrictMonoOn (deriv (fun s => f (γ s))) Set.univ :=
    hφconv.strictMonoOn_deriv (fun t _ => hφdiff t)
  have h01 : deriv (fun s => f (γ s)) 0 < deriv (fun s => f (γ s)) 1 :=
    hmono (Set.mem_univ 0) (Set.mem_univ 1) (by norm_num)
  rw [(hφderiv 0).deriv, (hφderiv 1).deriv, hγ0, hγ1] at h01
  -- ⟪g x, v⟫ < ⟪g y, v⟫  ⟹  0 < ⟪g y - g x, v⟫
  rw [inner_sub_left]
  linarith

/-- **The missing analytic input of `p:mean`, machine-checked: a strictly convex log-partition has an
injective gradient.** `ρ = ∇Λ` is injective because `Λ` is strictly convex (a minimal exponential
family has `∇²Λ = Cov ≻ 0`). This discharges the `Function.Injective ρ` hypothesis of
`posterior_mean_collapse` from the honest primitive. -/
theorem gradient_injective_of_strictConvexOn [CompleteSpace E] {f : E → ℝ} {g : E → E}
    (hconv : StrictConvexOn ℝ Set.univ f) (hg : ∀ x, HasGradientAt f (g x) x) :
    Function.Injective g :=
  injective_of_strictMono_gradient
    (fun _ _ hxy => strictMono_inner_gradient_of_strictConvexOn hconv hg hxy)

/-! ## Wiring into `p:mean`: the collapse from strict convexity of the log-partition -/

/-- **`p:mean` structural collapse, with the injectivity input replaced by strict convexity of the
log-partition.** If the reconstruction factors through the resolved statistic `T` (`xhat = ψ ∘ T`) and
its resolved component is `PR ∘ xhat = (∇Λ) ∘ T` for a strictly convex `Λ` on the statistic space,
then two measurements with the same resolved reconstruction give the same reconstruction — the
blind-fiber conditional is a Dirac. The injectivity `posterior_mean_collapse` required is now a
theorem (`gradient_injective_of_strictConvexOn`), not a hypothesis. -/
theorem posterior_mean_collapse_of_strictConvex
    {𝓨 X : Type*} {S : Type*} [NormedAddCommGroup S] [InnerProductSpace ℝ S] [CompleteSpace S]
    (xhat : 𝓨 → X) (T : 𝓨 → S) (ψ : S → X) (PR : X → S) (Λ : S → S) {Λfun : S → ℝ}
    (hfac : ∀ y, xhat y = ψ (T y)) (hres : ∀ y, PR (xhat y) = Λ (T y))
    (hconv : StrictConvexOn ℝ Set.univ Λfun) (hΛ : ∀ s, HasGradientAt Λfun (Λ s) s)
    {y y' : 𝓨} (h : PR (xhat y) = PR (xhat y')) :
    xhat y = xhat y' :=
  posterior_mean_collapse xhat T ψ PR Λ hfac hres
    (gradient_injective_of_strictConvexOn hconv hΛ) h

end PosteriorMean

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms PosteriorMean.injective_of_strictMono_gradient
#print axioms PosteriorMean.strictMono_inner_gradient_of_strictConvexOn
#print axioms PosteriorMean.gradient_injective_of_strictConvexOn
#print axioms PosteriorMean.posterior_mean_collapse_of_strictConvex
