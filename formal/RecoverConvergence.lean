/-
# `p:recover` (ii) — the correction loop converges on the resolved subspace

Part (i) of `p:recover` (the blind-fiber freeze, for any likelihood factoring through `A`) is
machine-checked in `BlindFreeze.lean` (`freeze_iterate`).  This file supplies part (ii), the
linear--Gaussian **convergence** half.

After whitening by `Γ^{-1/2}` the covariance step of the loop projects to an autonomous recursion in
the data-space covariance `D̃ₜ = Γ^{-1/2} A Σₜ Aᵀ Γ^{-1/2}`,

    D̃ₜ₊₁ = Ψ(D̃ₜ) := K̃ₜ + K̃ₜ S̃ K̃ₜ,    K̃ₜ = D̃ₜ (D̃ₜ + I)⁻¹,    S̃ = I + D̃⋆,

and the mean step to `A μₜ₊₁ − A μ⋆ = (I − Kₜ)(A μₜ − A μ⋆)`.

What is proved here, in the simultaneously-diagonalized (equivalently scalar) case — the form the
appendix's whitening puts the recursion in on each eigendirection:

* `psi_fixed`          : the truth's resolved covariance is a **fixed point** of the recursion, so
                         the loop targets the truth on the resolved subspace.
* `psi_pos`            : the recursion keeps a positive iterate positive (the interior floor).
* `psi_monotone`       : `Ψ` is monotone in the iterate.
* `psi_sub_self_sign`  : `Ψ(d) − d` has the sign of `d⋆ − d` (the exact identity is
                         `Ψ(d) − d = d²(d⋆ − d)/(d+1)²`), so the iterate always steps *toward* the
                         fixed point and never past it.
* `one_sub_K_le`       : a floor `d ≥ λ > 0` gives `|1 − k| ≤ (1+λ)⁻¹ < 1`, the appendix's
                         `‖I − K̃ₜ‖ ≤ (1+λ♭)⁻¹`.
* `mean_error_le`      : hence `|eₜ| ≤ (1+λ)⁻ᵗ |e₀|` — the mean error decays geometrically.
* `tendsto_mean_error` : and therefore vanishes.

**Scope, stated precisely.**  Both halves of `p:recover` (ii) are now assembled here in the
diagonalized case: the resolved *mean* error decays geometrically (`mean_error_le`,
`tendsto_mean_error`) and the resolved *covariance* converges to the truth's, also geometrically
(`covariance_error_le`, `tendsto_covariance`), the rate coming from the explicit factor
`q d = (2d+1)/(d+1)²` together with the invariance of `[min d₀ d⋆, max d₀ d⋆]`, which supplies the
permanent floor.  What is not treated here is the reduction of the non-commuting matrix recursion to
this diagonalized form, which is the appendix's whitening argument.

No `sorry`/`admit`; `#print axioms` on each result at the end.
-/
import Mathlib.Order.Filter.AtTopBot.Basic
import Mathlib.Topology.Order.MonotoneConvergence
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.SpecificLimits.Normed

open Filter
open scoped Topology

namespace Recover

/-- The gain `k = d/(d+1)` of one loop round at data-space covariance `d`. -/
noncomputable def K (d : ℝ) : ℝ := d / (d + 1)

/-- The whitened covariance recursion `Ψ(d) = k + s·k²`, `k = d/(d+1)`, with `s = 1 + d⋆`. -/
noncomputable def Psi (s d : ℝ) : ℝ := K d + s * (K d) ^ 2

lemma K_nonneg {d : ℝ} (hd : 0 ≤ d) : 0 ≤ K d := by
  have h1 : (0:ℝ) < d + 1 := by linarith
  exact div_nonneg hd h1.le

lemma K_pos {d : ℝ} (hd : 0 < d) : 0 < K d := by
  have h1 : (0:ℝ) < d + 1 := by linarith
  exact div_pos hd h1

lemma K_lt_one {d : ℝ} (hd : 0 ≤ d) : K d < 1 := by
  have h1 : (0:ℝ) < d + 1 := by linarith
  rw [K, div_lt_one h1]; linarith

/-- **The truth's resolved covariance is a fixed point of the loop.** With `s = 1 + d⋆`, the
recursion `Ψ` fixes `d⋆`: the loop targets the truth on the resolved subspace. -/
theorem psi_fixed {dstar : ℝ} (h : 0 < dstar) : Psi (1 + dstar) dstar = dstar := by
  have h1 : dstar + 1 ≠ 0 := by positivity
  simp only [Psi, K]
  field_simp
  ring

/-- The recursion keeps a strictly positive iterate strictly positive (the interior floor). -/
theorem psi_pos {s d : ℝ} (hs : 0 ≤ s) (hd : 0 < d) : 0 < Psi s d := by
  have hk := K_pos hd
  have : 0 ≤ s * (K d) ^ 2 := by positivity
  simpa [Psi] using lt_of_lt_of_le hk (by linarith)

/-- `Ψ` is monotone in the iterate. -/
theorem psi_monotone {s : ℝ} (hs : 0 ≤ s) : MonotoneOn (Psi s) (Set.Ici 0) := by
  intro a ha b hb hab
  simp only [Set.mem_Ici] at ha hb
  have hKmono : K a ≤ K b := by
    have h1 : (0:ℝ) < a + 1 := by linarith
    have h2 : (0:ℝ) < b + 1 := by linarith
    have hdiff : K b - K a = (b - a) / ((a + 1) * (b + 1)) := by
      simp only [K]; field_simp; ring
    have : 0 ≤ K b - K a := by rw [hdiff]; positivity
    linarith
  have hKa : 0 ≤ K a := K_nonneg ha
  have : s * (K a) ^ 2 ≤ s * (K b) ^ 2 := by
    apply mul_le_mul_of_nonneg_left _ hs
    exact pow_le_pow_left₀ hKa hKmono 2
  simp only [Psi]; linarith

/-- **Below the fixed point the recursion moves up; above it, down.**  With `s = 1 + d⋆`, the sign of
`Ψ(d) − d` is the sign of `d⋆ − d`, so the iterate always steps toward `d⋆` and never past it. -/
theorem psi_sub_self_sign {dstar d : ℝ} (hstar : 0 < dstar) (hd : 0 < d) :
    (Psi (1 + dstar) d - d) * (dstar - d) ≥ 0 := by
  have h1 : (0:ℝ) < d + 1 := by linarith
  have key : Psi (1 + dstar) d - d = d ^ 2 * (dstar - d) / (d + 1) ^ 2 := by
    simp only [Psi, K]; field_simp; ring
  rw [key]
  have hgoal : d ^ 2 * (dstar - d) / (d + 1) ^ 2 * (dstar - d)
      = d ^ 2 * (dstar - d) ^ 2 / (d + 1) ^ 2 := by ring
  rw [hgoal]
  positivity

/-- The contraction factor of one round at covariance `d`. -/
noncomputable def q (d : ℝ) : ℝ := (2 * d + 1) / (d + 1) ^ 2

/-- **The covariance error is multiplied by an explicit factor each round:**
`Ψ(d) − d⋆ = (d − d⋆) · q d`, with `q d = (2d+1)/(d+1)²`. -/
theorem psi_sub_dstar {dstar d : ℝ} (hd : 0 < d) :
    Psi (1 + dstar) d - dstar = (d - dstar) * q d := by
  have h1 : (0:ℝ) < d + 1 := by linarith
  simp only [Psi, K, q]; field_simp; ring

/-- The factor is `< 1` at any strictly positive covariance: `(d+1)² > 2d+1 ⟺ d ≠ 0`. -/
theorem q_lt_one {d : ℝ} (hd : 0 < d) : q d < 1 := by
  have h1 : (0:ℝ) < (d + 1) ^ 2 := by positivity
  rw [q, div_lt_one h1]; nlinarith

theorem q_nonneg {d : ℝ} (hd : 0 ≤ d) : 0 ≤ q d := by
  have h1 : (0:ℝ) < (d + 1) ^ 2 := by positivity
  apply div_nonneg _ h1.le; linarith

/-- The factor is decreasing, so a floor on the covariance caps it: `λ ≤ d → q d ≤ q λ`. -/
theorem q_le_of_le {lam d : ℝ} (hlam : 0 ≤ lam) (h : lam ≤ d) : q d ≤ q lam := by
  have hd0 : 0 ≤ d := le_trans hlam h
  have hl1 : (0:ℝ) < lam + 1 := by linarith
  have hd1 : (0:ℝ) < d + 1 := by linarith
  have key : q lam - q d
      = ((d - lam) * (2 * lam * d + lam + d)) / ((lam + 1) ^ 2 * (d + 1) ^ 2) := by
    simp only [q]; field_simp; ring
  have hnum : 0 ≤ (d - lam) * (2 * lam * d + lam + d) := by
    apply mul_nonneg (by linarith)
    nlinarith
  have hden : (0:ℝ) < (lam + 1) ^ 2 * (d + 1) ^ 2 := by positivity
  have hge : 0 ≤ q lam - q d := by rw [key]; exact div_nonneg hnum hden.le
  linarith

/-- **The interval between the start and the fixed point is invariant.**  Starting below `d⋆` the
iterate stays in `[d₀, d⋆]`; starting above, in `[d⋆, d₀]`.  Either way it never leaves, so
`min d₀ d⋆` is a permanent floor. -/
theorem psi_mem_interval {dstar d₀ d : ℝ} (hstar : 0 < dstar) (h0 : 0 < d₀)
    (hlo : min d₀ dstar ≤ d) (hhi : d ≤ max d₀ dstar) :
    min d₀ dstar ≤ Psi (1 + dstar) d ∧ Psi (1 + dstar) d ≤ max d₀ dstar := by
  have hdpos : 0 < d := lt_of_lt_of_le (lt_min h0 hstar) hlo
  have hkey : Psi (1 + dstar) d - dstar = (d - dstar) * q d := psi_sub_dstar hdpos
  have hq0 : 0 ≤ q d := q_nonneg hdpos.le
  have hq1 : q d ≤ 1 := (q_lt_one hdpos).le
  rcases le_total d dstar with h | h
  · -- below the fixed point: the step increases and stays under d⋆
    have hup : d ≤ Psi (1 + dstar) d := by
      have := psi_sub_self_sign hstar hdpos
      nlinarith [this, sub_nonneg.mpr h]
    have hunder : Psi (1 + dstar) d ≤ dstar := by nlinarith [mul_nonneg (sub_nonneg.mpr h) hq0]
    exact ⟨le_trans hlo hup, le_trans hunder (le_max_right _ _)⟩
  · -- above the fixed point: the step decreases and stays over d⋆
    have hdown : Psi (1 + dstar) d ≤ d := by
      have := psi_sub_self_sign hstar hdpos
      nlinarith [this, sub_nonneg.mpr h]
    have hover : dstar ≤ Psi (1 + dstar) d := by nlinarith [mul_nonneg (sub_nonneg.mpr h) hq0]
    exact ⟨le_trans (min_le_right _ _) hover, le_trans hdown hhi⟩

/-- **`p:recover` (ii), the covariance half: the iterate converges to the truth's resolved
covariance, geometrically.**  With `λ = min d₀ d⋆ > 0` a permanent floor and `q λ < 1` the resulting
cap on the per-round factor, `|dₜ − d⋆| ≤ (q λ)ᵗ |d₀ − d⋆|`. -/
theorem covariance_error_le {dstar d₀ : ℝ} (hstar : 0 < dstar) (h0 : 0 < d₀)
    (d : ℕ → ℝ) (hd0 : d 0 = d₀) (hrec : ∀ t, d (t + 1) = Psi (1 + dstar) (d t)) :
    ∀ t, |d t - dstar| ≤ (q (min d₀ dstar)) ^ t * |d₀ - dstar| := by
  have hlam : 0 < min d₀ dstar := lt_min h0 hstar
  -- the invariant interval, by induction
  have hinv : ∀ t, min d₀ dstar ≤ d t ∧ d t ≤ max d₀ dstar := by
    intro t; induction t with
    | zero => rw [hd0]; exact ⟨min_le_left _ _, le_max_left _ _⟩
    | succ n ih => rw [hrec n]; exact psi_mem_interval hstar h0 ih.1 ih.2
  intro t
  induction t with
  | zero => simp [hd0]
  | succ n ih =>
      have hdn : 0 < d n := lt_of_lt_of_le hlam (hinv n).1
      have hstep : |d (n + 1) - dstar| = |d n - dstar| * q (d n) := by
        rw [hrec n, psi_sub_dstar hdn, abs_mul, abs_of_nonneg (q_nonneg hdn.le)]
      have hcap : q (d n) ≤ q (min d₀ dstar) := q_le_of_le hlam.le (hinv n).1
      have hqpos : 0 ≤ q (min d₀ dstar) := q_nonneg hlam.le
      calc |d (n + 1) - dstar| = |d n - dstar| * q (d n) := hstep
        _ ≤ |d n - dstar| * q (min d₀ dstar) := by
              exact mul_le_mul_of_nonneg_left hcap (abs_nonneg _)
        _ ≤ ((q (min d₀ dstar)) ^ n * |d₀ - dstar|) * q (min d₀ dstar) := by
              exact mul_le_mul_of_nonneg_right ih hqpos
        _ = (q (min d₀ dstar)) ^ (n + 1) * |d₀ - dstar| := by ring

/-- **And therefore the covariance converges**: `dₜ → d⋆`.  This closes the covariance half of
`p:recover` (ii). -/
theorem tendsto_covariance {dstar d₀ : ℝ} (hstar : 0 < dstar) (h0 : 0 < d₀)
    (d : ℕ → ℝ) (hd0 : d 0 = d₀) (hrec : ∀ t, d (t + 1) = Psi (1 + dstar) (d t)) :
    Tendsto (fun t => |d t - dstar|) atTop (𝓝 0) := by
  have hlam : 0 < min d₀ dstar := lt_min h0 hstar
  have hq : |q (min d₀ dstar)| < 1 := by
    rw [abs_of_nonneg (q_nonneg hlam.le)]; exact q_lt_one hlam
  have hgeo : Tendsto (fun t => (q (min d₀ dstar)) ^ t * |d₀ - dstar|) atTop (𝓝 (0 * |d₀ - dstar|)) :=
    (tendsto_pow_atTop_nhds_zero_of_abs_lt_one hq).mul_const _
  rw [zero_mul] at hgeo
  exact squeeze_zero (fun t => abs_nonneg _)
    (fun t => covariance_error_le hstar h0 d hd0 hrec t) hgeo

/-- **The mean error contracts geometrically.**  The loop's mean step is
`eₜ₊₁ = (1 − kₜ) eₜ` with `kₜ = dₜ/(dₜ+1)`, so a floor `dₜ ≥ λ > 0` gives `|1 − kₜ| ≤ (1+λ)⁻¹ < 1`
and hence `|eₜ| ≤ (1+λ)⁻ᵗ |e₀|`.  This is the appendix's `‖I − K̃ₜ‖ ≤ (1+λ♭)⁻¹`. -/
theorem one_sub_K_le {d lam : ℝ} (hlam : 0 < lam) (hd : lam ≤ d) :
    |1 - K d| ≤ (1 + lam)⁻¹ := by
  have hd0 : 0 < d := lt_of_lt_of_le hlam hd
  have h1 : (0:ℝ) < d + 1 := by linarith
  have hK : 1 - K d = 1 / (d + 1) := by simp only [K]; field_simp; ring
  rw [hK, abs_of_pos (by positivity)]
  rw [one_div, inv_le_inv₀ h1 (by linarith)]
  linarith

/-- The geometric bound iterated: with a uniform floor `λ`, the mean error after `t` rounds is at
most `(1+λ)⁻ᵗ` times its initial size. -/
theorem mean_error_le {lam : ℝ} (hlam : 0 < lam) (d : ℕ → ℝ) (e : ℕ → ℝ)
    (hfloor : ∀ t, lam ≤ d t) (hrec : ∀ t, e (t + 1) = (1 - K (d t)) * e t) :
    ∀ t, |e t| ≤ ((1 + lam)⁻¹) ^ t * |e 0| := by
  intro t
  induction t with
  | zero => simp
  | succ n ih =>
      have hstep : |e (n + 1)| = |1 - K (d n)| * |e n| := by
        rw [hrec n, abs_mul]
      have hbound : |1 - K (d n)| ≤ (1 + lam)⁻¹ := one_sub_K_le hlam (hfloor n)
      have hpos : (0:ℝ) ≤ |e n| := abs_nonneg _
      calc |e (n + 1)| = |1 - K (d n)| * |e n| := hstep
        _ ≤ (1 + lam)⁻¹ * |e n| := by exact mul_le_mul_of_nonneg_right hbound hpos
        _ ≤ (1 + lam)⁻¹ * (((1 + lam)⁻¹) ^ n * |e 0|) := by
              apply mul_le_mul_of_nonneg_left ih
              positivity
        _ = ((1 + lam)⁻¹) ^ (n + 1) * |e 0| := by ring

/-- **The mean error vanishes.**  Under a uniform floor the geometric bound drives the resolved mean
error to zero, which is the paper's "the mean error decaying geometrically". -/
theorem tendsto_mean_error {lam : ℝ} (hlam : 0 < lam) (d : ℕ → ℝ) (e : ℕ → ℝ)
    (hfloor : ∀ t, lam ≤ d t) (hrec : ∀ t, e (t + 1) = (1 - K (d t)) * e t) :
    Tendsto (fun t => |e t|) atTop (𝓝 0) := by
  have hr : |(1 + lam)⁻¹| < 1 := by
    rw [abs_of_pos (by positivity), inv_lt_one_iff₀]
    right; linarith
  have hgeo : Tendsto (fun t => ((1 + lam)⁻¹) ^ t * |e 0|) atTop (𝓝 (0 * |e 0|)) :=
    (tendsto_pow_atTop_nhds_zero_of_abs_lt_one hr).mul_const _
  rw [zero_mul] at hgeo
  refine squeeze_zero (fun t => abs_nonneg _) (fun t => mean_error_le hlam d e hfloor hrec t) hgeo

end Recover

-- #print axioms audit (p:recover (ii))
#print axioms Recover.psi_fixed
#print axioms Recover.psi_pos
#print axioms Recover.psi_monotone
#print axioms Recover.psi_sub_self_sign
#print axioms Recover.one_sub_K_le
#print axioms Recover.mean_error_le
#print axioms Recover.tendsto_mean_error
#print axioms Recover.psi_sub_dstar
#print axioms Recover.q_lt_one
#print axioms Recover.q_le_of_le
#print axioms Recover.psi_mem_interval
#print axioms Recover.covariance_error_le
#print axioms Recover.tendsto_covariance
