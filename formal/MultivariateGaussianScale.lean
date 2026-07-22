import Mathlib
import ChiSquaredTV
import ChiSquaredInvariance
import ChiSquaredPi
import GaussianChiSquaredVar

/-!
# Multivariate isotropic Gaussian χ² under diagonal scaling

The χ² of the standard Gaussian on `EuclideanSpace ℝ (Fin n)` against its pushforward by the
diagonal scaling `x ↦ (cᵢ · xᵢ)ᵢ` is the product `∏ᵢ (cᵢ²/√(2cᵢ²−1)) − 1` — the identity-covariance
**spread-misreport** χ². This is the scaling analogue of `MultivariateGaussianChiSquared.lean`'s
mean-shift `chi2_stdGaussian_add`.

The route mirrors the mean-shift file exactly: the isotropic reduction (`toLp` invariance
`NearNull.chi2_map_measurableEquiv` + coordinatewise-scale `Measure.pi_map_pi`) to a product of scaled
1-D Gaussians, evaluated by `NearNull.chi2_pi` (tensorization) and the same-mean mismatched-variance
1-D Gaussian χ² `chi2_gaussianReal_var`. Coordinate `i` scaled by `cᵢ` becomes `N(0,cᵢ²)`, whose χ²
against `N(0,1)` contributes the `1 + χ² = cᵢ²/√(2cᵢ²−1)` factor (finite exactly when `1 < 2cᵢ²`).

## Results
* `NearNull.chi2_stdGaussian_scale` — `χ²(N₀ ‖ N₀∘scale c) = (∏ᵢ cᵢ²/√(2cᵢ²−1)) − 1`.
* `NearNull.chi2_stdGaussian_scale_single` — the single-coordinate specialization (scale only
  coordinate `0` by `s`): the product telescopes to `s²/√(2s²−1) − 1`, the whitening-step χ².

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each result lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure ProbabilityTheory
open scoped NNReal

namespace NearNull

/-- **Finiteness of the squared 1-D Gaussian density** (spread-mismatch case; feeds `chi2_pi`).
Derived from the near-null spread χ² finiteness `integrable_chi2_gaussianReal_var` (giving `(r−1)²`
integrable) and `Measure.integrable_toReal_rnDeriv` (giving `r` integrable), via `r² = (r−1)² + 2r − 1`. -/
private lemma integrable_rnDeriv_sq_gaussianReal_var (m : ℝ) {v₁ v₂ : ℝ≥0}
    (hv₁ : v₁ ≠ 0) (hv₂ : v₂ ≠ 0) (hlt : (v₁ : ℝ) < 2 * v₂) :
    Integrable
      (fun x => (((gaussianReal m v₁).rnDeriv (gaussianReal m v₂) x).toReal) ^ 2)
      (gaussianReal m v₂) := by
  have h1 := integrable_chi2_gaussianReal_var m hv₁ hv₂ hlt
  have h2 : Integrable
      (fun x => ((gaussianReal m v₁).rnDeriv (gaussianReal m v₂) x).toReal)
      (gaussianReal m v₂) := Measure.integrable_toReal_rnDeriv
  refine ((h1.add (h2.const_mul 2)).sub (integrable_const (1 : ℝ))).congr ?_
  filter_upwards with x
  simp only [Pi.add_apply, Pi.sub_apply]
  ring

/-- **Isotropic multivariate Gaussian χ² under diagonal scaling.** The χ² of the standard Gaussian on
`EuclideanSpace ℝ (Fin n)` against its pushforward by the diagonal scaling `x ↦ (cᵢ · xᵢ)ᵢ` equals
`(∏ᵢ cᵢ²/√(2cᵢ²−1)) − 1`. The hypothesis `1 < 2cᵢ²` is the finiteness condition of the 1-D spread χ². -/
theorem chi2_stdGaussian_scale {n : ℕ} (c : Fin n → ℝ) (hc : ∀ i, (1 : ℝ) < 2 * (c i) ^ 2) :
    NearNull.chi2 (stdGaussian (EuclideanSpace ℝ (Fin n)))
        ((stdGaussian (EuclideanSpace ℝ (Fin n))).map
          (fun x : EuclideanSpace ℝ (Fin n) =>
            (WithLp.toLp 2 (fun i => c i * WithLp.ofLp x i) : EuclideanSpace ℝ (Fin n))))
      = (∏ i, ((c i) ^ 2 / Real.sqrt (2 * (c i) ^ 2 - 1))) - 1 := by
  classical
  have hv1 : (1 : ℝ≥0) ≠ 0 := one_ne_zero
  set e : (Fin n → ℝ) ≃ᵐ EuclideanSpace ℝ (Fin n) := MeasurableEquiv.toLp 2 (Fin n → ℝ) with he
  set s : Fin n → ℝ≥0 := fun i => ⟨(c i) ^ 2, sq_nonneg (c i)⟩ with hs
  set sc : (Fin n → ℝ) → (Fin n → ℝ) := fun y i => c i * y i with hsc
  set Sc : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin n) :=
    (fun x : EuclideanSpace ℝ (Fin n) =>
      (WithLp.toLp 2 (fun i => c i * WithLp.ofLp x i) : EuclideanSpace ℝ (Fin n))) with hSc
  -- coercion / positivity bookkeeping for the per-coordinate variances `s i = cᵢ²`
  have hscoe : ∀ i, ((s i : ℝ)) = (c i) ^ 2 := fun i => rfl
  have hspos : ∀ i, (0 : ℝ) < (c i) ^ 2 := fun i => by linarith [hc i]
  have hsne : ∀ i, s i ≠ 0 := fun i => (NNReal.coe_pos.mp (by rw [hscoe i]; exact hspos i)).ne'
  have hlt' : ∀ i, ((1 : ℝ≥0) : ℝ) < 2 * ((s i : ℝ)) := fun i => by
    rw [NNReal.coe_one, hscoe i]; exact hc i
  -- coordinatewise absolute continuity `N(0,1) ≪ N(0,cᵢ²)`
  have hac_i : ∀ i, gaussianReal (0 : ℝ) 1 ≪ gaussianReal (0 : ℝ) (s i) := fun i =>
    (gaussianReal_absolutelyContinuous 0 hv1).trans (gaussianReal_absolutelyContinuous' 0 (hsne i))
  -- measurability of the two scaling maps
  have hsc_meas : Measurable sc := by rw [hsc]; fun_prop
  have hSc_meas : Measurable Sc := by rw [hSc]; fun_prop
  -- Step 1: `stdGaussian` is the pushforward of the product of standard 1-D Gaussians by `e = toLp`.
  have hstd : stdGaussian (EuclideanSpace ℝ (Fin n))
      = (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map e := by
    rw [he, MeasurableEquiv.coe_toLp]; exact map_pi_eq_stdGaussian.symm
  -- Step 2: the scaling `Sc` conjugates through `e` to the coordinatewise scaling `sc`.
  have hconj : (Sc ∘ e) = (e ∘ sc) := by
    funext y
    simp only [Function.comp_apply, hSc, hsc, he, MeasurableEquiv.coe_toLp]
  -- Step 3: push the scaling through: `stdGaussian.map Sc = ((pi P₀).map sc).map e`.
  have hscale : (stdGaussian (EuclideanSpace ℝ (Fin n))).map Sc
      = ((Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map sc).map e := by
    rw [hstd, Measure.map_map hSc_meas e.measurable, Measure.map_map e.measurable hsc_meas, hconj]
  -- Step 4: the coordinatewise scaling of the product is the product of scaled factors.
  have hpimap : (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map sc
      = Measure.pi (fun i => gaussianReal (0 : ℝ) (s i)) := by
    rw [hsc, pi_map_pi (fun i => (measurable_const_mul (c i)).aemeasurable)]
    congr 1
    funext i
    rw [gaussianReal_map_const_mul (c i), mul_zero, mul_one]
    rfl
  -- Step 5: absolute continuity of the product laws (coordinatewise).
  have hac : Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)
      ≪ Measure.pi (fun i => gaussianReal (0 : ℝ) (s i)) :=
    NearNull.pi_ac n (fun _ => ℝ) (fun _ => gaussianReal (0 : ℝ) 1)
      (fun i => gaussianReal (0 : ℝ) (s i)) hac_i
  -- L² integrability side condition (per coordinate) feeding `chi2_pi`.
  have hInt : ∀ i, Integrable
      (fun x => (((gaussianReal (0 : ℝ) 1).rnDeriv (gaussianReal (0 : ℝ) (s i)) x).toReal) ^ 2)
      (gaussianReal (0 : ℝ) (s i)) := fun i =>
    integrable_rnDeriv_sq_gaussianReal_var 0 hv1 (hsne i) (hlt' i)
  -- Reduce to the product χ² by `toLp`-invariance, then rewrite the coordinatewise scaling.
  rw [hscale, hstd, hpimap, chi2_map_measurableEquiv _ _ hac e]
  -- Evaluate the product χ² by tensorization.
  rw [chi2_pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1) (fun i => gaussianReal (0 : ℝ) (s i))
      hac_i hInt]
  -- Each factor `χ² + 1 = cᵢ²/√(2cᵢ²−1)`.
  have hfac : ∀ i, NearNull.chi2 (gaussianReal (0 : ℝ) 1) (gaussianReal (0 : ℝ) (s i)) + 1
      = (c i) ^ 2 / Real.sqrt (2 * (c i) ^ 2 - 1) := by
    intro i
    rw [chi2_gaussianReal_var 0 hv1 (hsne i) (hlt' i), NNReal.coe_one, hscoe i, one_mul]
    ring
  simp_rw [hfac]

/-- **Single-coordinate spread χ² (whitening step).** Scaling only coordinate `0` by `s`
(with `1 < 2s²`, the finiteness condition) leaves the joint χ² equal to `s²/√(2s²−1) − 1`: the
diagonal product of `chi2_stdGaussian_scale` telescopes because every unscaled coordinate contributes
the factor `1/√(2·1−1) = 1`. -/
theorem chi2_stdGaussian_scale_single {m : ℕ} (s : ℝ) (hs : (1 : ℝ) < 2 * s ^ 2) :
    NearNull.chi2 (stdGaussian (EuclideanSpace ℝ (Fin (m + 1))))
        ((stdGaussian (EuclideanSpace ℝ (Fin (m + 1)))).map
          (fun x : EuclideanSpace ℝ (Fin (m + 1)) => (WithLp.toLp 2
            (Function.update (WithLp.ofLp x) 0 (s * WithLp.ofLp x 0)) :
              EuclideanSpace ℝ (Fin (m + 1)))))
      = s ^ 2 / Real.sqrt (2 * s ^ 2 - 1) - 1 := by
  classical
  set c : Fin (m + 1) → ℝ := Function.update (fun _ => (1 : ℝ)) 0 s with hc_def
  have hc0 : c 0 = s := by rw [hc_def, Function.update_self]
  have hci : ∀ b, b ≠ 0 → c b = 1 := fun b hb => by rw [hc_def, Function.update_of_ne hb]
  -- the single-coordinate scaling map coincides with the diagonal scaling map for this `c`
  have hmapeq : (fun x : EuclideanSpace ℝ (Fin (m + 1)) =>
        (WithLp.toLp 2 (Function.update (WithLp.ofLp x) 0 (s * WithLp.ofLp x 0)) :
          EuclideanSpace ℝ (Fin (m + 1))))
      = (fun x : EuclideanSpace ℝ (Fin (m + 1)) =>
          (WithLp.toLp 2 (fun i => c i * WithLp.ofLp x i) :
          EuclideanSpace ℝ (Fin (m + 1)))) := by
    funext x
    congr 1
    funext i
    rcases eq_or_ne i 0 with hi | hi
    · subst hi; rw [Function.update_self, hc0]
    · rw [Function.update_of_ne hi, hci i hi, one_mul]
  -- the diagonal-scaling hypothesis holds for this `c`
  have hc : ∀ i, (1 : ℝ) < 2 * (c i) ^ 2 := by
    intro i
    rcases eq_or_ne i 0 with hi | hi
    · subst hi; rw [hc0]; exact hs
    · rw [hci i hi]; norm_num
  rw [hmapeq, chi2_stdGaussian_scale c hc]
  -- the product telescopes to the single scaled coordinate
  have hprod : (∏ i, ((c i) ^ 2 / Real.sqrt (2 * (c i) ^ 2 - 1)))
      = s ^ 2 / Real.sqrt (2 * s ^ 2 - 1) := by
    rw [Finset.prod_eq_single (0 : Fin (m + 1))
        (fun b _ hb => by rw [hci b hb]; norm_num [Real.sqrt_one])
        (fun h => absurd (Finset.mem_univ _) h), hc0]
  rw [hprod]

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.chi2_stdGaussian_scale
#print axioms NearNull.chi2_stdGaussian_scale_single
