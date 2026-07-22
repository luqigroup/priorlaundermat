import Mathlib
import ChiSquaredTV
import ChiSquaredInvariance
import ChiSquaredPi
import GaussianChiSquared

/-!
# Multivariate isotropic Gaussian χ²

The χ² of the standard Gaussian on `EuclideanSpace ℝ (Fin n)` shifted by `c`, against the standard
Gaussian, is `exp(‖c‖²) − 1` — the identity-covariance mean-shift χ².

The route is the isotropic reduction (`toLp` invariance `NearNull.chi2_map_measurableEquiv` +
coordinatewise-shift `Measure.pi_map_pi`) to a product of shifted 1-D Gaussians, evaluated by
`NearNull.chi2_pi` (tensorization) and the 1-D Gaussian χ² `chi2_gaussianReal`; the per-coordinate
`1 + χ²` factors are `exp((cᵢ)²)`, whose product is `exp(∑ (cᵢ)²) = exp(‖c‖²)`.

## Result
* `NearNull.chi2_stdGaussian_add` — `χ²(N₀(·+c) ‖ N₀) = exp(‖c‖²) − 1` on `EuclideanSpace ℝ (Fin n)`.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms NearNull.chi2_stdGaussian_add` lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure ProbabilityTheory
open scoped NNReal

namespace NearNull

/-- **Finiteness of the squared 1-D Gaussian density** (the side-condition feeding `chi2_pi`).
Derived from the near-null χ² finiteness `integrable_chi2_gaussianReal` (giving `(r−1)²` integrable)
and `Measure.integrable_toReal_rnDeriv` (giving `r` integrable), via `r² = (r−1)² + 2r − 1`. -/
private lemma integrable_rnDeriv_sq_gaussianReal (m : ℝ) :
    Integrable
      (fun x => (((gaussianReal m 1).rnDeriv (gaussianReal (0 : ℝ) 1) x).toReal) ^ 2)
      (gaussianReal (0 : ℝ) 1) := by
  have hv1 : (1 : ℝ≥0) ≠ 0 := one_ne_zero
  have h1 := integrable_chi2_gaussianReal m 0 hv1
  have h2 : Integrable
      (fun x => ((gaussianReal m 1).rnDeriv (gaussianReal (0 : ℝ) 1) x).toReal)
      (gaussianReal (0 : ℝ) 1) := Measure.integrable_toReal_rnDeriv
  refine ((h1.add (h2.const_mul 2)).sub (integrable_const (1 : ℝ))).congr ?_
  filter_upwards with x
  simp only [Pi.add_apply, Pi.sub_apply]
  ring

/-- **Isotropic multivariate Gaussian χ².** The χ² of the standard Gaussian on
`EuclideanSpace ℝ (Fin n)` shifted by `c`, against the standard Gaussian, is `exp(‖c‖²) − 1`. -/
theorem chi2_stdGaussian_add {n : ℕ} (c : EuclideanSpace ℝ (Fin n)) :
    chi2 ((stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + c))
         (stdGaussian (EuclideanSpace ℝ (Fin n)))
      = Real.exp (‖c‖ ^ 2) - 1 := by
  classical
  have hv1 : (1 : ℝ≥0) ≠ 0 := one_ne_zero
  set e : (Fin n → ℝ) ≃ᵐ EuclideanSpace ℝ (Fin n) := MeasurableEquiv.toLp 2 (Fin n → ℝ) with he
  set c' : Fin n → ℝ := WithLp.ofLp c with hc'
  -- shared coordinatewise absolute-continuity witness
  have hac_i : ∀ i : Fin n, gaussianReal (c' i) 1 ≪ gaussianReal (0 : ℝ) 1 := fun i =>
    (gaussianReal_absolutelyContinuous (c' i) hv1).trans
      (gaussianReal_absolutelyContinuous' 0 hv1)
  -- Step 1: `stdGaussian` is the pushforward of the product of standard 1-D Gaussians by `e = toLp`.
  have hstd : stdGaussian (EuclideanSpace ℝ (Fin n))
      = (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map e := by
    rw [he, MeasurableEquiv.coe_toLp]; exact map_pi_eq_stdGaussian.symm
  -- Step 2: the shift `· + c` conjugates through `e` to the coordinatewise shift `· + c'`.
  have hconj : ((· + c) ∘ e) = (e ∘ (· + c')) := by
    funext x
    simp only [Function.comp_apply, he, MeasurableEquiv.coe_toLp, hc', WithLp.toLp_add,
      WithLp.toLp_ofLp]
  -- Step 3: push the shift through: `stdGaussian.map (·+c) = ((pi P₀).map (·+c')).map e`.
  have hshift : (stdGaussian (EuclideanSpace ℝ (Fin n))).map (· + c)
      = ((Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map (· + c')).map e := by
    rw [hstd, Measure.map_map (by fun_prop) e.measurable,
        Measure.map_map e.measurable (by fun_prop), hconj]
  -- Step 4: the coordinatewise shift of the product is the product of shifted factors.
  have hpimap : (Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map (· + c')
      = Measure.pi (fun i => gaussianReal (c' i) 1) := by
    have hfun : ((· + c') : (Fin n → ℝ) → (Fin n → ℝ)) = (fun x i => x i + c' i) := rfl
    rw [hfun, pi_map_pi (fun i => (measurable_add_const (c' i)).aemeasurable)]
    congr 1
    funext i
    rw [gaussianReal_map_add_const, zero_add]
  -- Step 5: absolute continuity of the shifted product w.r.t. the product (coordinatewise).
  have hac : Measure.pi (fun i => gaussianReal (c' i) 1)
      ≪ Measure.pi (fun _ : Fin n => gaussianReal (0 : ℝ) 1) :=
    NearNull.pi_ac n (fun _ => ℝ) (fun i => gaussianReal (c' i) 1)
      (fun _ => gaussianReal (0 : ℝ) 1) hac_i
  -- Reduce to the product χ² by `toLp`-invariance, then rewrite the coordinatewise shift.
  rw [hshift, hstd, hpimap, chi2_map_measurableEquiv _ _ hac e]
  -- Step 6: evaluate the product χ² by tensorization.
  rw [chi2_pi (fun i => gaussianReal (c' i) 1) (fun _ : Fin n => gaussianReal (0 : ℝ) 1) hac_i
      (fun i => integrable_rnDeriv_sq_gaussianReal (c' i))]
  -- Step 7: each factor `χ² + 1 = exp((cᵢ)²)`; the product is `exp(∑ (cᵢ)²) = exp(‖c‖²)`.
  have hfac : ∀ i : Fin n,
      chi2 (gaussianReal (c' i) 1) (gaussianReal (0 : ℝ) 1) + 1 = Real.exp ((c' i) ^ 2) := by
    intro i
    rw [chi2_gaussianReal (c' i) 0 hv1, sub_zero, NNReal.coe_one, div_one]
    ring
  have hnorm : (∑ i : Fin n, (c' i) ^ 2) = ‖c‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq]
  simp_rw [hfac]
  rw [← Real.exp_sum, hnorm]

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.chi2_stdGaussian_add
