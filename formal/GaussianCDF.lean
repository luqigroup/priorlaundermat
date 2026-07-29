import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.CDF
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-!
# A small Gaussian-CDF layer for the paper

Reusable analytic helper mathlib lacks: the standard-normal CDF `Φ`, its order/symmetry properties,
the interval-probability (change-of-variables) identity, and the coverage closed form used by
`p:cover` (and later the near-null bounds).

* `Φ x := cdf (gaussianReal 0 1) x` — the standard normal CDF.
* `Φ_nonneg`, `Φ_le_one`, `monotone_Φ`, `strictMono_Φ`, `Φ_add_Φ_neg` (`Φ x + Φ (-x) = 1`).
* `gaussianReal_real_Ioc` — `(gaussianReal 0 1).real (Ioc a b) = Φ b - Φ a`.
* `stdGaussian_map` — the standardization `(gaussianReal m v).map ((· - m)/s) = gaussianReal 0 1`.
* `gaussian_real_Icc` / `coverage_eq` — the interval / coverage closed form
  `𝒩(m_⋆, s_⋆²)(|θ − m_reg| ≤ z s_reg) = Φ(δ + r) − Φ(δ − r)`, `δ = (m_reg − m_⋆)/s_⋆`,
  `r = z s_reg / s_⋆`.
-/

open MeasureTheory ProbabilityTheory Set
open scoped ENNReal NNReal

namespace GaussianCDF

/-- The **standard normal CDF** `Φ(x) = 𝒩(0,1)((-∞, x])`. -/
noncomputable def Φ (x : ℝ) : ℝ := cdf (gaussianReal 0 1) x

lemma Φ_eq_real (x : ℝ) : Φ x = (gaussianReal 0 1).real (Iic x) :=
  cdf_eq_real _ x

lemma Φ_nonneg (x : ℝ) : 0 ≤ Φ x := cdf_nonneg _ x

lemma Φ_le_one (x : ℝ) : Φ x ≤ 1 := cdf_le_one _ x

lemma monotone_Φ : Monotone Φ := monotone_cdf _

/-- `(gaussianReal 0 1).real (Ioc a b) = Φ b − Φ a` for `a ≤ b`. -/
lemma gaussianReal_real_Ioc {a b : ℝ} (hab : a ≤ b) :
    (gaussianReal 0 1).real (Ioc a b) = Φ b - Φ a := by
  have hsplit : (gaussianReal 0 1).real (Iic b)
      = (gaussianReal 0 1).real (Iic a) + (gaussianReal 0 1).real (Ioc a b) := by
    rw [← measureReal_union (Set.Iic_disjoint_Ioc le_rfl) measurableSet_Ioc,
      Set.Iic_union_Ioc_eq_Iic hab]
  rw [Φ_eq_real, Φ_eq_real]; linarith

/-- The gaussian measure gives positive mass to every nondegenerate interval. -/
lemma gaussianReal_Ioc_pos {a b : ℝ} (hab : a < b) : 0 < (gaussianReal 0 1) (Ioc a b) := by
  rw [pos_iff_ne_zero]
  intro h0
  have hac : volume ≪ gaussianReal (0 : ℝ) 1 := gaussianReal_absolutelyContinuous' 0 one_ne_zero
  have hv0 : volume (Ioc a b) = 0 := hac h0
  rw [Real.volume_Ioc, ENNReal.ofReal_eq_zero] at hv0
  linarith

/-- `Φ` is **strictly increasing** (the gaussian density is positive everywhere). -/
lemma strictMono_Φ : StrictMono Φ := by
  intro a b hab
  have h := gaussianReal_real_Ioc hab.le
  have h2 : (gaussianReal 0 1) (Ioc a b) ≠ 0 := (gaussianReal_Ioc_pos hab).ne'
  have hpos : 0 < (gaussianReal 0 1).real (Ioc a b) := by
    rw [measureReal_def]; exact ENNReal.toReal_pos h2 (measure_ne_top _ _)
  linarith

/-- **Symmetry**: `Φ x + Φ (-x) = 1`. -/
lemma Φ_add_Φ_neg (x : ℝ) : Φ x + Φ (-x) = 1 := by
  haveI : NoAtoms (gaussianReal (0 : ℝ) 1) := noAtoms_gaussianReal one_ne_zero
  have hmap : (gaussianReal (0 : ℝ) 1).map (fun t => -t) = gaussianReal 0 1 := by
    simpa using gaussianReal_map_neg (μ := (0 : ℝ)) (v := 1)
  -- reflection: `𝒩(0,1)(Iic (-x)) = 𝒩(0,1)(Ici x)`
  have hreflE : (gaussianReal 0 1) (Iic (-x)) = (gaussianReal 0 1) (Ici x) := by
    conv_lhs => rw [← hmap]
    rw [Measure.map_apply measurable_neg measurableSet_Iic]
    congr 1
    ext t; simp only [mem_preimage, mem_Iic, mem_Ici, neg_le_neg_iff]
  -- `Iic x` and `Ici x` cover `univ`, meeting in the null set `{x}`
  have hsumE : (gaussianReal 0 1) (Iic x) + (gaussianReal 0 1) (Ici x) = 1 := by
    have h := measure_union_add_inter (μ := gaussianReal 0 1) (Iic x)
      (measurableSet_Ici : MeasurableSet (Ici x))
    rw [Set.Iic_union_Ici, Set.Iic_inter_Ici, Set.Icc_self, measure_univ, measure_singleton,
      add_zero] at h
    exact h.symm
  have hrefl : (gaussianReal 0 1).real (Iic (-x)) = (gaussianReal 0 1).real (Ici x) := by
    rw [measureReal_def, measureReal_def, hreflE]
  have key : (gaussianReal 0 1).real (Iic x) + (gaussianReal 0 1).real (Ici x) = 1 := by
    rw [measureReal_def, measureReal_def,
      ← ENNReal.toReal_add (measure_ne_top _ _) (measure_ne_top _ _), hsumE, ENNReal.toReal_one]
  rw [Φ_eq_real, Φ_eq_real, hrefl]; exact key

/-- `Φ (-x) = 1 - Φ x`. -/
lemma Φ_neg (x : ℝ) : Φ (-x) = 1 - Φ x := by
  have := Φ_add_Φ_neg x; linarith

/-! ## The standardization change of variables and the coverage closed form -/

/-- **Standardization.** For a Gaussian with variance `v ≠ 0` and standard deviation `s = √v`,
the affine map `x ↦ (x − m)/s` pushes `𝒩(m, v)` to `𝒩(0, 1)`. -/
lemma stdGaussian_map (m : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    (gaussianReal m v).map (fun x => (x - m) / Real.sqrt v) = gaussianReal 0 1 := by
  have hcomp : (fun x => (x - m) / Real.sqrt v)
      = (fun x => x / Real.sqrt v) ∘ (fun x => x - m) := by ext x; simp
  rw [hcomp, ← Measure.map_map (by fun_prop) (by fun_prop), gaussianReal_map_sub_const, sub_self,
    gaussianReal_map_div_const, zero_div]
  congr 1
  rw [← NNReal.coe_inj, NNReal.coe_div, NNReal.coe_mk, NNReal.coe_one,
    Real.sq_sqrt v.coe_nonneg, div_self (by exact_mod_cast hv)]

/-- The preimage of `Icc c d` under standardization `x ↦ (x−m)/s` (`s > 0`) is
`Icc (c·s + m) (d·s + m)`. -/
lemma preimage_std {m s c d : ℝ} (hs : 0 < s) :
    (fun x => (x - m) / s) ⁻¹' Icc c d = Icc (c * s + m) (d * s + m) := by
  ext x
  simp only [Set.mem_preimage, Set.mem_Icc]
  rw [le_div_iff₀ hs, div_le_iff₀ hs]
  constructor
  · rintro ⟨h1, h2⟩; constructor <;> [linarith; linarith]
  · rintro ⟨h1, h2⟩; constructor <;> [linarith; linarith]

/-- **Interval closed form.** For variance `v ≠ 0`, standard deviation `s = √v`, and `a ≤ b`,
`𝒩(m, v)([a, b]) = Φ((b − m)/s) − Φ((a − m)/s)`. -/
lemma gaussian_real_Icc (m : ℝ) {v : ℝ≥0} (hv : v ≠ 0) {a b : ℝ} (hab : a ≤ b) :
    (gaussianReal m v).real (Icc a b)
      = Φ ((b - m) / Real.sqrt v) - Φ ((a - m) / Real.sqrt v) := by
  haveI : NoAtoms (gaussianReal (0 : ℝ) 1) := noAtoms_gaussianReal one_ne_zero
  set s : ℝ := Real.sqrt v with hs
  have hspos : 0 < s := Real.sqrt_pos.mpr (NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hv))
  have hsne : s ≠ 0 := ne_of_gt hspos
  have hcd : (a - m) / s ≤ (b - m) / s := (div_le_div_iff_of_pos_right hspos).mpr (by linarith)
  have he1 : (a - m) / s * s + m = a := by
    rw [div_mul_eq_mul_div, mul_div_assoc, div_self hsne, mul_one]; ring
  have he2 : (b - m) / s * s + m = b := by
    rw [div_mul_eq_mul_div, mul_div_assoc, div_self hsne, mul_one]; ring
  have hpre : (fun x => (x - m) / s) ⁻¹' Icc ((a - m) / s) ((b - m) / s) = Icc a b := by
    rw [preimage_std hspos, he1, he2]
  have hkey : (gaussianReal m v) (Icc a b)
      = (gaussianReal 0 1) (Icc ((a - m) / s) ((b - m) / s)) := by
    rw [← stdGaussian_map m hv, Measure.map_apply (by fun_prop) measurableSet_Icc, hpre]
  rw [measureReal_def, hkey, ← measureReal_def,
    ← measureReal_congr (Ioc_ae_eq_Icc (a := (a - m) / s) (b := (b - m) / s)),
    gaussianReal_real_Ioc hcd]

/-! ## The standard normal density `φ` and `HasDerivAt Φ φ` (for `∂C/∂δ` in `p:cover`) -/

/-- The **standard normal density** `φ(x) = (√(2π))⁻¹ exp(-x²/2)`. -/
noncomputable def φ (x : ℝ) : ℝ := gaussianPDFReal 0 1 x

lemma φ_apply (x : ℝ) : φ x = (Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-x ^ 2 / 2) := by
  unfold φ gaussianPDFReal; simp

lemma continuous_φ : Continuous φ := by
  have : φ = fun x => (Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-x ^ 2 / 2) := funext φ_apply
  rw [this]; fun_prop

/-- `φ` is **strictly decreasing in `|·|`**: `|a| < |b| → φ b < φ a`. -/
lemma φ_lt_of_abs_lt {a b : ℝ} (h : |a| < |b|) : φ b < φ a := by
  rw [φ_apply, φ_apply]
  have hab2 : a ^ 2 < b ^ 2 := by
    rw [← sq_abs a, ← sq_abs b]; exact pow_lt_pow_left₀ h (abs_nonneg a) (by norm_num)
  exact mul_lt_mul_of_pos_left (Real.exp_lt_exp.mpr (by linarith)) (by positivity)

/-- `Φ` is the antiderivative of the density: `Φ x = ∫_{-∞}^x φ`. -/
lemma Φ_eq_integral (x : ℝ) : Φ x = ∫ y in Set.Iic x, φ y := by
  rw [Φ_eq_real, measureReal_def, gaussianReal_apply_eq_integral (0 : ℝ) one_ne_zero (Set.Iic x),
    ENNReal.toReal_ofReal (integral_nonneg fun y => gaussianPDFReal_nonneg 0 1 y)]
  rfl

/-- **`HasDerivAt Φ (φ x) x`** — the CDF's derivative is the density (Fundamental Theorem of
Calculus, from the antiderivative representation). -/
lemma hasDerivAt_Φ (x : ℝ) : HasDerivAt Φ (φ x) x := by
  have hint : Integrable φ := integrable_gaussianPDFReal 0 1
  have hrep : Φ = fun u => (∫ y in Set.Iic (0 : ℝ), φ y) + ∫ y in (0 : ℝ)..u, φ y := by
    funext u
    rw [Φ_eq_integral]
    have := intervalIntegral.integral_Iic_sub_Iic (f := φ) (μ := volume) (a := (0 : ℝ)) (b := u)
      hint.integrableOn hint.integrableOn
    linarith
  rw [hrep]
  have hd : HasDerivAt (fun u => ∫ y in (0 : ℝ)..u, φ y) (φ x) x :=
    (continuous_φ.integral_hasStrictDerivAt 0 x).hasDerivAt
  exact hd.const_add (∫ y in Set.Iic (0 : ℝ), φ y)

lemma continuous_Φ : Continuous Φ :=
  continuous_iff_continuousAt.mpr fun x => (hasDerivAt_Φ x).continuousAt

/-- `Φ 0 = 1/2`. -/
lemma Φ_zero : Φ 0 = 1 / 2 := by
  have := Φ_add_Φ_neg 0; rw [neg_zero] at this; linarith

end GaussianCDF

-- #print axioms audit (the Φ facts the coverage layer consumes)
#print axioms GaussianCDF.strictMono_Φ
#print axioms GaussianCDF.hasDerivAt_Φ
#print axioms GaussianCDF.Φ_add_Φ_neg
#print axioms GaussianCDF.gaussian_real_Icc
