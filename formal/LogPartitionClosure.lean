/-
# Closing `p:mean`: the concrete envelope, strict convexity of the log-partition, and the
composed collapse from the model's own hypotheses

`LogPartitionGradient` proved `∇Λ = 𝔼[x_R ∣ τ]` under a locally-uniform integrable-envelope
hypothesis (`dom`) and composed with the collapse under a strict-convexity hypothesis (`hconv`).
This file **closes both**, so that `p:mean`'s conclusion follows from the model's stated
hypotheses alone:

* **(C1) The envelope** (`dom_of_moment`): if the tilt dominates a positive-definite quadratic —
  `q0‖x_R‖²/2 ≤ q` a.e., the paper's "`q` is positive definite" — then quadratic beats linear:
  for `‖τ‖ ≤ R`, `⟪τ,x⟫ − q0‖x‖²/2 ≤ R‖x‖ − q0‖x‖²/2 ≤ R²/(2q0)` (`exp_tilt_le`), so on the unit
  ball around any `τ₀` the integrand is bounded by a constant and the derivative integrand by
  `C‖X a‖`.  The envelope is integrable exactly under `p:mean`'s stated moment hypothesis
  `𝔼‖x_R‖ < ∞` (`hX1`) on a finite measure — no extra analytic input.
* **(C2) Strict convexity** (`strictConvexOn_logPartition`), *without Hessians*.
  Plain convexity of `Λ` is Hölder's inequality (`M(tτ+sτ′) ≤ M(τ)ᵗM(τ′)ˢ`,
  `convexOn_logPartition`); the **strict midpoint** inequality is the equality analysis of
  Cauchy–Schwarz done by hand: with `f = exp((⟪τ,X⟫−q)/2)`, `g = exp((⟪τ′,X⟫−q)/2)` and
  `c = M(mid)/M(τ′)`, `0 ≤ ∫(f−cg)² = M(τ) − 2c·M(mid) + c²·M(τ′)` gives
  `M(mid)² ≤ M(τ)M(τ′)`, and equality forces `f = cg` a.e., i.e. `⟪τ−τ′, X a⟫` a.e. **constant**
  — excluded by the affine-span hypothesis `hspan` (`sq_M_midpoint_lt`).  Midpoint-strict +
  convex ⟹ strictly convex by direct propagation (`strictConvexOn_of_convexOn_of_midpoint`:
  for `b < 1/2`, `a•x + b•y = 2b•mid + (1−2b)•x`, so
  `Λ(a•x+b•y) ≤ 2bΛ(mid) + (1−2b)Λ(x) < aΛ(x) + bΛ(y)` outright).
* **(C3) The capstone** (`posterior_mean_collapse_closed`, and with the concrete quadratic tilt
  `q(a) = ⟪X a, Q (X a)⟫/2`, `posterior_mean_collapse_quadratic`): `p:mean`'s collapse with
  hypotheses ONLY the measure (finite, nonzero), measurability, `𝔼‖x_R‖ < ∞`, the variationally
  positive-definite quadratic (`∀x, q0‖x‖² ≤ ⟪x, Qx⟫`), and the affine-span condition — both
  `dom` and `hconv` are now theorems, not inputs.
* **(C4) The Gaussian aggregate of `r:map-lg`** (`integral_Phi_affine`):
  `𝔼_{Z∼𝒩(0,1)}[Φ(a + bZ)] = Φ(a/√(1+b²))`, by reading `Φ(a+bz)` as `𝒩(0,1)(W ≤ a+bz)` and
  recognizing the double integral as the convolution `𝒩(0,b²) ∗ 𝒩(0,1) = 𝒩(0,1+b²)` of `−bZ`
  and `W`, then standardizing (`gaussianReal_real_Iic`).

## Explicit hypotheses (none hidden)

The structural inputs `hfac`/`hres` (the reconstruction factors through the resolved statistic;
its resolved component is the tilted mean) remain the preamble's definitional relations, exactly
as in `LogPartitionGradient`.  The affine-span condition is stated operationally: no direction
`v ≠ 0` has `⟪v, X a⟫` a.e. constant.  Everything else is proved.

No `sorry`/`admit`; `#print axioms` on every public result at the end.
-/
import Mathlib
import LogPartitionGradient
import GaussianCDF

open MeasureTheory Filter InnerProductSpace ProbabilityTheory
open scoped Topology InnerProductSpace ENNReal NNReal

namespace LogPartitionClosure

open LogPartitionGradient

/-! ## The midpoint bootstrap: convex + strictly midpoint-convex ⟹ strictly convex

Mathlib has no "midpoint-convex + continuous ⟹ convex" lemma; none is needed.  Convexity comes
from Hölder below, and strictness at a general interior `t` *propagates directly* from the strict
midpoint: for `b < 1/2` the point `a•x + b•y` is the combination `2b • mid + (1−2b) • x`, so
plain convexity applied to that combination plus the strict midpoint give strictness outright. -/

section Midpoint

variable {F : Type*} [AddCommGroup F] [Module ℝ F]

/-- **Convex and strictly midpoint-convex functions are strictly convex.** -/
theorem strictConvexOn_of_convexOn_of_midpoint {f : F → ℝ}
    (hconv : ConvexOn ℝ Set.univ f)
    (hmid : ∀ x y : F, x ≠ y → f ((1/2 : ℝ) • x + (1/2 : ℝ) • y) < (f x + f y) / 2) :
    StrictConvexOn ℝ Set.univ f := by
  refine ⟨convex_univ, fun x _ y _ hxy a b ha hb hab => ?_⟩
  rcases lt_trichotomy b (1/2 : ℝ) with hb2 | hb2 | hb2
  · -- `b < 1/2`:  a•x + b•y = 2b • mid + (1−2b) • x
    have h2b : (0:ℝ) < 2 * b := by linarith
    have h2b' : (0:ℝ) ≤ 1 - 2 * b := by linarith
    have hkey : a • x + b • y
        = (2*b) • ((1/2 : ℝ) • x + (1/2 : ℝ) • y) + (1 - 2*b) • x := by
      match_scalars <;> linarith
    have hcvx := hconv.2 (Set.mem_univ ((1/2 : ℝ) • x + (1/2 : ℝ) • y)) (Set.mem_univ x)
      h2b.le h2b' (by ring)
    have hm := hmid x y hxy
    have hlt := mul_lt_mul_of_pos_left hm h2b
    calc f (a • x + b • y)
        = f ((2*b) • ((1/2 : ℝ) • x + (1/2 : ℝ) • y) + (1 - 2*b) • x) := by rw [hkey]
      _ ≤ (2*b) • f ((1/2 : ℝ) • x + (1/2 : ℝ) • y) + (1 - 2*b) • f x := hcvx
      _ < (2*b) * ((f x + f y) / 2) + (1 - 2*b) * f x := by
          simp only [smul_eq_mul]; linarith
      _ = a • f x + b • f y := by
          simp only [smul_eq_mul]; linear_combination (-(f x)) * hab
  · -- `b = 1/2`: this is the strict midpoint itself
    subst hb2
    have ha' : a = 1/2 := by linarith
    subst ha'
    have hm := hmid x y hxy
    simp only [smul_eq_mul]
    linarith
  · -- `b > 1/2`, i.e. `a < 1/2`:  a•x + b•y = 2a • mid + (1−2a) • y
    have h2a : (0:ℝ) < 2 * a := by linarith
    have h2a' : (0:ℝ) ≤ 1 - 2 * a := by linarith
    have hkey : a • x + b • y
        = (2*a) • ((1/2 : ℝ) • x + (1/2 : ℝ) • y) + (1 - 2*a) • y := by
      match_scalars <;> linarith
    have hcvx := hconv.2 (Set.mem_univ ((1/2 : ℝ) • x + (1/2 : ℝ) • y)) (Set.mem_univ y)
      h2a.le h2a' (by ring)
    have hm := hmid x y hxy
    have hlt := mul_lt_mul_of_pos_left hm h2a
    calc f (a • x + b • y)
        = f ((2*a) • ((1/2 : ℝ) • x + (1/2 : ℝ) • y) + (1 - 2*a) • y) := by rw [hkey]
      _ ≤ (2*a) • f ((1/2 : ℝ) • x + (1/2 : ℝ) • y) + (1 - 2*a) • f y := hcvx
      _ < (2*a) * ((f x + f y) / 2) + (1 - 2*a) * f y := by
          simp only [smul_eq_mul]; linarith
      _ = a • f x + b • f y := by
          simp only [smul_eq_mul]; linear_combination (-(f y)) * hab

end Midpoint

variable {α : Type*} [MeasurableSpace α] {μ : Measure α}
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H] [CompleteSpace H]

/-! ## (C1) The concrete envelope: quadratic beats linear -/

section Envelope

variable {X : α → H} {q : α → ℝ}

omit [CompleteSpace H] in
/-- **Quadratic beats linear.**  If the tilt dominates the quadratic `q0‖x‖²/2` and `‖τ‖ ≤ R`,
then `⟪τ,x⟫ − q(x) ≤ R‖x‖ − q0‖x‖²/2 ≤ R²/(2q0)`: the integrand is bounded by a constant,
uniformly over the ball of radius `R`. -/
lemma exp_tilt_le {q0 R : ℝ} (hq0 : 0 < q0) {τ x : H} (hτ : ‖τ‖ ≤ R) {qx : ℝ}
    (hql : q0 * ‖x‖ ^ 2 / 2 ≤ qx) :
    Real.exp (⟪τ, x⟫_ℝ - qx) ≤ Real.exp (R ^ 2 / (2 * q0)) := by
  apply Real.exp_le_exp.mpr
  have h1 : ⟪τ, x⟫_ℝ ≤ R * ‖x‖ :=
    (real_inner_le_norm τ x).trans (mul_le_mul_of_nonneg_right hτ (norm_nonneg x))
  rw [le_div_iff₀ (by positivity)]
  nlinarith [sq_nonneg (q0 * ‖x‖ - R)]

omit [CompleteSpace H] in
/-- On a finite measure, the positive-definite tilt makes the integrand integrable at **every**
natural parameter — with no moment hypothesis at all (the integrand is bounded). -/
lemma integrable_exp_tilt [IsFiniteMeasure μ]
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    {q0 : ℝ} (hq0 : 0 < q0) (hql : ∀ᵐ a ∂μ, q0 * ‖X a‖ ^ 2 / 2 ≤ q a) (τ : H) :
    Integrable (fun a => Real.exp (⟪τ, X a⟫_ℝ - q a)) μ := by
  refine (integrable_const (Real.exp (‖τ‖ ^ 2 / (2 * q0)))).mono'
    (aestronglyMeasurable_integrand hX hq τ) ?_
  filter_upwards [hql] with a ha
  rw [Real.norm_eq_abs, Real.abs_exp]
  exact exp_tilt_le hq0 le_rfl ha

omit [CompleteSpace H] in
/-- **(C1) The envelope, discharging `LogPartitionGradient`'s `dom` hypothesis** from the model's
own hypotheses: measurability, the variationally positive-definite quadratic tilt
(`q0‖X a‖²/2 ≤ q a` a.e.), and the first moment `𝔼‖X‖ < ∞` — exactly `p:mean`'s stated moment
hypothesis.  On the unit ball around `τ₀` the derivative integrand is bounded by
`exp((‖τ₀‖+1)²/(2q0)) · ‖X a‖`, integrable by `hX1`. -/
theorem dom_of_moment [IsFiniteMeasure μ]
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    (hX1 : Integrable (fun a => ‖X a‖) μ)
    {q0 : ℝ} (hq0 : 0 < q0) (hql : ∀ᵐ a ∂μ, q0 * ‖X a‖ ^ 2 / 2 ≤ q a) :
    ∀ τ₀ : H, ∃ s ∈ 𝓝 τ₀, ∃ bound : α → ℝ,
      Integrable (fun a => Real.exp (⟪τ₀, X a⟫_ℝ - q a)) μ ∧
      (∀ᵐ a ∂μ, ∀ τ ∈ s, Real.exp (⟪τ, X a⟫_ℝ - q a) * ‖X a‖ ≤ bound a) ∧
      Integrable bound μ := by
  intro τ₀
  refine ⟨Metric.ball τ₀ 1, Metric.ball_mem_nhds τ₀ one_pos,
    fun a => Real.exp ((‖τ₀‖ + 1) ^ 2 / (2 * q0)) * ‖X a‖,
    integrable_exp_tilt hX hq hq0 hql τ₀, ?_, hX1.const_mul _⟩
  filter_upwards [hql] with a ha τ hτ
  have hτn : ‖τ‖ ≤ ‖τ₀‖ + 1 := by
    have h1 : ‖τ - τ₀‖ < 1 := mem_ball_iff_norm.mp hτ
    have h2 : ‖τ‖ - ‖τ₀‖ ≤ ‖τ - τ₀‖ := norm_sub_norm_le τ τ₀
    linarith
  exact mul_le_mul_of_nonneg_right (exp_tilt_le hq0 hτn ha) (norm_nonneg _)

/-- The multivariate gradient `∇Λ(τ) = 𝔼[x_R ∣ τ]` at **every** `τ`, from the model's hypotheses
alone (the former `dom` input is now `dom_of_moment`). -/
theorem hasGradientAt_logPartition_of_moment [IsFiniteMeasure μ] (hμ : μ ≠ 0)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    (hX1 : Integrable (fun a => ‖X a‖) μ)
    {q0 : ℝ} (hq0 : 0 < q0) (hql : ∀ᵐ a ∂μ, q0 * ‖X a‖ ^ 2 / 2 ≤ q a) :
    ∀ τ, HasGradientAt (Λ μ X q) (condMean μ X q τ) τ :=
  hasGradientAt_logPartition_forall X q hX hq hμ (dom_of_moment hX hq hX1 hq0 hql)

end Envelope

/-! ## (C2) Strict convexity of `Λ` from the affine-span condition, without Hessians -/

section Convexity

variable {X : α → H} {q : α → ℝ}

omit [CompleteSpace H] in
/-- **Convexity of the log-partition is Hölder's inequality**:
`M(tτ + sτ′) = ∫ (e^{⟪τ,X⟫−q})ᵗ(e^{⟪τ′,X⟫−q})ˢ ≤ M(τ)ᵗ M(τ′)ˢ` with conjugate exponents
`1/t, 1/s`, then take logarithms. -/
lemma convexOn_logPartition (hμ : μ ≠ 0)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    (hInt : ∀ τ : H, Integrable (fun a => Real.exp (⟪τ, X a⟫_ℝ - q a)) μ) :
    ConvexOn ℝ Set.univ (Λ μ X q) := by
  refine ⟨convex_univ, fun τ _ τ' _ t s ht hs hts => ?_⟩
  rcases eq_or_lt_of_le ht with ht0 | ht0
  · have hs1 : s = 1 := by linarith
    subst hs1
    rw [← ht0]
    simp
  rcases eq_or_lt_of_le hs with hs0 | hs0
  · have ht1 : t = 1 := by linarith
    subst ht1
    rw [← hs0]
    simp
  -- interior case `0 < t`, `0 < s`, `t + s = 1`: Hölder with exponents `1/t`, `1/s`
  have hpq : (1/t).HolderConjugate (1/s) := Real.holderConjugate_one_div ht0 hs0 hts
  have hbase : AEStronglyMeasurable (fun x => ⟪τ, X x⟫_ℝ - q x) μ :=
    ((Continuous.inner continuous_const continuous_id).comp_aestronglyMeasurable hX).sub hq
  have hbase' : AEStronglyMeasurable (fun x => ⟪τ', X x⟫_ℝ - q x) μ :=
    ((Continuous.inner continuous_const continuous_id).comp_aestronglyMeasurable hX).sub hq
  have hfm : AEStronglyMeasurable (fun x => Real.exp (t * (⟪τ, X x⟫_ℝ - q x))) μ :=
    Real.continuous_exp.comp_aestronglyMeasurable (hbase.const_mul t)
  have hgm : AEStronglyMeasurable (fun x => Real.exp (s * (⟪τ', X x⟫_ℝ - q x))) μ :=
    Real.continuous_exp.comp_aestronglyMeasurable (hbase'.const_mul s)
  have hPt0 : ENNReal.ofReal (1/t) ≠ 0 := by
    rw [Ne, ENNReal.ofReal_eq_zero, not_le]; positivity
  have hPs0 : ENNReal.ofReal (1/s) ≠ 0 := by
    rw [Ne, ENNReal.ofReal_eq_zero, not_le]; positivity
  have hfpow : (fun x => ‖Real.exp (t * (⟪τ, X x⟫_ℝ - q x))‖ ^ (ENNReal.ofReal (1/t)).toReal)
      = fun x => Real.exp (⟪τ, X x⟫_ℝ - q x) := by
    funext x
    rw [ENNReal.toReal_ofReal (by positivity), Real.norm_eq_abs, Real.abs_exp, ← Real.exp_mul]
    congr 1
    field_simp
  have hgpow : (fun x => ‖Real.exp (s * (⟪τ', X x⟫_ℝ - q x))‖ ^ (ENNReal.ofReal (1/s)).toReal)
      = fun x => Real.exp (⟪τ', X x⟫_ℝ - q x) := by
    funext x
    rw [ENNReal.toReal_ofReal (by positivity), Real.norm_eq_abs, Real.abs_exp, ← Real.exp_mul]
    congr 1
    field_simp
  have hfLp : MemLp (fun x => Real.exp (t * (⟪τ, X x⟫_ℝ - q x))) (ENNReal.ofReal (1/t)) μ := by
    refine (memLp_norm_rpow_iff hfm hPt0 ENNReal.ofReal_ne_top).mp ?_
    rw [ENNReal.div_self hPt0 ENNReal.ofReal_ne_top]
    show MemLp (fun x =>
      ‖Real.exp (t * (⟪τ, X x⟫_ℝ - q x))‖ ^ (ENNReal.ofReal (1/t)).toReal) 1 μ
    rw [hfpow, memLp_one_iff_integrable]
    exact hInt τ
  have hgLp : MemLp (fun x => Real.exp (s * (⟪τ', X x⟫_ℝ - q x))) (ENNReal.ofReal (1/s)) μ := by
    refine (memLp_norm_rpow_iff hgm hPs0 ENNReal.ofReal_ne_top).mp ?_
    rw [ENNReal.div_self hPs0 ENNReal.ofReal_ne_top]
    show MemLp (fun x =>
      ‖Real.exp (s * (⟪τ', X x⟫_ℝ - q x))‖ ^ (ENNReal.ofReal (1/s)).toReal) 1 μ
    rw [hgpow, memLp_one_iff_integrable]
    exact hInt τ'
  have hH := integral_mul_le_Lp_mul_Lq_of_nonneg hpq
    (Eventually.of_forall fun x => (Real.exp_pos _).le)
    (Eventually.of_forall fun x => (Real.exp_pos _).le) hfLp hgLp
  -- identify the three integrals with partition functions
  have e1 : (∫ x, Real.exp (t * (⟪τ, X x⟫_ℝ - q x)) ^ (1/t : ℝ) ∂μ) = M μ X q τ := by
    refine integral_congr_ae (Eventually.of_forall fun x => ?_)
    show Real.exp (t * (⟪τ, X x⟫_ℝ - q x)) ^ (1/t : ℝ) = Real.exp (⟪τ, X x⟫_ℝ - q x)
    rw [← Real.exp_mul]
    congr 1
    field_simp
  have e2 : (∫ x, Real.exp (s * (⟪τ', X x⟫_ℝ - q x)) ^ (1/s : ℝ) ∂μ) = M μ X q τ' := by
    refine integral_congr_ae (Eventually.of_forall fun x => ?_)
    show Real.exp (s * (⟪τ', X x⟫_ℝ - q x)) ^ (1/s : ℝ) = Real.exp (⟪τ', X x⟫_ℝ - q x)
    rw [← Real.exp_mul]
    congr 1
    field_simp
  have hMle : M μ X q (t • τ + s • τ') ≤ (M μ X q τ) ^ (t : ℝ) * (M μ X q τ') ^ (s : ℝ) := by
    calc M μ X q (t • τ + s • τ')
        = ∫ x, Real.exp (t * (⟪τ, X x⟫_ℝ - q x)) * Real.exp (s * (⟪τ', X x⟫_ℝ - q x)) ∂μ := by
          refine integral_congr_ae (Eventually.of_forall fun x => ?_)
          show Real.exp (⟪t • τ + s • τ', X x⟫_ℝ - q x)
            = Real.exp (t * (⟪τ, X x⟫_ℝ - q x)) * Real.exp (s * (⟪τ', X x⟫_ℝ - q x))
          rw [← Real.exp_add]
          congr 1
          simp only [inner_add_left, real_inner_smul_left]
          linear_combination (q x) * hts
      _ ≤ (∫ x, Real.exp (t * (⟪τ, X x⟫_ℝ - q x)) ^ (1/t : ℝ) ∂μ) ^ (1/(1/t) : ℝ)
            * (∫ x, Real.exp (s * (⟪τ', X x⟫_ℝ - q x)) ^ (1/s : ℝ) ∂μ) ^ (1/(1/s) : ℝ) := hH
      _ = (M μ X q τ) ^ (t : ℝ) * (M μ X q τ') ^ (s : ℝ) := by
          rw [e1, e2, one_div_one_div, one_div_one_div]
  have hMτ := M_pos (hInt τ) hμ
  have hMτ' := M_pos (hInt τ') hμ
  have hMm := M_pos (hInt (t • τ + s • τ')) hμ
  simp only [smul_eq_mul]
  show Real.log (M μ X q (t • τ + s • τ'))
    ≤ t * Real.log (M μ X q τ) + s * Real.log (M μ X q τ')
  calc Real.log (M μ X q (t • τ + s • τ'))
      ≤ Real.log ((M μ X q τ) ^ (t : ℝ) * (M μ X q τ') ^ (s : ℝ)) := Real.log_le_log hMm hMle
    _ = t * Real.log (M μ X q τ) + s * Real.log (M μ X q τ') := by
        rw [Real.log_mul (Real.rpow_pos_of_pos hMτ t).ne' (Real.rpow_pos_of_pos hMτ' s).ne',
          Real.log_rpow hMτ, Real.log_rpow hMτ']

omit [CompleteSpace H] in
/-- **Strict Cauchy–Schwarz at the midpoint, by hand.**  `M(mid)² < M(τ)·M(τ′)` for `τ ≠ τ′`:
with `f = e^{(⟪τ,X⟫−q)/2}`, `g = e^{(⟪τ′,X⟫−q)/2}` and `c = M(mid)/M(τ′)`, the expansion
`0 ≤ ∫(f − cg)² = M(τ) − 2c·M(mid) + c²·M(τ′)` gives `M(mid)² ≤ M(τ)M(τ′)`; equality would force
`f = c·g` a.e., i.e. `⟪τ−τ′, X a⟫` a.e. equal to the constant `2 log c` — excluded by the
affine-span condition `hspan`. -/
lemma sq_M_midpoint_lt (hμ : μ ≠ 0)
    (hInt : ∀ τ : H, Integrable (fun a => Real.exp (⟪τ, X a⟫_ℝ - q a)) μ)
    (hspan : ∀ v : H, v ≠ 0 → ¬∃ c : ℝ, ∀ᵐ a ∂μ, ⟪v, X a⟫_ℝ = c)
    {τ τ' : H} (hττ' : τ ≠ τ') :
    (M μ X q ((1/2 : ℝ) • τ + (1/2 : ℝ) • τ')) ^ 2 < M μ X q τ * M μ X q τ' := by
  set m : H := (1/2 : ℝ) • τ + (1/2 : ℝ) • τ' with hm
  -- pointwise product identities
  have hff : ∀ a : α, Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2)
      = Real.exp (⟪τ, X a⟫_ℝ - q a) := fun a => by
    rw [← Real.exp_add]; congr 1; ring
  have hgg : ∀ a : α, Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)
      = Real.exp (⟪τ', X a⟫_ℝ - q a) := fun a => by
    rw [← Real.exp_add]; congr 1; ring
  have hfg : ∀ a : α, Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)
      = Real.exp (⟪m, X a⟫_ℝ - q a) := fun a => by
    rw [← Real.exp_add]
    congr 1
    rw [hm]
    simp only [inner_add_left, real_inner_smul_left]
    ring
  -- integrability of the three products
  have hIff : Integrable
      (fun a => Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2)) μ := by
    rw [funext hff]; exact hInt τ
  have hIgg : Integrable
      (fun a => Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)) μ := by
    rw [funext hgg]; exact hInt τ'
  have hIfg : Integrable
      (fun a => Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)) μ := by
    rw [funext hfg]; exact hInt m
  -- the three integral values
  have i1 : ∫ a, Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) ∂μ
      = M μ X q τ := integral_congr_ae (Eventually.of_forall hff)
  have i2 : ∫ a, Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2) ∂μ
      = M μ X q m := integral_congr_ae (Eventually.of_forall hfg)
  have i3 : ∫ a, Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2) ∂μ
      = M μ X q τ' := integral_congr_ae (Eventually.of_forall hgg)
  have hA := M_pos (hInt τ) hμ
  have hB := M_pos (hInt τ') hμ
  have hC := M_pos (hInt m) hμ
  set c : ℝ := M μ X q m / M μ X q τ' with hc
  -- the quadratic expansion of ∫ (f − c·g)²
  have hexp : (fun a =>
      (Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) - c * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)) ^ 2)
      = fun a =>
        (Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2)
          - (2*c) * (Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)))
        + c^2 * (Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)) := by
    funext a; ring
  have h1 : Integrable (fun a =>
      Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2)
        - (2*c) * (Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2))) μ :=
    hIff.sub' (hIfg.const_mul (2*c))
  have h2 : Integrable (fun a =>
      c^2 * (Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2) * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2))) μ :=
    hIgg.const_mul (c^2)
  have hIsq : Integrable (fun a =>
      (Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) - c * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)) ^ 2) μ := by
    rw [hexp]; exact h1.fun_add h2
  have hval : ∫ a,
      (Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) - c * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)) ^ 2 ∂μ
      = M μ X q τ - 2*c*(M μ X q m) + c^2*(M μ X q τ') := by
    rw [hexp, integral_add h1 h2, integral_sub hIff (hIfg.const_mul (2*c)),
      integral_const_mul, integral_const_mul, i1, i2, i3]
  have hnn : (0:ℝ) ≤ M μ X q τ - 2*c*(M μ X q m) + c^2*(M μ X q τ') := by
    rw [← hval]; exact integral_nonneg fun a => sq_nonneg _
  have hkey : M μ X q τ * M μ X q τ' - (M μ X q m)^2
      = (M μ X q τ') * (M μ X q τ - 2*c*(M μ X q m) + c^2*(M μ X q τ')) := by
    rw [hc]; field_simp; ring
  have hprod : (0:ℝ) ≤ (M μ X q τ') * (M μ X q τ - 2*c*(M μ X q m) + c^2*(M μ X q τ')) :=
    mul_nonneg hB.le hnn
  have hle : (M μ X q m)^2 ≤ M μ X q τ * M μ X q τ' := by linarith
  -- strictness: equality would force `f = c·g` a.e., contradicting the affine span
  refine lt_of_le_of_ne hle (fun heq => ?_)
  have h7 : (M μ X q τ') * (M μ X q τ - 2*c*(M μ X q m) + c^2*(M μ X q τ')) = 0 := by linarith
  have h8 : M μ X q τ - 2*c*(M μ X q m) + c^2*(M μ X q τ') = 0 := by
    rcases mul_eq_zero.mp h7 with h | h
    · exact absurd h hB.ne'
    · exact h
  have h9 : ∫ a,
      (Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) - c * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)) ^ 2 ∂μ = 0 := by
    rw [hval]; exact h8
  have h10 := (integral_eq_zero_iff_of_nonneg (fun a => sq_nonneg _) hIsq).mp h9
  have h11 : ∀ᵐ a ∂μ,
      Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2) = c * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2) := by
    filter_upwards [h10] with a ha
    have ha' : (Real.exp ((⟪τ, X a⟫_ℝ - q a) / 2)
        - c * Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2)) ^ 2 = 0 := by simpa using ha
    exact sub_eq_zero.mp ((pow_eq_zero_iff (by norm_num)).mp ha')
  haveI : (MeasureTheory.ae μ).NeBot := ae_neBot.mpr hμ
  obtain ⟨a₀, ha₀⟩ := h11.exists
  have hc0 : 0 < c := by
    nlinarith [Real.exp_pos ((⟪τ, X a₀⟫_ℝ - q a₀) / 2), Real.exp_pos ((⟪τ', X a₀⟫_ℝ - q a₀) / 2),
      ha₀]
  have h12 : ∀ᵐ a ∂μ, ⟪τ - τ', X a⟫_ℝ = 2 * Real.log c := by
    filter_upwards [h11] with a ha
    have hE2 : (0:ℝ) < Real.exp ((⟪τ', X a⟫_ℝ - q a) / 2) := Real.exp_pos _
    have hlog := congrArg Real.log ha
    rw [Real.log_exp, Real.log_mul hc0.ne' hE2.ne', Real.log_exp] at hlog
    rw [inner_sub_left]
    linarith
  exact hspan (τ - τ') (sub_ne_zero.mpr hττ') ⟨2 * Real.log c, h12⟩

omit [CompleteSpace H] in
/-- The strict **midpoint** inequality for `Λ = log M`: taking logarithms in
`M(mid)² < M(τ)·M(τ′)`. -/
lemma logPartition_midpoint_lt (hμ : μ ≠ 0)
    (hInt : ∀ τ : H, Integrable (fun a => Real.exp (⟪τ, X a⟫_ℝ - q a)) μ)
    (hspan : ∀ v : H, v ≠ 0 → ¬∃ c : ℝ, ∀ᵐ a ∂μ, ⟪v, X a⟫_ℝ = c)
    {τ τ' : H} (hττ' : τ ≠ τ') :
    Λ μ X q ((1/2 : ℝ) • τ + (1/2 : ℝ) • τ') < (Λ μ X q τ + Λ μ X q τ') / 2 := by
  have hA := M_pos (hInt τ) hμ
  have hB := M_pos (hInt τ') hμ
  have hC := M_pos (hInt ((1/2 : ℝ) • τ + (1/2 : ℝ) • τ')) hμ
  have hsq := sq_M_midpoint_lt hμ hInt hspan hττ'
  have hlog := Real.log_lt_log (by positivity) hsq
  rw [Real.log_pow, Real.log_mul hA.ne' hB.ne'] at hlog
  push_cast at hlog
  show Real.log (M μ X q ((1/2 : ℝ) • τ + (1/2 : ℝ) • τ'))
    < (Real.log (M μ X q τ) + Real.log (M μ X q τ')) / 2
  linarith

omit [CompleteSpace H] in
/-- **(C2) The log-partition is strictly convex** under the affine-span condition — from
integrability at every parameter (no Hessian, no second differentiation): Hölder gives convexity,
the by-hand Cauchy–Schwarz equality analysis gives the strict midpoint, and the midpoint
bootstrap upgrades to strict convexity. -/
theorem strictConvexOn_logPartition_of_integrable (hμ : μ ≠ 0)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    (hInt : ∀ τ : H, Integrable (fun a => Real.exp (⟪τ, X a⟫_ℝ - q a)) μ)
    (hspan : ∀ v : H, v ≠ 0 → ¬∃ c : ℝ, ∀ᵐ a ∂μ, ⟪v, X a⟫_ℝ = c) :
    StrictConvexOn ℝ Set.univ (Λ μ X q) :=
  strictConvexOn_of_convexOn_of_midpoint (convexOn_logPartition hμ hX hq hInt)
    (fun _ _ hττ' => logPartition_midpoint_lt hμ hInt hspan hττ')

omit [CompleteSpace H] in
/-- **(C2), from the model's own hypotheses**: on a finite nonzero measure with a variationally
positive-definite quadratic tilt, the log-partition is strictly convex exactly when no direction
has a.e.-constant resolved coordinate — the paper's affine-span condition. -/
theorem strictConvexOn_logPartition [IsFiniteMeasure μ] (hμ : μ ≠ 0)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    {q0 : ℝ} (hq0 : 0 < q0) (hql : ∀ᵐ a ∂μ, q0 * ‖X a‖ ^ 2 / 2 ≤ q a)
    (hspan : ∀ v : H, v ≠ 0 → ¬∃ c : ℝ, ∀ᵐ a ∂μ, ⟪v, X a⟫_ℝ = c) :
    StrictConvexOn ℝ Set.univ (Λ μ X q) :=
  strictConvexOn_logPartition_of_integrable hμ hX hq
    (integrable_exp_tilt hX hq hq0 hql) hspan

end Convexity

/-! ## (C3) The capstone: `p:mean`'s collapse from the model's hypotheses only -/

section Capstone

/-- **(C3) `p:mean`'s collapse, closed.**  Hypotheses: the archive measure `μ` (finite, nonzero),
measurability of the resolved coordinate `X` and tilt `q`, the first moment `𝔼‖X‖ < ∞`, the
variationally positive-definite quadratic lower bound on the tilt, and the affine-span condition
— plus the preamble's structural relations `hfac`/`hres`.  Both the envelope (`dom`) and the
strict convexity (`hconv`) that `LogPartitionGradient.posterior_mean_collapse_of_logPartition`
consumed as hypotheses are now discharged: equal resolved components force equal
reconstructions — the blind-fiber conditional is a Dirac. -/
theorem posterior_mean_collapse_closed
    {𝓨 W : Type*} (xhat : 𝓨 → W) (T : 𝓨 → H) (ψ : H → W) (PR : W → H)
    (X : α → H) (q : α → ℝ) [IsFiniteMeasure μ] (hμ : μ ≠ 0)
    (hX : AEStronglyMeasurable X μ) (hq : AEStronglyMeasurable q μ)
    (hX1 : Integrable (fun a => ‖X a‖) μ)
    {q0 : ℝ} (hq0 : 0 < q0) (hql : ∀ᵐ a ∂μ, q0 * ‖X a‖ ^ 2 / 2 ≤ q a)
    (hspan : ∀ v : H, v ≠ 0 → ¬∃ c : ℝ, ∀ᵐ a ∂μ, ⟪v, X a⟫_ℝ = c)
    (hfac : ∀ y, xhat y = ψ (T y))
    (hres : ∀ y, PR (xhat y) = condMean μ X q (T y))
    {y y' : 𝓨} (h : PR (xhat y) = PR (xhat y')) :
    xhat y = xhat y' :=
  posterior_mean_collapse_of_logPartition xhat T ψ PR X q hX hq hμ
    (dom_of_moment hX hq hX1 hq0 hql)
    (strictConvexOn_logPartition hμ hX hq hq0 hql hspan) hfac hres h

/-- **(C3), with the paper's concrete quadratic tilt** `q(a) = ⟪X a, Q (X a)⟫/2` for an operator
`Q` that is variationally positive definite (`∀ x, q0‖x‖² ≤ ⟪x, Qx⟫` — the operational form of
`Q ≻ 0`, as in the paper's `q(x_R) = (Ax)ᵀΓ⁻¹Ax` on the resolved subspace).  Measurability of the
tilt and the quadratic lower bound are derived, not assumed. -/
theorem posterior_mean_collapse_quadratic
    {𝓨 W : Type*} (xhat : 𝓨 → W) (T : 𝓨 → H) (ψ : H → W) (PR : W → H)
    (X : α → H) (Q : H →L[ℝ] H) [IsFiniteMeasure μ] (hμ : μ ≠ 0)
    (hX : AEStronglyMeasurable X μ)
    (hX1 : Integrable (fun a => ‖X a‖) μ)
    {q0 : ℝ} (hq0 : 0 < q0) (hQ : ∀ x : H, q0 * ‖x‖ ^ 2 ≤ ⟪x, Q x⟫_ℝ)
    (hspan : ∀ v : H, v ≠ 0 → ¬∃ c : ℝ, ∀ᵐ a ∂μ, ⟪v, X a⟫_ℝ = c)
    (hfac : ∀ y, xhat y = ψ (T y))
    (hres : ∀ y, PR (xhat y) = condMean μ X (fun a => ⟪X a, Q (X a)⟫_ℝ / 2) (T y))
    {y y' : 𝓨} (h : PR (xhat y) = PR (xhat y')) :
    xhat y = xhat y' := by
  have hcont : Continuous fun x : H => ⟪x, Q x⟫_ℝ := continuous_id.inner Q.continuous
  have hinner : AEStronglyMeasurable (fun a => ⟪X a, Q (X a)⟫_ℝ) μ :=
    hcont.comp_aestronglyMeasurable hX
  have hq : AEStronglyMeasurable (fun a => ⟪X a, Q (X a)⟫_ℝ / 2) μ := by
    have h : (fun a => ⟪X a, Q (X a)⟫_ℝ / 2) = fun a => (2:ℝ)⁻¹ * ⟪X a, Q (X a)⟫_ℝ := by
      funext a; ring
    rw [h]
    exact hinner.const_mul _
  have hql : ∀ᵐ a ∂μ, q0 * ‖X a‖ ^ 2 / 2 ≤ ⟪X a, Q (X a)⟫_ℝ / 2 :=
    Eventually.of_forall fun a => by have := hQ (X a); linarith
  exact posterior_mean_collapse_closed xhat T ψ PR X
    (fun a => ⟪X a, Q (X a)⟫_ℝ / 2) hμ hX hq hX1 hq0 hql hspan hfac hres h

end Capstone

/-! ## (C4) The Gaussian aggregate `𝔼_Z[Φ(a + bZ)] = Φ(a/√(1+b²))` (`r:map-lg` / `p:cover-na`) -/

section GaussianAggregate

/-- Standardized `Iic`-probability of a centered Gaussian: for `v ≠ 0`,
`𝒩(0,v)((-∞, x]) = Φ(x/√v)`. -/
lemma gaussianReal_real_Iic {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    (gaussianReal (0:ℝ) v).real (Set.Iic x) = GaussianCDF.Φ (x / Real.sqrt v) := by
  have hvpos : (0:ℝ) < (v:ℝ) := by exact_mod_cast pos_iff_ne_zero.mpr hv
  have hs : (0:ℝ) < Real.sqrt v := Real.sqrt_pos.mpr hvpos
  have hmap := GaussianCDF.stdGaussian_map (0:ℝ) hv
  rw [GaussianCDF.Φ_eq_real, ← hmap, measureReal_def, measureReal_def,
    Measure.map_apply (by fun_prop) measurableSet_Iic]
  congr 2
  ext y
  simp only [Set.mem_Iic, Set.mem_preimage, sub_zero]
  exact (div_le_div_iff_of_pos_right hs).symm

/-- **(C4) The Gaussian aggregate identity** used by `r:map-lg` for `p:cover-na`:
for a standard normal `Z`,

    𝔼_Z[Φ(a + b·Z)] = Φ(a / √(1 + b²)).

Route: `Φ(a + bz) = 𝒩(0,1){w ∣ w − bz ≤ a}`; Fubini identifies the average with
`(𝒩(0,1) ⊗ 𝒩(0,1)){(z,w) ∣ w − bz ≤ a}`, the pushforward under `(z,w) ↦ w − bz` is the
convolution `𝒩(0,b²) ∗ 𝒩(0,1) = 𝒩(0, b² + 1)`, and standardization finishes. -/
theorem integral_Phi_affine (a b : ℝ) :
    ∫ z, GaussianCDF.Φ (a + b * z) ∂(gaussianReal (0:ℝ) 1)
      = GaussianCDF.Φ (a / Real.sqrt (1 + b ^ 2)) := by
  have hmeas : AEStronglyMeasurable (fun z : ℝ => GaussianCDF.Φ (a + b * z))
      (gaussianReal (0:ℝ) 1) :=
    (GaussianCDF.continuous_Φ.comp
      (by fun_prop : Continuous fun z : ℝ => a + b * z)).aestronglyMeasurable
  rw [integral_eq_lintegral_of_nonneg_ae
    (Eventually.of_forall fun z => GaussianCDF.Φ_nonneg _) hmeas]
  have hSm : MeasurableSet {p : ℝ × ℝ | p.2 - b * p.1 ≤ a} :=
    measurableSet_le (measurable_snd.sub (measurable_fst.const_mul b)) measurable_const
  have hS : Measurable fun p : ℝ × ℝ => p.2 - b * p.1 :=
    measurable_snd.sub (measurable_fst.const_mul b)
  have h2 : (∫⁻ z, ENNReal.ofReal (GaussianCDF.Φ (a + b * z)) ∂(gaussianReal (0:ℝ) 1))
      = (gaussianReal (0:ℝ) (NNReal.mk ((-b)^2) (sq_nonneg _) * 1 + 1)) (Set.Iic a) := by
    calc ∫⁻ z, ENNReal.ofReal (GaussianCDF.Φ (a + b * z)) ∂(gaussianReal (0:ℝ) 1)
        = ∫⁻ z, (gaussianReal (0:ℝ) 1) {w | w - b * z ≤ a} ∂(gaussianReal (0:ℝ) 1) := by
          refine lintegral_congr fun z => ?_
          rw [GaussianCDF.Φ_eq_real, measureReal_def,
            ENNReal.ofReal_toReal (measure_ne_top _ _)]
          congr 1
          ext w
          simp
      _ = ((gaussianReal (0:ℝ) 1).prod (gaussianReal (0:ℝ) 1))
            {p : ℝ × ℝ | p.2 - b * p.1 ≤ a} := by
          rw [Measure.prod_apply hSm]
          rfl
      _ = (((gaussianReal (0:ℝ) 1).prod (gaussianReal (0:ℝ) 1)).map
            (fun p : ℝ × ℝ => p.2 - b * p.1)) (Set.Iic a) := by
          rw [Measure.map_apply hS measurableSet_Iic]
          rfl
      _ = (((gaussianReal (0:ℝ) 1).map (fun z : ℝ => -b * z))
            ∗ (gaussianReal (0:ℝ) 1)) (Set.Iic a) := by
          congr 1
          have hmb : Measurable fun z : ℝ => -b * z := by fun_prop
          have hcomp : (fun p : ℝ × ℝ => p.2 - b * p.1)
              = (fun p : ℝ × ℝ => p.1 + p.2)
                ∘ Prod.map (fun z : ℝ => -b * z) (id : ℝ → ℝ) := by
            funext p
            simp only [Function.comp_apply, Prod.map_fst, Prod.map_snd, id_eq]
            ring
          rw [hcomp, ← Measure.map_map (measurable_fst.add measurable_snd)
              (hmb.prodMap measurable_id),
            ← Measure.map_prod_map _ _ hmb measurable_id,
            Measure.map_id]
          rfl
      _ = (gaussianReal (0:ℝ) (NNReal.mk ((-b)^2) (sq_nonneg _) * 1 + 1)) (Set.Iic a) := by
          have h0 : ((-b) * 0 + 0 : ℝ) = 0 := by ring
          rw [gaussianReal_map_const_mul (-b), gaussianReal_conv_gaussianReal, h0]
  rw [h2, ← measureReal_def]
  have hv : (NNReal.mk ((-b)^2) (sq_nonneg _) * 1 + 1 : ℝ≥0) ≠ 0 :=
    (one_pos.trans_le le_add_self).ne'
  rw [gaussianReal_real_Iic hv]
  have hcoe : ((NNReal.mk ((-b)^2) (sq_nonneg _) * 1 + 1 : ℝ≥0) : ℝ) = 1 + b ^ 2 := by
    rw [NNReal.coe_add, NNReal.coe_mul, NNReal.coe_one, NNReal.coe_mk]
    ring
  rw [hcoe]

end GaussianAggregate

end LogPartitionClosure

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms LogPartitionClosure.strictConvexOn_of_convexOn_of_midpoint
#print axioms LogPartitionClosure.exp_tilt_le
#print axioms LogPartitionClosure.integrable_exp_tilt
#print axioms LogPartitionClosure.dom_of_moment
#print axioms LogPartitionClosure.hasGradientAt_logPartition_of_moment
#print axioms LogPartitionClosure.convexOn_logPartition
#print axioms LogPartitionClosure.sq_M_midpoint_lt
#print axioms LogPartitionClosure.logPartition_midpoint_lt
#print axioms LogPartitionClosure.strictConvexOn_logPartition_of_integrable
#print axioms LogPartitionClosure.strictConvexOn_logPartition
#print axioms LogPartitionClosure.posterior_mean_collapse_closed
#print axioms LogPartitionClosure.posterior_mean_collapse_quadratic
#print axioms LogPartitionClosure.gaussianReal_real_Iic
#print axioms LogPartitionClosure.integral_Phi_affine
