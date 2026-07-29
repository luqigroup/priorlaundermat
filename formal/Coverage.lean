import GaussianCDF
import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Proposition `p:cover` of the paper: the closed-form coverage shortfall

Machine-checks **Proposition `p:cover`** ("A closed-form coverage shortfall", linear–Gaussian).

By the freeze (`p:blind`) the blind functional `θ = v⊤x` has reported law `𝒩(m_reg, s_reg²)`; scored
against the true law `𝒩(m_⋆, s_⋆²)` the advertised central interval `m_reg ± z·s_reg`
(`z = z_{1−α/2}`) covers the truth with probability, in the standardized gap `δ = (m_reg − m_⋆)/s_⋆`
and width ratio `r = z·s_reg/s_⋆`,

  `C = Φ(δ + r) − Φ(δ − r)`   (eq:coverage-gen),   `C = 2Φ(r) − 1` at `δ = 0` (eq:coverage),

and it is **overconfident** (`C < 1 − α`) exactly when the inherited width is tighter than the
truth's, `s_reg < s_⋆`.

## Results

* `coverage_eq` — the closed form `𝒩(m_⋆, s_⋆²)(|θ − m_reg| ≤ z s_reg) = C(δ, r)`.
* `C_zero` — `C(0, r) = 2 Φ(r) − 1` (the aligned form; the nominal level is `C(0, z) = 2Φ(z) − 1`).
* `overconfident_of_tighter` — `s_reg < s_⋆ ⟹ C(0, z s_reg/s_⋆) < C(0, z)`: tighter-than-truth width
  ⟹ actual coverage strictly below nominal.

The monotonicity `∂C/∂δ < 0` (a frozen mean error only lowers coverage) is the density-derivative
layer, flagged in the reconciliation.

## Honesty

No `sorry`/`admit`.  `#print axioms` on each main theorem lists only `propext`, `Classical.choice`,
`Quot.sound` (no `sorryAx`).  Reconciliation (statement + proof level) at the end.
-/

open MeasureTheory ProbabilityTheory Set GaussianCDF
open scoped NNReal

namespace Coverage

/-- The **coverage** of the level-`(1−α)` interval `m_reg ± z·s_reg` scored against `𝒩(m_⋆, s_⋆²)`,
as a function of the standardized gap `δ` and width ratio `r`: `C(δ, r) = Φ(δ + r) − Φ(δ − r)`. -/
noncomputable def C (δ r : ℝ) : ℝ := Φ (δ + r) - Φ (δ - r)

/-- **`p:cover` — the closed-form coverage** (eq:coverage-gen). The advertised interval
`m_reg ± z·s_reg` (`z, s_reg ≥ 0`), scored against the truth `𝒩(m_⋆, s_⋆²)` (variance `vstar ≠ 0`,
`s_⋆ = √vstar`), covers with probability `C(δ, r)`, `δ = (m_reg − m_⋆)/s_⋆`, `r = z·s_reg/s_⋆`. -/
theorem coverage_eq (m_reg m_star z sreg : ℝ) {vstar : ℝ≥0} (hv : vstar ≠ 0)
    (hz : 0 ≤ z) (hsreg : 0 ≤ sreg) :
    (gaussianReal m_star vstar).real (Icc (m_reg - z * sreg) (m_reg + z * sreg))
      = C ((m_reg - m_star) / Real.sqrt vstar) (z * sreg / Real.sqrt vstar) := by
  have hab : m_reg - z * sreg ≤ m_reg + z * sreg := by nlinarith [mul_nonneg hz hsreg]
  have h1 : (m_reg + z * sreg - m_star) / Real.sqrt vstar
      = (m_reg - m_star) / Real.sqrt vstar + z * sreg / Real.sqrt vstar := by
    rw [← add_div]; congr 1; ring
  have h2 : (m_reg - z * sreg - m_star) / Real.sqrt vstar
      = (m_reg - m_star) / Real.sqrt vstar - z * sreg / Real.sqrt vstar := by
    rw [← sub_div]; congr 1; ring
  rw [gaussian_real_Icc m_star hv hab, C, h1, h2]

/-- `C(0, r) = 2 Φ(r) − 1` (aligned, `δ = 0`; eq:coverage). -/
lemma C_zero (r : ℝ) : C 0 r = 2 * Φ r - 1 := by
  rw [C, zero_add, zero_sub, Φ_neg]; ring

/-- **The coverage is strictly below nominal exactly when the prior is tighter than the truth.**
For a genuine level (`0 < z`) and `0 < s_⋆`, an inherited width `s_reg < s_⋆` makes the aligned
coverage `C(0, z s_reg/s_⋆)` strictly below the nominal `C(0, z) = 2Φ(z) − 1 = 1 − α`: the advertised
interval under-covers.  (Overconfidence `C < 1 − α ⟺ s_reg < s_⋆`.) -/
theorem overconfident_of_tighter {z sstar sreg : ℝ} (hz : 0 < z) (hs : 0 < sstar)
    (hlt : sreg < sstar) :
    C 0 (z * sreg / sstar) < C 0 z := by
  have hr : z * sreg / sstar < z := by
    rw [div_lt_iff₀ hs]; exact mul_lt_mul_of_pos_left hlt hz
  rw [C, C, zero_add, zero_sub, zero_add, zero_sub]
  have h1 : Φ (z * sreg / sstar) < Φ z := strictMono_Φ hr
  have h2 : Φ (-z) < Φ (-(z * sreg / sstar)) := strictMono_Φ (by linarith)
  linarith

/-- The `δ`-derivative of the coverage: `∂C/∂δ = φ(δ + r) − φ(δ − r)`. -/
lemma hasDerivAt_C (r δ : ℝ) : HasDerivAt (fun δ => C δ r) (φ (δ + r) - φ (δ - r)) δ := by
  have h1 : HasDerivAt (fun δ => Φ (δ + r)) (φ (δ + r)) δ := by
    simpa [Function.comp_def] using (hasDerivAt_Φ (δ + r)).comp δ ((hasDerivAt_id δ).add_const r)
  have h2 : HasDerivAt (fun δ => Φ (δ - r)) (φ (δ - r)) δ := by
    simpa [Function.comp_def] using (hasDerivAt_Φ (δ - r)).comp δ ((hasDerivAt_id δ).sub_const r)
  exact h1.sub h2

/-- **`∂C/∂δ < 0` — a frozen mean error only worsens the shortfall.** For `r > 0`, the coverage is
strictly decreasing in the standardized mean gap for `δ > 0`: `C(δ, r) < C(0, r)`.  (The centered
interval maximizes coverage; any inherited-mean error `δ ≠ 0` lowers it, by the mean value theorem
and `φ` decreasing in `|·|`, since `|δ + r| > |δ − r|`.) -/
theorem C_lt_of_pos {r δ : ℝ} (hr : 0 < r) (hδ : 0 < δ) : C δ r < C 0 r := by
  obtain ⟨c, hc, hslope⟩ := exists_hasDerivAt_eq_slope (fun δ => C δ r)
    (fun δ => φ (δ + r) - φ (δ - r)) hδ
    (continuous_iff_continuousAt.mpr fun δ => (hasDerivAt_C r δ).continuousAt).continuousOn
    (fun δ _ => hasDerivAt_C r δ)
  have hc0 : 0 < c := hc.1
  have hcr : |c - r| < |c + r| := by
    rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ c + r), abs_lt]
    constructor <;> linarith
  have hφlt : φ (c + r) - φ (c - r) < 0 := by linarith [φ_lt_of_abs_lt hcr]
  rw [hslope, sub_zero] at hφlt
  have hnum : C δ r - C 0 r < 0 := by
    rcases div_neg_iff.mp hφlt with ⟨h1, h2⟩ | ⟨h1, _⟩
    · linarith
    · exact h1
  linarith

/-- **The smoothing floor vanishes** (`r:map-lg`): as the smoothing scale `→ 0` (so the width ratio
`r = z h c_K / s_⋆ → 0`), the advertised coverage `C(0, r) = 2Φ(r) − 1 → 0`.  A generative estimator
fit to the `MAP` point mass reports a blind coverage set by its own smoothing scale, vanishing with
it. -/
theorem tendsto_C_zero_atZero : Filter.Tendsto (fun r => C 0 r) (nhds 0) (nhds 0) := by
  have h : Filter.Tendsto (fun r => 2 * Φ r - 1) (nhds 0) (nhds (2 * Φ 0 - 1)) :=
    ((continuous_Φ.tendsto 0).const_mul 2).sub_const 1
  rw [Φ_zero] at h
  norm_num at h
  simpa only [C_zero] using h

/-- `C` is **even** in the standardized gap: `C(−δ, r) = C(δ, r)`.  Reflecting the truth's
conditional mean about the advertised center leaves the covered mass unchanged, because the
standard normal is symmetric (`Φ(−x) = 1 − Φ(x)`). -/
lemma C_neg (δ r : ℝ) : C (-δ) r = C δ r := by
  have h1 : -δ + r = -(δ - r) := by ring
  have h2 : -δ - r = -(δ + r) := by ring
  simp only [C, h1, h2, Φ_neg]
  ring

/-- **`p:cover-na` — the distribution-free cap.**  For any width ratio `r > 0` and **any**
standardized gap `δ`, the fiber-conditional coverage is at most its aligned value,
`C(δ, r) ≤ C(0, r) = 2Φ(r) − 1`.  Nothing is assumed about how `δ` arises: the conditional-mean maps
may be affine or not, and the readout may be distributed however the truth likes.  This is what makes
the cap survive every aggregation over readouts. -/
theorem C_le_C_zero {r : ℝ} (hr : 0 < r) (δ : ℝ) : C δ r ≤ C 0 r := by
  rcases lt_trichotomy δ 0 with h | h | h
  · have h' : C (-δ) r < C 0 r := C_lt_of_pos hr (neg_pos.mpr h)
    rw [C_neg] at h'
    exact h'.le
  · subst h; exact le_rfl
  · exact (C_lt_of_pos hr h).le

/-- **Equality in the cap holds exactly at alignment** (`δ = 0`). -/
theorem C_eq_C_zero_iff {r : ℝ} (hr : 0 < r) (δ : ℝ) : C δ r = C 0 r ↔ δ = 0 := by
  refine ⟨fun hEq => ?_, fun h => by subst h; rfl⟩
  by_contra hne
  rcases lt_or_gt_of_ne hne with h | h
  · have h' : C (-δ) r < C 0 r := C_lt_of_pos hr (neg_pos.mpr h)
    rw [C_neg] at h'
    exact absurd hEq (ne_of_lt h')
  · exact absurd hEq (ne_of_lt (C_lt_of_pos hr h))

/-- **The cap survives aggregation.**  Since `C(δ, r) ≤ C(0, r)` holds at *every* readout, any
average of per-readout coverages over a finite set of held-out truths obeys the same cap.  (The
docstring above claims the cap "survives every aggregation"; this is that claim, proved, for the
finite average the experiments actually report.  The paper's closed *form* for the aggregate in the
linear--Gaussian case is a Gaussian-convolution identity and is not formalized.) -/
theorem C_average_le {r : ℝ} (hr : 0 < r) {ι : Type*} (s : Finset ι) (hs : s.Nonempty)
    (δ : ι → ℝ) :
    (∑ i ∈ s, C (δ i) r) / s.card ≤ C 0 r := by
  have hcard : (0:ℝ) < s.card := by
    exact_mod_cast Finset.card_pos.mpr hs
  rw [div_le_iff₀ hcard]
  calc ∑ i ∈ s, C (δ i) r ≤ ∑ _i ∈ s, C 0 r :=
        Finset.sum_le_sum fun i _ => C_le_C_zero hr (δ i)
    _ = C 0 r * s.card := by rw [Finset.sum_const, nsmul_eq_mul]; ring

/-- **Below nominal at every readout, with no alignment assumed.**  If the inherited width is
tighter than the truth's (`r < z`), then at **every** readout — whatever the standardized gap `δ`
there — the fiber-conditional coverage is strictly below the advertised level `C(0, z) = 1 − α`.
Since the bound is uniform in `δ`, it passes to any average over readouts. -/
theorem C_lt_nominal {z r : ℝ} (hr : 0 < r) (hrz : r < z) (δ : ℝ) : C δ r < C 0 z := by
  refine lt_of_le_of_lt (C_le_C_zero hr δ) ?_
  rw [C_zero, C_zero]
  have := strictMono_Φ hrz
  linarith

/-!
## Reconciliation — what `p:cover`'s formalization needs vs. what the paper states

**(a)** already in the text; **(b)** load-bearing analytic layer taken as hypothesis / not
formalized; **(c)** automatic.

### Statement level

* The truth's blind functional is `𝒩(m_⋆, s_⋆²)` and the report is the frozen `𝒩(m_reg, s_reg²)`
  — **(a)**: this is the freeze `p:blind` fed into `p:cover` (the report's law is the regularizer's,
  fixed).  Here `p:cover` is stated directly at the level of the truth's Gaussian law, so the freeze
  enters as the choice of the reported center/width `(m_reg, s_reg)`; the coverage formula itself is
  a fact about two Gaussians, exactly what is proved.
* `vstar ≠ 0` (nondegenerate truth), `z, s_reg ≥ 0` — **(c)**: a genuine truth has positive variance;
  `z = z_{1−α/2} > 0` and a spread `s_reg ≥ 0`.  `z > 0`, `s_⋆ > 0` in `overconfident_of_tighter`.
* nominal `= 2Φ(z) − 1 = 1 − α` — **(a)**: `C(0, z) = 2Φ(z) − 1` (`C_zero`) is the coverage of a
  correctly specified interval, i.e. the advertised level `1 − α` by the definition of `z`.

### Proof level (Lean proof vs. the appendix)

The appendix writes `C = Pr_{𝒩(m_⋆,s_⋆²)}(|θ − m_reg| ≤ z s_reg)` and "by direct integration of the
mis-scaled Gaussian interval" reads off `Φ(δ ± r)`.  The Lean proof makes "direct integration"
precise and reusable, surfacing:

* **(b→closed) the change of variables is an explicit standardization.** `Φ(δ ± r)` comes from
  pushing `𝒩(m_⋆, s_⋆²)` to `𝒩(0, 1)` by `x ↦ (x − m_⋆)/s_⋆` (`GaussianCDF.stdGaussian_map`) and the
  interval identity `𝒩(0,1)((c, d]) = Φ(d) − Φ(c)` (`gaussianReal_real_Ioc`), with the endpoint
  `{c}` discarded by `NoAtoms`.  The appendix's one phrase "direct integration" is this whole layer,
  now a reusable `GaussianCDF` module (also serving the near-null bounds and the smoothing floor).
* **(a) the overconfidence sign needs only strict monotonicity, not the error function.** The paper
  argues overconfidence via `s_reg < s_⋆`; the Lean proof shows the entire content is
  `StrictMono Φ` (positive gaussian density) — `C(0, r) < C(0, z) ⟸ r < z`, no closed-form `Φ`
  needed.  `mathlib` has no `erf`; `Φ := cdf (gaussianReal 0 1)` and its order/symmetry properties
  are enough.
* **(a, now closed) `∂C/∂δ < 0`.** "A frozen mean error only worsens it" is now proved
  (`C_lt_of_pos`): the density layer added to `GaussianCDF` (`φ`, `HasDerivAt Φ φ` by the FTC, and
  `φ` strictly decreasing in `|·|`) gives `∂C/∂δ = φ(δ+r) − φ(δ−r)`, negative for `δ > 0` since
  `|δ+r| > |δ−r|`; the mean value theorem then yields `C(δ,r) < C(0,r)`.  **Proof level:** the paper's
  `∂C/∂δ = φ(δ+r) − φ(δ−r) < 0` "since `φ` is decreasing in `|·|`" is exactly this — the Lean proof
  pins that the sign needs only `φ`'s unimodality (`exp` ∘ `-x²/2`), no other Gaussian specifics.
* **The smoothing floor** (`r:map-lg`, `tendsto_C_zero_atZero`): `C(0, r) = 2Φ(r) − 1 → 0` as the
  width ratio `r → 0` (from `Φ` continuous, `Φ(0) = 1/2`) — the vanishing advertised coverage a
  generative estimator reports when it fits the `MAP` point mass with a smoothing scale `h → 0`.
-/

end Coverage

-- #print axioms audit (p:cover-na)
#print axioms Coverage.C_neg
#print axioms Coverage.C_le_C_zero
#print axioms Coverage.C_eq_C_zero_iff
#print axioms Coverage.C_lt_nominal
#print axioms Coverage.C_average_le
-- #print axioms audit (p:cover)
#print axioms Coverage.coverage_eq
#print axioms Coverage.overconfident_of_tighter
#print axioms Coverage.C_lt_of_pos
#print axioms Coverage.tendsto_C_zero_atZero
