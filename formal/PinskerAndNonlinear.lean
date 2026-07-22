import Mathlib
import NearNullLeCam

/-!
# Pinsker's inequality (built from scratch), the Gaussian KL crossover, and `r:nonlinear`

Three deliverables, layered on the pinned mathlib (which has `klDiv`/`klFun` but **no Pinsker
inequality** and **no KL tensorization**):

**(A) Pinsker.** For probability measures and every measurable event `A`,
`2 (P A − Q A)² ≤ KL(P‖Q)` and hence `|P A − Q A| ≤ √(KL/2)`. Since the statistical total
variation is the supremum of `|P A − Q A|` over events, this *is* `TV ≤ √(KL/2)`, in the setwise
form the Le Cam engine (`NearNullLeCam.lean`) consumes. Route: the scalar two-point inequality
`kl(p,q) ≥ 2(p−q)²` (one-variable calculus: the logit `log x − log(1−x) − 4x` is monotone on
`(0,1)` because `1/(x(1−x)) ≥ 4`), plus the two-set partition ("restricted data-processing") step
`q·klFun(p/q) + (1−q)·klFun((1−p)/(1−q)) ≤ KL`, obtained by applying mathlib's Jensen step
`mul_le_integral_rnDeriv_of_ac` to the restrictions `P|_A ≪ Q|_A` and `P|_{Aᶜ} ≪ Q|_{Aᶜ}`.

**(B) The Gaussian KL closed form and the paper's crossover** (`p:nearnull` (ii), `eq:nearnull-kl`,
`eq:nearnull-cross`). `KL(N(m₁,v) ‖ N(m₂,v)) = (m₁−m₂)²/(2v)` in 1-D; KL **tensorizes over finite
products** (`klDiv_pi`, proved here unconditionally from the chain rule via constant kernels and a
measurable-equivalence invariance — absent from the pinned mathlib); hence the whitened
multivariate mean-shift KL is `‖shift‖²/2` and the `n`-survey iid product has KL `n(m₁−m₂)²/(2v)`.
Crossovers via Pinsker + Le Cam: for the **spread** normalization of `eq:nearnull-kl`
(`KL ≤ t²/4`, from `½(t−log(1+t)) ≤ t²/4`, proved here) the paper's `n < n⋆ = 2/t²` gives summed
type-I+II error `> 1/2` — exactly `eq:nearnull-cross`. For the **mean** misreport the paper's exact
`KL = s²/2` per survey gives the crossover `n < 1/s²` by the same route. (Note: with `KL = t²/2`
per survey, `n < 2/t²` only gives error `≥ 1 − √(1/2)`, NOT `> 1/2`; the paper's `2/t²` constant
belongs to the spread bound `KL ≤ t²/4`, and the formal statements keep the two normalizations
distinct.)

**(C) `r:nonlinear`** (`eq:nonlinear-deriv`). For additive noise with a differentiable density and
a dominating envelope allowing differentiation under the expectation, the curation reweight
`t ↦ E_τ[ν(y − F(x₀ + t v))/ρ(y)]` has vanishing derivative at `t = 0` along every direction `v`
in the Jacobian null space (`J(x₀) v = 0`): a derivative at the point, exactly as the remark
claims.

## Explicit hypotheses (flagged, none hidden)
* Pinsker's measure-side statements take `klDiv P Q ≠ ∞` (equivalently `P ≪ Q` and `llr`
  integrable) where the real-valued form needs it; `ofReal_two_mul_sq_le_klDiv` is
  hypothesis-free (both sides are `∞`-monotone).
* `r:nonlinear` takes the paper's own hypotheses: differentiability of `F` along the line
  `x₀ + t v`, a differentiable density `ν`, measurability of the integrand, integrability at
  `t = 0`, and an integrable envelope dominating the pointwise `t`-derivative on a neighborhood —
  the "dominating envelope allowing differentiation under the expectation" of the remark.

## Honesty
No `sorry`/`admit`/`native_decide`; `#print axioms` on each public result lists only `propext`,
`Classical.choice`, `Quot.sound`.
-/

open MeasureTheory ProbabilityTheory InformationTheory Real Set
open scoped ENNReal NNReal

namespace NearNull

/-! ### (A.1) The scalar two-point inequality `kl(p,q) ≥ 2(p−q)²` -/

/-- The (shifted) logit `log x − log(1−x) − 4x`, whose monotonicity on `(0,1)` encodes
`kl''(p) ≥ 4`. -/
private noncomputable def klLogit : ℝ → ℝ := fun x => Real.log x - Real.log (1 - x) - 4 * x

private lemma klLogit_hasDerivAt {x : ℝ} (hx0 : 0 < x) (hx1 : x < 1) :
    HasDerivAt klLogit (x⁻¹ + (1 - x)⁻¹ - 4) x := by
  have h1x : (0:ℝ) < 1 - x := by linarith
  have hin : HasDerivAt (fun s : ℝ => 1 - s) (-1) x := by
    simpa using (hasDerivAt_id x).const_sub (1 : ℝ)
  have hlog1 : HasDerivAt Real.log x⁻¹ x := Real.hasDerivAt_log hx0.ne'
  have hlog2 : HasDerivAt (fun s : ℝ => Real.log (1 - s)) (-1 / (1 - x)) x :=
    hin.log h1x.ne'
  have hlin : HasDerivAt (fun s : ℝ => 4 * s) 4 x := by
    simpa using (hasDerivAt_id x).const_mul (4 : ℝ)
  exact ((hlog1.sub hlog2).sub hlin).congr_deriv (by ring)

private lemma klLogit_monotoneOn : MonotoneOn klLogit (Set.Ioo (0:ℝ) 1) := by
  apply monotoneOn_of_deriv_nonneg (convex_Ioo 0 1)
  · intro x hx
    exact ((klLogit_hasDerivAt hx.1 hx.2).differentiableAt.continuousAt).continuousWithinAt
  · intro x hx
    rw [interior_Ioo] at hx
    exact (klLogit_hasDerivAt hx.1 hx.2).differentiableAt.differentiableWithinAt
  · intro x hx
    rw [interior_Ioo] at hx
    rw [(klLogit_hasDerivAt hx.1 hx.2).deriv]
    have hx0 := hx.1
    have h1x : (0:ℝ) < 1 - x := by linarith [hx.2]
    have hkey : x⁻¹ + (1 - x)⁻¹ - 4 = (1 - 2 * x) ^ 2 / (x * (1 - x)) := by
      field_simp
      ring
    rw [hkey]
    positivity

/-- The gap `kl(p,q) − 2(p−q)²` as a function of `p`, for a fixed reference `q`. -/
private noncomputable def klGap (q : ℝ) : ℝ → ℝ := fun p =>
  p * Real.log p - p * Real.log q + (1 - p) * Real.log (1 - p) - (1 - p) * Real.log (1 - q)
    - 2 * (p - q) ^ 2

private lemma klGap_continuous' (q : ℝ) : Continuous (klGap q) := by
  have c1 : Continuous fun s : ℝ => s * Real.log s := Real.continuous_mul_log
  have c2 : Continuous fun s : ℝ => (1 - s) * Real.log (1 - s) := by
    have h : Continuous fun s : ℝ => (1:ℝ) - s := continuous_const.sub continuous_id
    exact Real.continuous_mul_log.comp h
  have c3 : Continuous fun s : ℝ => s * Real.log q := continuous_id.mul continuous_const
  have c4 : Continuous fun s : ℝ => (1 - s) * Real.log (1 - q) :=
    (continuous_const.sub continuous_id).mul continuous_const
  have c5 : Continuous fun s : ℝ => 2 * (s - q) ^ 2 :=
    continuous_const.mul ((continuous_id.sub continuous_const).pow 2)
  exact (((c1.sub c3).add c2).sub c4).sub c5

private lemma klGap_hasDerivAt (q : ℝ) {x : ℝ} (hx0 : 0 < x) (hx1 : x < 1) :
    HasDerivAt (klGap q) (klLogit x - klLogit q) x := by
  have h1x : (0:ℝ) < 1 - x := by linarith
  have hin : HasDerivAt (fun s : ℝ => 1 - s) (-1) x := by
    simpa using (hasDerivAt_id x).const_sub (1 : ℝ)
  have d1 : HasDerivAt (fun s : ℝ => s * Real.log s) (Real.log x + 1) x :=
    Real.hasDerivAt_mul_log hx0.ne'
  have d2 : HasDerivAt (fun s : ℝ => s * Real.log q) (Real.log q) x := by
    simpa using (hasDerivAt_id x).mul_const (Real.log q)
  have hval : (1 - x) * (-1 / (1 - x)) = -1 := by
    rw [← mul_div_assoc, mul_neg_one, neg_div, div_self h1x.ne']
  have d3 : HasDerivAt (fun s : ℝ => (1 - s) * Real.log (1 - s)) (-(Real.log (1 - x) + 1)) x := by
    refine (hin.mul (hin.log h1x.ne')).congr_deriv ?_
    rw [hval]
    ring
  have d4 : HasDerivAt (fun s : ℝ => (1 - s) * Real.log (1 - q)) (-Real.log (1 - q)) x := by
    simpa using hin.mul_const (Real.log (1 - q))
  have d5 : HasDerivAt (fun s : ℝ => 2 * (s - q) ^ 2) (2 * (2 * (x - q) ^ 1 * 1)) x := by
    have hbase : HasDerivAt (fun s : ℝ => s - q) 1 x := by
      simpa using (hasDerivAt_id x).sub_const q
    exact (hbase.pow 2).const_mul 2
  refine ((((d1.sub d2).add d3).sub d4).sub d5).congr_deriv ?_
  simp only [klLogit]
  ring

private lemma klGap_nonneg {q p : ℝ} (hq0 : 0 < q) (hq1 : q < 1) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ klGap q p := by
  have hq01 : q ∈ Set.Ioo (0:ℝ) 1 := ⟨hq0, hq1⟩
  have hGq : klGap q q = 0 := by
    simp only [klGap]
    ring
  rcases le_total p q with hpq | hqp
  · -- antitone on `[0, q]`, so `klGap q p ≥ klGap q q = 0`
    have hanti : AntitoneOn (klGap q) (Set.Icc 0 q) := by
      apply antitoneOn_of_deriv_nonpos (convex_Icc 0 q) (klGap_continuous' q).continuousOn
      · intro x hx
        rw [interior_Icc] at hx
        exact (klGap_hasDerivAt q hx.1 (hx.2.trans hq1)).differentiableAt.differentiableWithinAt
      · intro x hx
        rw [interior_Icc] at hx
        have hx01 : x ∈ Set.Ioo (0:ℝ) 1 := ⟨hx.1, hx.2.trans hq1⟩
        rw [(klGap_hasDerivAt q hx01.1 hx01.2).deriv]
        have := klLogit_monotoneOn hx01 hq01 hx.2.le
        linarith
    have h := hanti ⟨hp0, hpq⟩ ⟨hq0.le, le_refl q⟩ hpq
    rw [hGq] at h
    exact h
  · -- monotone on `[q, 1]`, so `klGap q p ≥ klGap q q = 0`
    have hmono : MonotoneOn (klGap q) (Set.Icc q 1) := by
      apply monotoneOn_of_deriv_nonneg (convex_Icc q 1) (klGap_continuous' q).continuousOn
      · intro x hx
        rw [interior_Icc] at hx
        exact (klGap_hasDerivAt q (hq0.trans hx.1) hx.2).differentiableAt.differentiableWithinAt
      · intro x hx
        rw [interior_Icc] at hx
        have hx01 : x ∈ Set.Ioo (0:ℝ) 1 := ⟨hq0.trans hx.1, hx.2⟩
        rw [(klGap_hasDerivAt q hx01.1 hx01.2).deriv]
        have := klLogit_monotoneOn hq01 hx01 hx.1.le
        linarith
    have h := hmono ⟨le_refl q, hq1.le⟩ ⟨hqp, hp1⟩ hqp
    rw [hGq] at h
    exact h

/-- **The scalar two-point (binary) Pinsker inequality**, in the weighted-`klFun` form the
partition step produces: for `q ∈ (0,1)` and `p ∈ [0,1]`,
`2 (p − q)² ≤ q·klFun(p/q) + (1−q)·klFun((1−p)/(1−q))`, the right side being the binary KL
`p log(p/q) + (1−p) log((1−p)/(1−q))`. -/
theorem two_mul_sq_le_weighted_klFun {p q : ℝ} (hq0 : 0 < q) (hq1 : q < 1)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    2 * (p - q) ^ 2 ≤ q * klFun (p / q) + (1 - q) * klFun ((1 - p) / (1 - q)) := by
  have h1q : (0:ℝ) < 1 - q := by linarith
  -- the weighted `klFun` sum IS the binary KL: the linear parts cancel
  have e1 : p * Real.log (p / q) = p * Real.log p - p * Real.log q := by
    rcases eq_or_lt_of_le hp0 with h | h
    · simp [← h]
    · rw [Real.log_div h.ne' hq0.ne']; ring
  have e2 : (1 - p) * Real.log ((1 - p) / (1 - q))
      = (1 - p) * Real.log (1 - p) - (1 - p) * Real.log (1 - q) := by
    rcases eq_or_lt_of_le (show (0:ℝ) ≤ 1 - p by linarith) with h | h
    · simp [← h]
    · rw [Real.log_div h.ne' h1q.ne']; ring
  have hq_ne : q ≠ 0 := hq0.ne'
  have h1q_ne : (1:ℝ) - q ≠ 0 := h1q.ne'
  have hkl1 : q * klFun (p / q) = p * Real.log (p / q) + q - p := by
    rw [klFun_apply]
    field_simp
  have hkl2 : (1 - q) * klFun ((1 - p) / (1 - q))
      = (1 - p) * Real.log ((1 - p) / (1 - q)) + (1 - q) - (1 - p) := by
    rw [klFun_apply]
    field_simp
  have hnn := klGap_nonneg hq0 hq1 hp0 hp1
  simp only [klGap] at hnn
  rw [hkl1, hkl2, e1, e2]
  linarith

/-! ### (A.2) The two-set partition step and Pinsker for measures -/

variable {Ω : Type*} [MeasurableSpace Ω] {P Q : Measure Ω}

/-- Jensen on one cell of the partition: restricting both measures to `A`,
`Q(A)·klFun(P(A)/Q(A)) ≤ ∫_A klFun(dP/dQ) dQ`. -/
private lemma set_jensen [IsFiniteMeasure P] [IsFiniteMeasure Q] (hPQ : P ≪ Q)
    {A : Set Ω} (hA : MeasurableSet A)
    (hklint : Integrable (fun x => klFun ((P.rnDeriv Q x).toReal)) Q) :
    Q.real A * klFun (P.real A / Q.real A)
      ≤ ∫ x in A, klFun ((P.rnDeriv Q x).toReal) ∂Q := by
  have hPA : P.restrict A = (Q.restrict A).withDensity (P.rnDeriv Q) := by
    conv_lhs => rw [← Measure.withDensity_rnDeriv_eq P Q hPQ]
    exact restrict_withDensity hA _
  have hac : P.restrict A ≪ Q.restrict A := by
    rw [hPA]; exact withDensity_absolutelyContinuous _ _
  have hrn : (P.restrict A).rnDeriv (Q.restrict A) =ᵐ[Q.restrict A] P.rnDeriv Q := by
    rw [hPA]; exact Measure.rnDeriv_withDensity _ (Measure.measurable_rnDeriv P Q)
  have hint' : Integrable
      (fun x => klFun (((P.restrict A).rnDeriv (Q.restrict A) x).toReal)) (Q.restrict A) := by
    refine (hklint.restrict (s := A)).congr ?_
    filter_upwards [hrn] with x hx
    rw [hx]
  have h := mul_le_integral_rnDeriv_of_ac convexOn_klFun
    continuous_klFun.continuousWithinAt hint' hac
  have hQu : (Q.restrict A).real Set.univ = Q.real A := by
    simp [measureReal_def]
  have hPu : (P.restrict A).real Set.univ = P.real A := by
    simp [measureReal_def]
  have hI : ∫ x, klFun (((P.restrict A).rnDeriv (Q.restrict A) x).toReal) ∂(Q.restrict A)
      = ∫ x in A, klFun ((P.rnDeriv Q x).toReal) ∂Q := by
    refine integral_congr_ae ?_
    filter_upwards [hrn] with x hx
    rw [hx]
  rw [hQu, hPu, hI] at h
  exact h

/-- **The two-set partition ("restricted data-processing") inequality**: the binary KL of the
partition `{A, Aᶜ}` is at most the full KL, in the weighted-`klFun` form. -/
theorem binary_partition_le_toReal_klDiv [IsFiniteMeasure P] [IsFiniteMeasure Q]
    (hPQ : P ≪ Q) (h_int : Integrable (llr P Q) P) {A : Set Ω} (hA : MeasurableSet A) :
    Q.real A * klFun (P.real A / Q.real A) + Q.real Aᶜ * klFun (P.real Aᶜ / Q.real Aᶜ)
      ≤ (klDiv P Q).toReal := by
  have hklint : Integrable (fun x => klFun ((P.rnDeriv Q x).toReal)) Q :=
    (integrable_klFun_rnDeriv_iff hPQ).mpr h_int
  have h1 := set_jensen hPQ hA hklint
  have h2 := set_jensen hPQ hA.compl hklint
  have hsplit : (∫ x in A, klFun ((P.rnDeriv Q x).toReal) ∂Q)
      + ∫ x in Aᶜ, klFun ((P.rnDeriv Q x).toReal) ∂Q
      = ∫ x, klFun ((P.rnDeriv Q x).toReal) ∂Q := integral_add_compl hA hklint
  rw [toReal_klDiv_eq_integral_klFun hPQ]
  linarith

/-- **Pinsker, squared form**: `2 (P A − Q A)² ≤ KL(P‖Q)` for probability measures, every
measurable `A`. -/
theorem two_mul_sq_le_toReal_klDiv [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hPQ : P ≪ Q) (h_int : Integrable (llr P Q) P) {A : Set Ω} (hA : MeasurableSet A) :
    2 * (P.real A - Q.real A) ^ 2 ≤ (klDiv P Q).toReal := by
  by_cases hq0 : Q A = 0
  · have hp' : P A = 0 := hPQ hq0
    have hP0 : P.real A = 0 := by simp [measureReal_def, hp']
    have hQ0 : Q.real A = 0 := by simp [measureReal_def, hq0]
    rw [hP0, hQ0]
    simpa using ENNReal.toReal_nonneg
  by_cases hqc : Q Aᶜ = 0
  · have hpc : P Aᶜ = 0 := hPQ hqc
    have hPc0 : P.real Aᶜ = 0 := by simp [measureReal_def, hpc]
    have hQc0 : Q.real Aᶜ = 0 := by simp [measureReal_def, hqc]
    have hP1 : P.real A = 1 := by
      have h := measureReal_add_measureReal_compl (μ := P) hA
      rw [probReal_univ] at h
      linarith
    have hQ1 : Q.real A = 1 := by
      have h := measureReal_add_measureReal_compl (μ := Q) hA
      rw [probReal_univ] at h
      linarith
    rw [hP1, hQ1]
    simpa using ENNReal.toReal_nonneg
  · -- main case: `0 < Q A` and `0 < Q Aᶜ`, so `Q.real A ∈ (0,1)`
    have hQA_ne : Q.real A ≠ 0 := by
      simp only [measureReal_def]
      exact ENNReal.toReal_ne_zero.mpr ⟨hq0, measure_ne_top Q A⟩
    have hQAc_ne : Q.real Aᶜ ≠ 0 := by
      simp only [measureReal_def]
      exact ENNReal.toReal_ne_zero.mpr ⟨hqc, measure_ne_top Q Aᶜ⟩
    have hq0' : 0 < Q.real A := lt_of_le_of_ne measureReal_nonneg (Ne.symm hQA_ne)
    have hQsum : Q.real A + Q.real Aᶜ = 1 := by
      have h := measureReal_add_measureReal_compl (μ := Q) hA
      rwa [probReal_univ] at h
    have hq1' : Q.real A < 1 := by
      have : 0 < Q.real Aᶜ := lt_of_le_of_ne measureReal_nonneg (Ne.symm hQAc_ne)
      linarith
    have hPsum : P.real A + P.real Aᶜ = 1 := by
      have h := measureReal_add_measureReal_compl (μ := P) hA
      rwa [probReal_univ] at h
    have hp0 : 0 ≤ P.real A := measureReal_nonneg
    have hp1 : P.real A ≤ 1 := by
      have : 0 ≤ P.real Aᶜ := measureReal_nonneg
      linarith
    have hPc : P.real Aᶜ = 1 - P.real A := by linarith
    have hQc : Q.real Aᶜ = 1 - Q.real A := by linarith
    calc 2 * (P.real A - Q.real A) ^ 2
        ≤ Q.real A * klFun (P.real A / Q.real A)
          + (1 - Q.real A) * klFun ((1 - P.real A) / (1 - Q.real A)) :=
          two_mul_sq_le_weighted_klFun hq0' hq1' hp0 hp1
      _ = Q.real A * klFun (P.real A / Q.real A)
          + Q.real Aᶜ * klFun (P.real Aᶜ / Q.real Aᶜ) := by rw [hPc, hQc]
      _ ≤ (klDiv P Q).toReal := binary_partition_le_toReal_klDiv hPQ h_int hA

/-- **Pinsker, hypothesis-free `ℝ≥0∞` form**: `ofReal (2 (P A − Q A)²) ≤ klDiv P Q` — when the KL
is infinite the bound is trivial, so no absolute-continuity or integrability hypothesis is
needed. -/
theorem ofReal_two_mul_sq_le_klDiv [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {A : Set Ω} (hA : MeasurableSet A) :
    ENNReal.ofReal (2 * (P.real A - Q.real A) ^ 2) ≤ klDiv P Q := by
  by_cases hac : P ≪ Q
  swap
  · rw [klDiv_of_not_ac hac]; exact le_top
  by_cases hint : Integrable (llr P Q) P
  swap
  · rw [klDiv_of_not_integrable hint]; exact le_top
  calc ENNReal.ofReal (2 * (P.real A - Q.real A) ^ 2)
      ≤ ENNReal.ofReal ((klDiv P Q).toReal) :=
        ENNReal.ofReal_le_ofReal (two_mul_sq_le_toReal_klDiv hac hint hA)
    _ = klDiv P Q := ENNReal.ofReal_toReal (klDiv_ne_top hac hint)

/-- **Pinsker's inequality, setwise form**: `|P A − Q A| ≤ √(KL(P‖Q)/2)` for every measurable
`A`, whenever the KL is finite. The statistical total variation is the supremum of the left side
over events, so this is `TV(P,Q) ≤ √(KL/2)`. -/
theorem pinsker_setwise [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hfin : klDiv P Q ≠ ⊤) {A : Set Ω} (hA : MeasurableSet A) :
    |P.real A - Q.real A| ≤ Real.sqrt ((klDiv P Q).toReal / 2) := by
  obtain ⟨hac, hint⟩ := klDiv_ne_top_iff.mp hfin
  have h := two_mul_sq_le_toReal_klDiv hac hint hA
  rw [← Real.sqrt_sq_eq_abs]
  exact Real.sqrt_le_sqrt (by linarith)

/-- **Pinsker fed to Le Cam**: every test `A` between `P` and `Q` has summed type-I + type-II
error at least `1 − √(KL(P‖Q)/2)`. -/
theorem pinsker_lecam [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hfin : klDiv P Q ≠ ⊤) {A : Set Ω} (hA : MeasurableSet A) :
    1 - Real.sqrt ((klDiv P Q).toReal / 2) ≤ P.real A + Q.real Aᶜ :=
  lecam_two_point (fun _B hB => pinsker_setwise hfin hB) hA

/-- If `KL(P‖Q) < 1/2` then every test's summed error exceeds `1/2`: the generic crossover
underlying `eq:nearnull-cross`. -/
theorem summed_error_gt_half_of_klDiv_lt_half [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hfin : klDiv P Q ≠ ⊤) (hlt : (klDiv P Q).toReal < 1 / 2)
    {A : Set Ω} (hA : MeasurableSet A) :
    1 / 2 < P.real A + Q.real Aᶜ := by
  have hp := pinsker_lecam hfin hA
  have hs : Real.sqrt ((klDiv P Q).toReal / 2) < 1 / 2 := by
    rw [Real.sqrt_lt' (by norm_num : (0:ℝ) < 1 / 2)]
    have := ENNReal.toReal_nonneg (a := klDiv P Q)
    nlinarith
  linarith

/-! ### (B.1) KL tensorization: invariance, products, finite `pi` products -/

/-- KL divergence is invariant under a measurable equivalence of the sample space (both the
absolute-continuity failure and the integrability failure transport, so no hypotheses). -/
theorem klDiv_map_measurableEquiv {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (P Q : Measure α) [IsFiniteMeasure P] [IsFiniteMeasure Q] (e : α ≃ᵐ β) :
    klDiv (P.map e) (Q.map e) = klDiv P Q := by
  by_cases hac : P ≪ Q
  swap
  · have hac' : ¬ P.map e ≪ Q.map e := by
      intro h
      have h2 := h.map e.symm.measurable
      rw [e.map_symm_map, e.map_symm_map] at h2
      exact hac h2
    rw [klDiv_of_not_ac hac, klDiv_of_not_ac hac']
  · have hmac : P.map e ≪ Q.map e := hac.map e.measurable
    have hr_meas : Measurable (P.rnDeriv Q) := Measure.measurable_rnDeriv P Q
    have hf_meas : Measurable fun y => P.rnDeriv Q (e.symm y) := hr_meas.comp e.symm.measurable
    -- transport the density through `e`
    have hcomm : (Q.withDensity (P.rnDeriv Q)).map e
        = (Q.map e).withDensity (fun y => P.rnDeriv Q (e.symm y)) := by
      refine Measure.ext fun s hs => ?_
      rw [Measure.map_apply e.measurable hs, withDensity_apply _ (e.measurable hs),
        withDensity_apply _ hs, setLIntegral_map hs hf_meas e.measurable]
      simp only [MeasurableEquiv.symm_apply_apply]
    have hPmap : P.map e = (Q.map e).withDensity (fun y => P.rnDeriv Q (e.symm y)) := by
      conv_lhs => rw [← Measure.withDensity_rnDeriv_eq P Q hac]
      exact hcomm
    have hrn : (P.map e).rnDeriv (Q.map e) =ᵐ[Q.map e] fun y => P.rnDeriv Q (e.symm y) := by
      rw [hPmap]
      exact Measure.rnDeriv_withDensity _ hf_meas
    have hrnP : (P.map e).rnDeriv (Q.map e) =ᵐ[P.map e] fun y => P.rnDeriv Q (e.symm y) :=
      hmac.ae_eq hrn
    have hllr : llr (P.map e) (Q.map e) =ᵐ[P.map e] fun y => llr P Q (e.symm y) := by
      filter_upwards [hrnP] with y hy
      simp only [llr_def]
      rw [hy]
    have hcompeq : ((fun y => llr P Q (e.symm y)) ∘ ⇑e) = llr P Q := by
      funext x
      simp
    have hint_iff : Integrable (llr (P.map e) (Q.map e)) (P.map e) ↔ Integrable (llr P Q) P := by
      rw [integrable_congr hllr, e.measurableEmbedding.integrable_map_iff, hcompeq]
    by_cases hint : Integrable (llr P Q) P
    swap
    · rw [klDiv_of_not_integrable hint,
        klDiv_of_not_integrable (fun h => hint (hint_iff.mp h))]
    · have hint' : Integrable (llr (P.map e) (Q.map e)) (P.map e) := hint_iff.mpr hint
      rw [klDiv_of_ac_of_integrable hac hint, klDiv_of_ac_of_integrable hmac hint']
      congr 1
      have hmeas' : AEStronglyMeasurable (fun y => llr P Q (e.symm y)) (P.map e) :=
        ((measurable_llr P Q).comp e.symm.measurable).aestronglyMeasurable
      have hIeq : ∫ y, llr (P.map e) (Q.map e) y ∂(P.map e) = ∫ x, llr P Q x ∂P := by
        rw [integral_congr_ae hllr, integral_map e.measurable.aemeasurable hmeas']
        simp
      have hPu : (P.map e).real Set.univ = P.real Set.univ := by
        simp [measureReal_def, Measure.map_apply e.measurable MeasurableSet.univ]
      have hQu : (Q.map e).real Set.univ = Q.real Set.univ := by
        simp [measureReal_def, Measure.map_apply e.measurable MeasurableSet.univ]
      rw [hIeq, hPu, hQu]

/-- **KL tensorizes over a product of two independent pairs**:
`KL(μ₁ ⊗ ν₁ ‖ μ₂ ⊗ ν₂) = KL(μ₁‖μ₂) + KL(ν₁‖ν₂)`, unconditionally, from the chain rule with
constant kernels plus the swap invariance. -/
theorem klDiv_prod {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ₁ μ₂ : Measure α) (ν₁ ν₂ : Measure β) [IsProbabilityMeasure μ₁] [IsProbabilityMeasure μ₂]
    [IsProbabilityMeasure ν₁] [IsProbabilityMeasure ν₂] :
    klDiv (μ₁.prod ν₁) (μ₂.prod ν₂) = klDiv μ₁ μ₂ + klDiv ν₁ ν₂ := by
  have hmid : klDiv (μ₁.prod ν₁) (μ₁.prod ν₂) = klDiv ν₁ ν₂ := by
    have hcoe : ⇑(MeasurableEquiv.prodComm (α := α) (β := β)) = Prod.swap := rfl
    calc klDiv (μ₁.prod ν₁) (μ₁.prod ν₂)
        = klDiv ((μ₁.prod ν₁).map (MeasurableEquiv.prodComm))
            ((μ₁.prod ν₂).map (MeasurableEquiv.prodComm)) :=
          (klDiv_map_measurableEquiv _ _ _).symm
      _ = klDiv (ν₁.prod μ₁) (ν₂.prod μ₁) := by
          rw [hcoe, Measure.prod_swap, Measure.prod_swap]
      _ = klDiv (ν₁ ⊗ₘ Kernel.const β μ₁) (ν₂ ⊗ₘ Kernel.const β μ₁) := by
          rw [Measure.compProd_const, Measure.compProd_const]
      _ = klDiv ν₁ ν₂ := klDiv_compProd_left ν₁ ν₂ (Kernel.const β μ₁)
  calc klDiv (μ₁.prod ν₁) (μ₂.prod ν₂)
      = klDiv (μ₁ ⊗ₘ Kernel.const α ν₁) (μ₂ ⊗ₘ Kernel.const α ν₂) := by
        rw [Measure.compProd_const, Measure.compProd_const]
    _ = klDiv μ₁ μ₂ + klDiv (μ₁ ⊗ₘ Kernel.const α ν₁) (μ₁ ⊗ₘ Kernel.const α ν₂) :=
        klDiv_compProd_eq_add μ₁ μ₂ (Kernel.const α ν₁) (Kernel.const α ν₂)
    _ = klDiv μ₁ μ₂ + klDiv ν₁ ν₂ := by
        rw [Measure.compProd_const, Measure.compProd_const, hmid]

universe u

/-- **KL tensorizes over a finite independent product** (absent from the pinned mathlib):
`KL(⨂ᵢ Pᵢ ‖ ⨂ᵢ Qᵢ) = ∑ᵢ KL(Pᵢ‖Qᵢ)` over `i : Fin n`, unconditionally. Induction splitting off
coordinate `0` through the measure-preserving `piFinSuccAbove`, as in `ChiSquaredPi`. -/
theorem klDiv_pi : ∀ (n : ℕ) (X : Fin n → Type u) [∀ i, MeasurableSpace (X i)]
    (P Q : ∀ i, Measure (X i)) [∀ i, IsProbabilityMeasure (P i)]
    [∀ i, IsProbabilityMeasure (Q i)],
    klDiv (Measure.pi P) (Measure.pi Q) = ∑ i, klDiv (P i) (Q i) := by
  intro n
  induction n with
  | zero =>
    intro X _ P Q _ _
    rw [Measure.pi_of_empty P, Measure.pi_of_empty Q]
    simp [klDiv_self]
  | succ n ih =>
    intro X _ P Q _ _
    set e := MeasurableEquiv.piFinSuccAbove X 0 with he
    have he_P := measurePreserving_piFinSuccAbove P 0
    have he_Q := measurePreserving_piFinSuccAbove Q 0
    calc klDiv (Measure.pi P) (Measure.pi Q)
        = klDiv ((Measure.pi P).map e) ((Measure.pi Q).map e) :=
          (klDiv_map_measurableEquiv _ _ e).symm
      _ = klDiv ((P 0).prod (Measure.pi fun j => P (Fin.succAbove 0 j)))
            ((Q 0).prod (Measure.pi fun j => Q (Fin.succAbove 0 j))) := by
          rw [he_P.map_eq, he_Q.map_eq]
      _ = klDiv (P 0) (Q 0)
            + klDiv (Measure.pi fun j => P (Fin.succAbove 0 j))
                (Measure.pi fun j => Q (Fin.succAbove 0 j)) := klDiv_prod _ _ _ _
      _ = klDiv (P 0) (Q 0)
            + ∑ j : Fin n, klDiv (P (Fin.succAbove 0 j)) (Q (Fin.succAbove 0 j)) := by
          rw [ih _ _ _]
      _ = ∑ i, klDiv (P i) (Q i) :=
          (Fin.sum_univ_succAbove (fun i => klDiv (P i) (Q i)) 0).symm

/-! ### (B.2) The 1-D Gaussian KL closed form -/

/-- The Radon–Nikodym density of two same-variance Gaussians is a.e. the pdf ratio (chain rule
through `volume`; same argument as `GaussianChiSquared.lean`). -/
private lemma rnDeriv_gaussian_toReal_ae (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    (fun x => ((gaussianReal m₁ v).rnDeriv (gaussianReal m₂ v) x).toReal)
      =ᵐ[gaussianReal m₂ v]
    (fun x => gaussianPDFReal m₁ v x / gaussianPDFReal m₂ v x) := by
  have hP1v : gaussianReal m₁ v ≪ volume := gaussianReal_absolutelyContinuous m₁ hv
  have hP2v : gaussianReal m₂ v ≪ volume := gaussianReal_absolutelyContinuous m₂ hv
  have hvP2 : volume ≪ gaussianReal m₂ v := gaussianReal_absolutelyContinuous' m₂ hv
  have hP1P2 : gaussianReal m₁ v ≪ gaussianReal m₂ v := hP1v.trans hvP2
  refine hP2v.ae_eq ?_
  have hchain := Measure.rnDeriv_mul_rnDeriv (μ := gaussianReal m₁ v)
    (ν := gaussianReal m₂ v) (κ := volume) hP1P2
  have h1v := rnDeriv_gaussianReal m₁ v
  have h2v := rnDeriv_gaussianReal m₂ v
  filter_upwards [hchain, h1v, h2v] with x hx h1 h2
  rw [Pi.mul_apply, h1, h2] at hx
  have hpos2 : (0 : ℝ) < gaussianPDFReal m₂ v x := gaussianPDFReal_pos m₂ v x hv
  have hxR := congrArg ENNReal.toReal hx
  simp only [ENNReal.toReal_mul, toReal_gaussianPDF] at hxR
  rw [eq_div_iff hpos2.ne']
  exact hxR

/-- The log of the same-variance Gaussian pdf ratio is the quadratic difference. -/
private lemma log_gaussianPDFReal_div (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (x : ℝ) :
    Real.log (gaussianPDFReal m₁ v x / gaussianPDFReal m₂ v x)
      = ((x - m₂) ^ 2 - (x - m₁) ^ 2) / (2 * v) := by
  have hvpos : (0:ℝ) < v := by exact_mod_cast pos_of_ne_zero hv
  have hs : (0:ℝ) < Real.sqrt (2 * π * v) := Real.sqrt_pos.mpr (by positivity)
  have hsinv : ((Real.sqrt (2 * π * v))⁻¹ : ℝ) ≠ 0 := (inv_pos.mpr hs).ne'
  simp only [gaussianPDFReal]
  rw [mul_div_mul_left _ _ hsinv, ← Real.exp_sub, Real.log_exp]
  ring

/-- The affine log-likelihood ratio is integrable and integrates to `(m₁−m₂)²/(2v)` under
`N(m₁,v)`. -/
private lemma quad_ratio_rw (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    (fun x : ℝ => ((x - m₂) ^ 2 - (x - m₁) ^ 2) / (2 * (v:ℝ)))
      = fun x : ℝ => ((m₁ - m₂) / (v:ℝ)) * x + (m₂ ^ 2 - m₁ ^ 2) / (2 * (v:ℝ)) := by
  have hvpos : (0:ℝ) < v := by exact_mod_cast pos_of_ne_zero hv
  funext x
  field_simp
  ring

private lemma integrable_id_gaussian (m : ℝ) (v : ℝ≥0) :
    Integrable (fun x : ℝ => x) (gaussianReal m v) := by
  have h := memLp_id_gaussianReal (μ := m) (v := v) 1
  rw [ENNReal.coe_one] at h
  exact memLp_one_iff_integrable.mp h

private lemma integrable_quad_gaussian (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    Integrable (fun x : ℝ => ((x - m₂) ^ 2 - (x - m₁) ^ 2) / (2 * (v:ℝ)))
      (gaussianReal m₁ v) := by
  rw [quad_ratio_rw m₁ m₂ hv]
  exact ((integrable_id_gaussian m₁ v).const_mul _).add (integrable_const _)

private lemma integral_quad_gaussian (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    ∫ x, ((x - m₂) ^ 2 - (x - m₁) ^ 2) / (2 * (v:ℝ)) ∂(gaussianReal m₁ v)
      = (m₁ - m₂) ^ 2 / (2 * v) := by
  have hvpos : (0:ℝ) < v := by exact_mod_cast pos_of_ne_zero hv
  rw [quad_ratio_rw m₁ m₂ hv,
    integral_add ((integrable_id_gaussian m₁ v).const_mul _) (integrable_const _),
    integral_const_mul]
  have hid : ∫ x, x ∂(gaussianReal m₁ v) = m₁ := integral_id_gaussianReal
  rw [hid, integral_const]
  simp only [probReal_univ, one_smul]
  field_simp
  ring

/-- **The 1-D Gaussian KL closed form** (`p:nearnull` (ii), mean case, per survey):
`KL(N(m₁,v) ‖ N(m₂,v)) = (m₁ − m₂)²/(2v)`. -/
theorem klDiv_gaussianReal (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    klDiv (gaussianReal m₁ v) (gaussianReal m₂ v)
      = ENNReal.ofReal ((m₁ - m₂) ^ 2 / (2 * v)) := by
  have hac : gaussianReal m₁ v ≪ gaussianReal m₂ v :=
    (gaussianReal_absolutelyContinuous m₁ hv).trans
      (gaussianReal_absolutelyContinuous' m₂ hv)
  have hrnP : (fun x => ((gaussianReal m₁ v).rnDeriv (gaussianReal m₂ v) x).toReal)
      =ᵐ[gaussianReal m₁ v]
      (fun x => gaussianPDFReal m₁ v x / gaussianPDFReal m₂ v x) :=
    hac.ae_eq (rnDeriv_gaussian_toReal_ae m₁ m₂ hv)
  have hllr : llr (gaussianReal m₁ v) (gaussianReal m₂ v) =ᵐ[gaussianReal m₁ v]
      fun x => ((x - m₂) ^ 2 - (x - m₁) ^ 2) / (2 * v) := by
    filter_upwards [hrnP] with x hx
    simp only [llr_def]
    rw [hx, log_gaussianPDFReal_div m₁ m₂ hv x]
  have hLint := integrable_quad_gaussian m₁ m₂ hv
  have hint : Integrable (llr (gaussianReal m₁ v) (gaussianReal m₂ v)) (gaussianReal m₁ v) :=
    hLint.congr hllr.symm
  rw [klDiv_of_ac_of_integrable hac hint]
  have hI : ∫ x, llr (gaussianReal m₁ v) (gaussianReal m₂ v) x ∂(gaussianReal m₁ v)
      = (m₁ - m₂) ^ 2 / (2 * v) := by
    rw [integral_congr_ae hllr]
    exact integral_quad_gaussian m₁ m₂ hv
  rw [hI, probReal_univ, probReal_univ]
  norm_num

/-- **The `n`-survey iid Gaussian mean-shift KL**: `n` independent surveys sum the per-survey
divergences, `KL = n (m₁−m₂)²/(2v)`. -/
theorem klDiv_pi_gaussianReal (n : ℕ) (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0) :
    klDiv (Measure.pi fun _ : Fin n => gaussianReal m₁ v)
        (Measure.pi fun _ : Fin n => gaussianReal m₂ v)
      = ENNReal.ofReal ((n : ℝ) * ((m₁ - m₂) ^ 2 / (2 * v))) := by
  rw [klDiv_pi n (fun _ => ℝ) (fun _ => gaussianReal m₁ v) (fun _ => gaussianReal m₂ v)]
  simp only [klDiv_gaussianReal m₁ m₂ hv]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    ← ENNReal.ofReal_natCast n, ← ENNReal.ofReal_mul (Nat.cast_nonneg n)]

/-- **The whitened multivariate mean-shift KL** (`p:nearnull` (ii) after whitening): in whitened
(unit-variance, independent) coordinates the mean shift `a − b` has
`KL = (∑ᵢ (aᵢ − bᵢ)²)/2 = ‖a − b‖²/2`. -/
theorem klDiv_pi_gaussian_mean_shift {d : ℕ} (a b : Fin d → ℝ) :
    klDiv (Measure.pi fun i => gaussianReal (a i) 1)
        (Measure.pi fun i => gaussianReal (b i) 1)
      = ENNReal.ofReal ((∑ i, (a i - b i) ^ 2) / 2) := by
  rw [klDiv_pi d (fun _ => ℝ) (fun i => gaussianReal (a i) 1) (fun i => gaussianReal (b i) 1)]
  have hterm : ∀ i, klDiv (gaussianReal (a i) 1) (gaussianReal (b i) 1)
      = ENNReal.ofReal ((a i - b i) ^ 2 / 2) := by
    intro i
    rw [klDiv_gaussianReal _ _ one_ne_zero]
    norm_num
  simp_rw [hterm]
  rw [← ENNReal.ofReal_sum_of_nonneg (fun i _ => by positivity)]
  congr 1
  rw [Finset.sum_div]

/-! ### (B.3) The paper's crossovers (`eq:nearnull-kl`, `eq:nearnull-cross`) -/

/-- The scalar content of `eq:nearnull-kl`'s quadratic bound: `½(t − log(1+t)) ≤ t²/4` for
`t ≥ 0`, i.e. `log(1+t) ≥ t − t²/2`. -/
theorem half_sub_log_le_quarter_sq {t : ℝ} (ht : 0 ≤ t) :
    (t - Real.log (1 + t)) / 2 ≤ t ^ 2 / 4 := by
  suffices h : t - t ^ 2 / 2 ≤ Real.log (1 + t) by linarith
  have hφd : ∀ x ∈ Set.Ioi (0:ℝ),
      HasDerivAt (fun s : ℝ => Real.log (1 + s) - s + s ^ 2 / 2) ((1 + x)⁻¹ - 1 + x) x := by
    intro x hx
    have hx0 : (0:ℝ) < x := hx
    have h1x : (0:ℝ) < 1 + x := by linarith
    have h1 : HasDerivAt (fun s : ℝ => 1 + s) 1 x := by
      simpa using (hasDerivAt_id x).const_add (1 : ℝ)
    have h2 : HasDerivAt (fun s : ℝ => Real.log (1 + s)) (1 / (1 + x)) x := h1.log h1x.ne'
    have h3 : HasDerivAt (fun s : ℝ => s ^ 2 / 2) x x := by
      have := ((hasDerivAt_id x).pow 2).div_const 2
      simpa using this
    exact ((h2.sub (hasDerivAt_id x)).add h3).congr_deriv (by rw [one_div])
  have hφc : ContinuousOn (fun s : ℝ => Real.log (1 + s) - s + s ^ 2 / 2) (Set.Ici 0) := by
    apply ContinuousOn.add
    · apply ContinuousOn.sub
      · apply ContinuousOn.log
        · exact (continuous_const.add continuous_id).continuousOn
        · intro x hx
          have hx' : (0:ℝ) ≤ x := hx
          intro hcontra
          linarith [hcontra]
      · exact continuous_id.continuousOn
    · exact (continuous_id.pow 2).div_const 2 |>.continuousOn
  have hmono : MonotoneOn (fun s : ℝ => Real.log (1 + s) - s + s ^ 2 / 2) (Set.Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0) hφc
    · intro x hx
      rw [interior_Ici] at hx
      exact (hφd x hx).differentiableAt.differentiableWithinAt
    · intro x hx
      rw [interior_Ici] at hx
      rw [(hφd x hx).deriv]
      have hx0 : (0:ℝ) < x := hx
      have h1x : (0:ℝ) < 1 + x := by linarith
      have hkey : (1 + x)⁻¹ - 1 + x = x ^ 2 / (1 + x) := by
        field_simp
        ring
      rw [hkey]
      positivity
  have h := hmono Set.self_mem_Ici ht ht
  norm_num at h
  linarith

/-- **The paper's spread crossover, `eq:nearnull-cross`**: with the per-survey KL bounded by
`t²/4` (the quadratic bound of `eq:nearnull-kl`), `n < n⋆ = 2/t²` surveys leave every test's
summed error above `1/2`. -/
theorem nearnull_spread_crossover [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hfin : klDiv P Q ≠ ⊤) {t : ℝ} (ht : 0 < t) {n : ℕ}
    (hKL : (klDiv P Q).toReal ≤ (n : ℝ) * (t ^ 2 / 4)) (hn : (n : ℝ) < 2 / t ^ 2)
    {A : Set Ω} (hA : MeasurableSet A) :
    1 / 2 < P.real A + Q.real Aᶜ := by
  apply summed_error_gt_half_of_klDiv_lt_half hfin _ hA
  have ht2 : (0:ℝ) < t ^ 2 := by positivity
  have h1 : (n : ℝ) * (t ^ 2 / 4) < (2 / t ^ 2) * (t ^ 2 / 4) :=
    mul_lt_mul_of_pos_right hn (by positivity)
  have h2 : (2 / t ^ 2) * (t ^ 2 / 4) = 1 / 2 := by
    have ht2' : t ^ 2 ≠ 0 := ht2.ne'
    field_simp
    ring
  linarith

/-- **The mean-misreport crossover**: with the paper's exact per-survey `KL = s²/2` (mean case of
`eq:nearnull-kl`), `n < 1/s²` surveys leave every test's summed error above `1/2`. (Note the
constant: via Pinsker, `KL = s²/2` per survey gives `n⋆ = 1/s²`; the paper's `2/t²` belongs to
the spread bound `KL ≤ t²/4`.) -/
theorem nearnull_mean_crossover [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hfin : klDiv P Q ≠ ⊤) {s : ℝ} (hs : 0 < s) {n : ℕ}
    (hKL : (klDiv P Q).toReal ≤ (n : ℝ) * (s ^ 2 / 2)) (hn : (n : ℝ) < 1 / s ^ 2)
    {A : Set Ω} (hA : MeasurableSet A) :
    1 / 2 < P.real A + Q.real Aᶜ := by
  apply summed_error_gt_half_of_klDiv_lt_half hfin _ hA
  have hs2 : (0:ℝ) < s ^ 2 := by positivity
  have h1 : (n : ℝ) * (s ^ 2 / 2) < (1 / s ^ 2) * (s ^ 2 / 2) :=
    mul_lt_mul_of_pos_right hn (by positivity)
  have h2 : (1 / s ^ 2) * (s ^ 2 / 2) = 1 / 2 := by
    field_simp
  linarith

/-- **The `n`-survey Gaussian mean-misreport crossover, instantiated**: for `n` iid surveys of
`N(m₁,v)` vs `N(m₂,v)`, if `n (m₁−m₂)²/v < 1` then every test's summed error exceeds `1/2` —
`p:nearnull` (ii)'s undetectability below the crossover, by the Pinsker route (the paper's χ²
route is `NearNullOperator`/`GaussianDataLaw`). -/
theorem gaussianReal_nsurvey_mean_crossover (n : ℕ) (m₁ m₂ : ℝ) {v : ℝ≥0} (hv : v ≠ 0)
    (hn : (n : ℝ) * ((m₁ - m₂) ^ 2 / v) < 1)
    {A : Set (Fin n → ℝ)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n => gaussianReal m₁ v).real A
        + (Measure.pi fun _ : Fin n => gaussianReal m₂ v).real Aᶜ := by
  have hvpos : (0:ℝ) < v := by exact_mod_cast pos_of_ne_zero hv
  have hkl := klDiv_pi_gaussianReal n m₁ m₂ hv
  apply summed_error_gt_half_of_klDiv_lt_half
  · rw [hkl]; exact ENNReal.ofReal_ne_top
  · rw [hkl, ENNReal.toReal_ofReal (by positivity)]
    have hsplit : (n : ℝ) * ((m₁ - m₂) ^ 2 / (2 * v)) = ((n : ℝ) * ((m₁ - m₂) ^ 2 / v)) / 2 := by
      field_simp
    rw [hsplit]
    linarith
  · exact hA

/-! ### (C) `r:nonlinear`: the reweight's derivative vanishes along a blind direction -/

/-- **`r:nonlinear` (`eq:nonlinear-deriv`)**: with additive noise density `ν ∈ C¹`, a positive
archive weight `ρ`, and a dominating envelope `bound` allowing differentiation under the
expectation, the curation reweight `t ↦ ∫ ν(y − F(x₀ + t v))/ρ(y) dτ(y)` has **vanishing
derivative at `t = 0`** along every direction `v` in the Jacobian null space, `J(x₀) v = 0`:
a derivative statement at the base point, exactly as the remark claims. The hypotheses are the
remark's own: differentiability of `F` along the line (its Jacobian at `x₀` being `J`),
differentiability of `ν`, measurability of the integrand near `t = 0`, integrability at `t = 0`,
and the integrable envelope dominating the pointwise `t`-derivative on the unit ball. -/
theorem reweight_deriv_zero_along_blind
    {X Y : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
    [NormedAddCommGroup Y] [NormedSpace ℝ Y] [MeasurableSpace Y]
    (τ : Measure Y) (Fm : X → Y) (J : X →L[ℝ] Y) (x₀ v : X)
    (hJ : HasFDerivAt Fm J x₀) (hJv : J v = 0)
    (hFdiff : ∀ t : ℝ, DifferentiableAt ℝ Fm (x₀ + t • v))
    (ν : Y → ℝ) (hν : Differentiable ℝ ν) (ρ : Y → ℝ)
    (hmeas : ∀ t : ℝ, AEStronglyMeasurable (fun y => ν (y - Fm (x₀ + t • v)) / ρ y) τ)
    (hint : Integrable (fun y => ν (y - Fm x₀) / ρ y) τ)
    (bound : Y → ℝ) (hbound_int : Integrable bound τ)
    (hbound : ∀ᵐ y ∂τ, ∀ t ∈ Metric.ball (0:ℝ) 1,
      |fderiv ℝ ν (y - Fm (x₀ + t • v)) (fderiv ℝ Fm (x₀ + t • v) v) / ρ y| ≤ bound y) :
    HasDerivAt (fun t : ℝ => ∫ y, ν (y - Fm (x₀ + t • v)) / ρ y ∂τ) 0 0 := by
  -- the line `s ↦ x₀ + s • v` and the composite `s ↦ F(x₀ + s • v)`
  have hline : ∀ t : ℝ, HasDerivAt (fun s : ℝ => x₀ + s • v) v t := by
    intro t
    simpa using ((hasDerivAt_id t).smul_const v).const_add x₀
  have hc : ∀ t : ℝ, HasDerivAt (fun s : ℝ => Fm (x₀ + s • v))
      (fderiv ℝ Fm (x₀ + t • v) v) t := by
    intro t
    exact (hFdiff t).hasFDerivAt.comp_hasDerivAt t (hline t)
  -- the blind direction kills the derivative of the composite at `t = 0`
  have hc0 : fderiv ℝ Fm x₀ v = 0 := by
    rw [hJ.fderiv]
    exact hJv
  -- the pointwise derivative family
  set D : ℝ → Y → ℝ := fun t y =>
    -(fderiv ℝ ν (y - Fm (x₀ + t • v)) (fderiv ℝ Fm (x₀ + t • v) v)) / ρ y with hD
  have hDdiff : ∀ y, ∀ t : ℝ,
      HasDerivAt (fun s : ℝ => ν (y - Fm (x₀ + s • v)) / ρ y) (D t y) t := by
    intro y t
    have hinner : HasDerivAt (fun s : ℝ => y - Fm (x₀ + s • v))
        (-(fderiv ℝ Fm (x₀ + t • v) v)) t := (hc t).const_sub y
    have houter := (hν (y - Fm (x₀ + t • v))).hasFDerivAt.comp_hasDerivAt t hinner
    have := houter.div_const (ρ y)
    simpa [hD, map_neg] using this
  have hD0 : D 0 = fun _ => (0:ℝ) := by
    funext y
    simp [hD, hc0]
  -- differentiation under the integral
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (F := fun t y => ν (y - Fm (x₀ + t • v)) / ρ y) (F' := D) (bound := bound)
    (μ := τ) (x₀ := (0:ℝ)) (Metric.ball_mem_nhds (0:ℝ) one_pos)
    (Filter.Eventually.of_forall hmeas)
    (by simpa using hint)
    (by rw [hD0]; exact aestronglyMeasurable_const)
    (by
      filter_upwards [hbound] with y hy t ht
      have h := hy t ht
      simpa [hD, neg_div, Real.norm_eq_abs, abs_neg, abs_div] using h)
    hbound_int
    (Filter.Eventually.of_forall fun y t _ => hDdiff y t)
  have hzero : ∫ y, D 0 y ∂τ = 0 := by
    rw [hD0]
    simp
  have := key.2
  rw [hzero] at this
  exact this

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.two_mul_sq_le_weighted_klFun
#print axioms NearNull.binary_partition_le_toReal_klDiv
#print axioms NearNull.two_mul_sq_le_toReal_klDiv
#print axioms NearNull.ofReal_two_mul_sq_le_klDiv
#print axioms NearNull.pinsker_setwise
#print axioms NearNull.pinsker_lecam
#print axioms NearNull.summed_error_gt_half_of_klDiv_lt_half
#print axioms NearNull.klDiv_map_measurableEquiv
#print axioms NearNull.klDiv_prod
#print axioms NearNull.klDiv_pi
#print axioms NearNull.klDiv_gaussianReal
#print axioms NearNull.klDiv_pi_gaussianReal
#print axioms NearNull.klDiv_pi_gaussian_mean_shift
#print axioms NearNull.half_sub_log_le_quarter_sq
#print axioms NearNull.nearnull_spread_crossover
#print axioms NearNull.nearnull_mean_crossover
#print axioms NearNull.gaussianReal_nsurvey_mean_crossover
#print axioms NearNull.reweight_deriv_zero_along_blind
