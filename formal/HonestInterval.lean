import Mathlib

/-!
# Corollary `c:honest` of "Priors learned from legacy reconstructions inherit undetectable
overconfidence": no finite blind interval is an inference

This file machine-checks **Corollary `c:honest`** ("No finite blind interval is an inference;
any procedure; linear–Gaussian witness").

Paper statement: fix a blind direction `v`; let `𝒫` be the truths consistent with the measurement
law, carrying arbitrary blind-fiber conditionals; the witness with `W = w·vvᵀ` places in `𝒫`
truths whose conditional spread along `v` is arbitrarily large.  Let `Î` be ANY interval rule for
`θ = vᵀx` computed from the recorded measurements and the archive.  Its law is the same for every
member of `𝒫`, while the truth's conditional density along `v` is at most `(s√(2π))⁻¹` at
conditional spread `s`.  Hence
`inf_{p∈𝒫} Pr_p(θ⋆ ∈ Î) ≤ E|Î| / (√(2π) · sup_{p∈𝒫} s_p) = 0` whenever `E|Î| < ∞`, so an
interval with coverage `1−α` uniformly over `𝒫` has infinite expected length.

## Formal setting

The carrier is an abstract measurable space `𝒲` — **everything the procedure sees**: the recorded
measurements, the archive, and (for a randomized rule) any external randomization, folded into
`𝒲` as extra coordinates whose law does not depend on the truth.  The interval rule is a pair of
measurable endpoint maps `lo hi : 𝒲 → ℝ`, reporting `[lo w, hi w]`; its coverage under a truth is
evaluated against the **joint** law `ρ` of `(W, θ)` on `𝒲 × ℝ`.  (Closed intervals are the
strongest case: an open or half-open report has no larger coverage and the same length, so the
bound for `Icc` covers them all.  `lo ≤ hi` is not needed: an inverted report has empty coverage
and zero length.)  The conditional law of `θ` given `W` is mathlib's canonical disintegration
`ρ.condKernel` — its existence and validity (`ρ = ρ.fst ⊗ₘ ρ.condKernel`) is a **theorem**
(`Measure.disintegrate`), not a hypothesis.

## Results

* `coverage_compProd_le` — **the core lemma**: if the conditional law of `θ` given `W` is
  dominated by `B · Lebesgue`, then `Pr(θ ∈ [lo W, hi W]) ≤ B · E[len]`.  (Kernel form; proved by
  `Measure.compProd_apply` + monotonicity, i.e. the density bound integrated over the law of `W` —
  exactly the paper's proof sketch.)
* `coverage_le_of_condKernel_bound` — the core lemma at a joint law, with the conditional supplied
  by mathlib's canonical disintegration.
* `gaussianPDFReal_le` / `gaussianReal_apply_le_mul_volume` / `gaussianReal_spread_apply_le` —
  the Gaussian density maximum: `N(m, s²)` has density `≤ (s√(2π))⁻¹`, hence
  `N(m, s²)(t) ≤ (s√(2π))⁻¹ · |t|` for every set `t`.
* `coverage_le_of_gaussian_fiber` — the paper's displayed per-member bound
  `Pr_p(θ⋆ ∈ Î) ≤ E|Î| / (s_p √(2π))`.
* `iInf_eq_zero_of_le_mul` — the family arithmetic: `c i ≤ B i · L`, `⨅ B = 0`, `L ≠ ∞` force
  `⨅ c = 0`.
* `iInf_gaussian_bound_eq_zero` — unbounded spread makes the Gaussian density bounds infimum to
  zero.
* `iInf_coverage_eq_zero` — **the display**: over a family of truths with the SAME information
  law and Gaussian blind conditionals of unbounded spread, `inf_p Pr_p(θ⋆ ∈ Î) = 0` whenever the
  expected length is finite.
* `uniform_coverage_infinite_expected_length` — **`c:honest`**: an interval rule with coverage
  `≥ 1 − α` (`α < 1`) uniformly over the family has infinite expected length.

## Honesty — named hypotheses (flagged)

No `sorry`/`admit`; `#print axioms` on every public theorem lists only `propext`,
`Classical.choice`, `Quot.sound`.  Three analytic/modeling steps enter as **named hypotheses** of
the family theorems rather than being derived from the physical linear–Gaussian model:

* `hfst : (ρ i).fst = μ` — **the constant information law**: the law of everything the procedure
  sees is the same for every member of the family.  This is the content of Corollary `c:nonident`
  (machine-checked in `NonIdentifiability.lean`: `data_marginal_eq_of_resolved_eq`,
  `nonident_curatedPrior` — record law and curated archive depend on the truth only through its
  resolved marginal, and the witness `A N = 0` leaves them fixed).  The identification of THIS
  abstract `𝒲`-marginal with THAT physical record+archive law is a modeling step not formalized
  here.
* `hcond : ∀ᵐ w ∂μ, (ρ i).condKernel w = gaussianReal (m i w) (s i)²` — **the witness's blind
  fiber**: under the linear–Gaussian witness `W = w·vvᵀ` the conditional of `θ = vᵀx` given the
  record is Gaussian with spread free of the data.  The multivariate-Gaussian conditioning
  computation producing this from the witness joint law is not formalized here; only the
  Gaussian-ness of the fiber with the stated spread is consumed, and in fact only its density
  bound (`coverage_le_of_condKernel_bound` needs just `ν w ≤ B · Lebesgue`).
* `hsup : ∀ M, ∃ i, M < s i` — **unbounded conditional spread**: the witness places truths with
  arbitrarily large blind spread (the paper's "arbitrarily large", from `W = w·vvᵀ` with `w`
  free — `NonIdentifiability.witness_cov_annihilated` checks that every such `w` leaves the data
  law fixed).

Everything else — the density-bound coverage inequality, the Gaussian density maximum, the
disintegration, the family arithmetic, the endpoint measurability bookkeeping — is proved.
-/

open MeasureTheory ProbabilityTheory Real
open scoped ENNReal NNReal

namespace HonestInterval

variable {𝒲 : Type*} [MeasurableSpace 𝒲]

/-! ## The interval rule, its coverage, and its expected length -/

/-- The **coverage event** of the interval rule `w ↦ [lo w, hi w]`: the blind coordinate `θ`
(second component) lands in the interval reported from the information `w` (first component). -/
def coverageEvent (lo hi : 𝒲 → ℝ) : Set (𝒲 × ℝ) :=
  {p | p.2 ∈ Set.Icc (lo p.1) (hi p.1)}

/-- The **coverage** of the interval rule under the joint law `ρ` of `(W, θ)`:
`Pr_ρ(θ ∈ [lo W, hi W])`. -/
noncomputable def coverage (ρ : Measure (𝒲 × ℝ)) (lo hi : 𝒲 → ℝ) : ℝ≥0∞ :=
  ρ (coverageEvent lo hi)

/-- The **expected length** `E|Î|` of the interval rule under the information law `μ`
(in `ℝ≥0∞`; an inverted report contributes `0`). -/
noncomputable def expLength (μ : Measure 𝒲) (lo hi : 𝒲 → ℝ) : ℝ≥0∞ :=
  ∫⁻ w, ENNReal.ofReal (hi w - lo w) ∂μ

omit [MeasurableSpace 𝒲] in
lemma preimage_coverageEvent (lo hi : 𝒲 → ℝ) (w : 𝒲) :
    Prod.mk w ⁻¹' coverageEvent lo hi = Set.Icc (lo w) (hi w) := rfl

lemma measurableSet_coverageEvent {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi) :
    MeasurableSet (coverageEvent lo hi) := by
  have h : coverageEvent lo hi
      = {p : 𝒲 × ℝ | lo p.1 ≤ p.2} ∩ {p : 𝒲 × ℝ | p.2 ≤ hi p.1} := by
    ext p; simp [coverageEvent, Set.mem_Icc]
  rw [h]
  exact (measurableSet_le (hlo.comp measurable_fst) measurable_snd).inter
    (measurableSet_le measurable_snd (hhi.comp measurable_fst))

/-! ## The core lemma: the density bound integrated over the law of the information -/

/-- **The density-bound coverage bound** (kernel form).  If the conditional law `ν w` of the
blind coordinate given the information `w` is dominated by `B ·` Lebesgue (a.e. in `w`), then the
coverage of ANY measurable interval rule is at most `B ·` its expected length:
`Pr(θ ∈ [lo W, hi W]) ≤ B · E[hi W − lo W]`.  This is the paper's "the coverage bound is the
density bound integrated over that law", via `Measure.compProd_apply`. -/
theorem coverage_compProd_le (μ : Measure 𝒲) [SFinite μ] (ν : Kernel 𝒲 ℝ) [IsSFiniteKernel ν]
    {B : ℝ≥0∞} (hB : ∀ᵐ w ∂μ, ∀ t : Set ℝ, ν w t ≤ B * volume t)
    {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi) :
    coverage (μ ⊗ₘ ν) lo hi ≤ B * expLength μ lo hi := by
  unfold coverage
  rw [Measure.compProd_apply (measurableSet_coverageEvent hlo hhi)]
  simp_rw [preimage_coverageEvent]
  calc ∫⁻ w, ν w (Set.Icc (lo w) (hi w)) ∂μ
      ≤ ∫⁻ w, B * ENNReal.ofReal (hi w - lo w) ∂μ := by
        refine lintegral_mono_ae ?_
        filter_upwards [hB] with w hw
        calc ν w (Set.Icc (lo w) (hi w)) ≤ B * volume (Set.Icc (lo w) (hi w)) := hw _
          _ = B * ENNReal.ofReal (hi w - lo w) := by rw [Real.volume_Icc]
    _ = B * expLength μ lo hi := by
        rw [expLength, lintegral_const_mul _ ((hhi.sub hlo).ennreal_ofReal)]

/-- **The core lemma at a joint law.**  For any finite joint law `ρ` of `(W, θ)` whose canonical
conditional (mathlib's `ρ.condKernel`, whose existence and validity is a theorem) is dominated by
`B ·` Lebesgue, the coverage is at most `B ·` the expected length under the information marginal
`ρ.fst`. -/
theorem coverage_le_of_condKernel_bound (ρ : Measure (𝒲 × ℝ)) [IsFiniteMeasure ρ]
    {B : ℝ≥0∞} (hB : ∀ᵐ w ∂ρ.fst, ∀ t : Set ℝ, ρ.condKernel w t ≤ B * volume t)
    {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi) :
    coverage ρ lo hi ≤ B * expLength ρ.fst lo hi := by
  have h : coverage ρ lo hi = coverage (ρ.fst ⊗ₘ ρ.condKernel) lo hi := by
    rw [ρ.disintegrate ρ.condKernel]
  rw [h]
  exact coverage_compProd_le ρ.fst ρ.condKernel hB hlo hhi

/-! ## The Gaussian density maximum -/

/-- **The Gaussian density maximum**: the `N(m, v)` density is everywhere at most
`(√(2πv))⁻¹`. -/
theorem gaussianPDFReal_le (m : ℝ) (v : ℝ≥0) (x : ℝ) :
    gaussianPDFReal m v x ≤ (Real.sqrt (2 * π * v))⁻¹ := by
  simp only [gaussianPDFReal]
  have hexp : rexp (-(x - m) ^ 2 / (2 * v)) ≤ 1 :=
    Real.exp_le_one_iff.mpr
      (div_nonpos_iff.mpr (Or.inr ⟨neg_nonpos.mpr (sq_nonneg _), by positivity⟩))
  exact mul_le_of_le_one_right (by positivity) hexp

/-- The Gaussian **measure** is dominated by `(√(2πv))⁻¹ ·` Lebesgue on every set. -/
theorem gaussianReal_apply_le_mul_volume (m : ℝ) {v : ℝ≥0} (hv : v ≠ 0) (t : Set ℝ) :
    gaussianReal m v t ≤ ENNReal.ofReal (Real.sqrt (2 * π * v))⁻¹ * volume t := by
  rw [gaussianReal_apply m hv t]
  calc ∫⁻ x in t, gaussianPDF m v x
      ≤ ∫⁻ _ in t, ENNReal.ofReal (Real.sqrt (2 * π * v))⁻¹ :=
        lintegral_mono fun x => by
          simp only [gaussianPDF]
          exact ENNReal.ofReal_le_ofReal (gaussianPDFReal_le m v x)
    _ = ENNReal.ofReal (Real.sqrt (2 * π * v))⁻¹ * volume t := setLIntegral_const t _

/-- The spread form: `N(m, s²)` is dominated by `(s√(2π))⁻¹ ·` Lebesgue — the paper's
"the truth's conditional density along `v` is at most `(s√(2π))⁻¹` at conditional spread `s`". -/
theorem gaussianReal_spread_apply_le (m : ℝ) {s : ℝ} (hs : 0 < s) (t : Set ℝ) :
    gaussianReal m (Real.toNNReal (s ^ 2)) t
      ≤ ENNReal.ofReal (s * Real.sqrt (2 * π))⁻¹ * volume t := by
  have hv : Real.toNNReal (s ^ 2) ≠ 0 :=
    fun h => absurd (Real.toNNReal_eq_zero.mp h) (not_le.mpr (by positivity))
  have hcoe : ((Real.toNNReal (s ^ 2) : ℝ≥0) : ℝ) = s ^ 2 := Real.coe_toNNReal _ (sq_nonneg s)
  have hsqrt : Real.sqrt (2 * π * (Real.toNNReal (s ^ 2) : ℝ≥0)) = s * Real.sqrt (2 * π) := by
    rw [hcoe, mul_comm (2 * π) (s ^ 2), Real.sqrt_mul (sq_nonneg s), Real.sqrt_sq hs.le]
  have h := gaussianReal_apply_le_mul_volume m hv t
  rwa [hsqrt] at h

/-! ## The per-member paper bound -/

/-- **The paper's displayed per-member bound**: if the blind conditional given the information is
`N(m(w), s²)` a.e., then `Pr_ρ(θ⋆ ∈ Î) ≤ (s√(2π))⁻¹ · E|Î|`.  (`E|Î|` under the information
marginal `ρ.fst`.) -/
theorem coverage_le_of_gaussian_fiber (ρ : Measure (𝒲 × ℝ)) [IsFiniteMeasure ρ]
    (m : 𝒲 → ℝ) {s : ℝ} (hs : 0 < s)
    (hcond : ∀ᵐ w ∂ρ.fst, ρ.condKernel w = gaussianReal (m w) (Real.toNNReal (s ^ 2)))
    {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi) :
    coverage ρ lo hi ≤ ENNReal.ofReal (s * Real.sqrt (2 * π))⁻¹ * expLength ρ.fst lo hi := by
  refine coverage_le_of_condKernel_bound ρ ?_ hlo hhi
  filter_upwards [hcond] with w hw t
  rw [hw]
  exact gaussianReal_spread_apply_le (m w) hs t

/-! ## The family step -/

/-- **The family arithmetic**: if each member's coverage is at most `B i · L` with a common
finite `L` (the expected length, whose law — hence value — is the same for every member) and the
density bounds `B i` infimum to zero, then the infimum coverage is zero. -/
theorem iInf_eq_zero_of_le_mul {ι : Type*} [Nonempty ι] (c B : ι → ℝ≥0∞) {L : ℝ≥0∞}
    (hL : L ≠ ⊤) (hle : ∀ i, c i ≤ B i * L) (hB : ⨅ i, B i = 0) :
    ⨅ i, c i = 0 := by
  have hbot : (⨅ i, c i) = ⊥ := by
    refine iInf_eq_bot.mpr fun ε hε => ?_
    have hεpos : (0 : ℝ≥0∞) < ε := by simpa using hε
    rcases eq_or_ne L 0 with h0 | h0
    · have hc0 : c (Classical.arbitrary ι) ≤ 0 := by
        simpa [h0] using hle (Classical.arbitrary ι)
      exact ⟨Classical.arbitrary ι, lt_of_le_of_lt hc0 hεpos⟩
    · have hpos : 0 < ε / L := ENNReal.div_pos hεpos.ne' hL
      have hlt : ⨅ i, B i < ε / L := by rw [hB]; exact hpos
      obtain ⟨i, hi⟩ := iInf_lt_iff.mp hlt
      refine ⟨i, lt_of_le_of_lt (hle i) ?_⟩
      calc B i * L < (ε / L) * L := by
            rw [mul_comm (B i) L, mul_comm (ε / L) L]
            exact ENNReal.mul_lt_mul_right h0 hL hi
        _ = ε := ENNReal.div_mul_cancel h0 hL
  simpa using hbot

/-- **Unbounded spread sends the Gaussian density bounds to zero**: if the family's conditional
spreads `s i > 0` are unbounded, then `⨅ i, (s i √(2π))⁻¹ = 0` (in `ℝ≥0∞`).  This is the paper's
`sup_{p∈𝒫} s_p = ∞` clause, realized by the witness `W = w·vvᵀ` with `w` free. -/
theorem iInf_gaussian_bound_eq_zero {ι : Type*} (s : ι → ℝ) (hs : ∀ i, 0 < s i)
    (hsup : ∀ M : ℝ, ∃ i, M < s i) :
    ⨅ i, ENNReal.ofReal (s i * Real.sqrt (2 * π))⁻¹ = 0 := by
  haveI : Nonempty ι := ⟨(hsup 0).choose⟩
  have h2π : (0 : ℝ) < Real.sqrt (2 * π) := Real.sqrt_pos.mpr (by positivity)
  have hbot : (⨅ i, ENNReal.ofReal (s i * Real.sqrt (2 * π))⁻¹) = ⊥ := by
    refine iInf_eq_bot.mpr fun ε hε => ?_
    rcases eq_or_ne ε ⊤ with rfl | hεtop
    · exact ⟨Classical.arbitrary ι, ENNReal.ofReal_lt_top⟩
    · have hεne : ε ≠ 0 := by simpa using hε.ne'
      have hεpos : 0 < ε.toReal := ENNReal.toReal_pos hεne hεtop
      obtain ⟨i, hi⟩ := hsup ((ε.toReal)⁻¹ * (Real.sqrt (2 * π))⁻¹)
      have hprod : (ε.toReal)⁻¹ < s i * Real.sqrt (2 * π) := by
        calc (ε.toReal)⁻¹
            = (ε.toReal)⁻¹ * (Real.sqrt (2 * π))⁻¹ * Real.sqrt (2 * π) := by
              rw [mul_assoc, inv_mul_cancel₀ h2π.ne', mul_one]
          _ < s i * Real.sqrt (2 * π) := mul_lt_mul_of_pos_right hi h2π
      have hlt : (s i * Real.sqrt (2 * π))⁻¹ < ε.toReal :=
        (inv_lt_comm₀ (mul_pos (hs i) h2π) hεpos).mpr hprod
      refine ⟨i, ?_⟩
      calc ENNReal.ofReal (s i * Real.sqrt (2 * π))⁻¹
          < ENNReal.ofReal ε.toReal := (ENNReal.ofReal_lt_ofReal_iff hεpos).mpr hlt
        _ = ε := ENNReal.ofReal_toReal hεtop
  simpa using hbot

/-! ## The corollary -/

/-- **The display of `c:honest`**: over a family of truths whose joint laws `ρ i` of
`(information, blind coordinate)` share the SAME information marginal `μ` (`hfst` — the
`c:nonident` constancy, see module docstring) and whose blind conditionals are Gaussian with
spreads `s i` unbounded above (`hcond`, `hsup` — the linear–Gaussian witness), every measurable
interval rule of finite expected length has `inf_p Pr_p(θ⋆ ∈ Î) = 0`. -/
theorem iInf_coverage_eq_zero {ι : Type*} (ρ : ι → Measure (𝒲 × ℝ))
    [∀ i, IsFiniteMeasure (ρ i)] (μ : Measure 𝒲) (hfst : ∀ i, (ρ i).fst = μ)
    (m : ι → 𝒲 → ℝ) (s : ι → ℝ) (hs : ∀ i, 0 < s i)
    (hcond : ∀ i, ∀ᵐ w ∂μ, (ρ i).condKernel w = gaussianReal (m i w) (Real.toNNReal (s i ^ 2)))
    (hsup : ∀ M : ℝ, ∃ i, M < s i)
    {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi)
    (hlen : expLength μ lo hi ≠ ⊤) :
    ⨅ i, coverage (ρ i) lo hi = 0 := by
  haveI : Nonempty ι := ⟨(hsup 0).choose⟩
  refine iInf_eq_zero_of_le_mul _ (fun i => ENNReal.ofReal (s i * Real.sqrt (2 * π))⁻¹) hlen
    (fun i => ?_) (iInf_gaussian_bound_eq_zero s hs hsup)
  have h := coverage_le_of_gaussian_fiber (ρ i) (m i) (hs i)
    (by rw [hfst i]; exact hcond i) hlo hhi
  rwa [hfst i] at h

/-- **Corollary `c:honest` — no finite blind interval is an inference.**  In the setting of
`iInf_coverage_eq_zero` (constant information law, Gaussian blind conditionals of unbounded
spread — the linear–Gaussian witness; see the module docstring for the flagged hypotheses), any
interval rule achieving coverage `≥ 1 − α` with `α < 1` **uniformly** over the family has
**infinite expected length**. -/
theorem uniform_coverage_infinite_expected_length {ι : Type*} (ρ : ι → Measure (𝒲 × ℝ))
    [∀ i, IsFiniteMeasure (ρ i)] (μ : Measure 𝒲) (hfst : ∀ i, (ρ i).fst = μ)
    (m : ι → 𝒲 → ℝ) (s : ι → ℝ) (hs : ∀ i, 0 < s i)
    (hcond : ∀ i, ∀ᵐ w ∂μ, (ρ i).condKernel w = gaussianReal (m i w) (Real.toNNReal (s i ^ 2)))
    (hsup : ∀ M : ℝ, ∃ i, M < s i)
    {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi)
    {α : ℝ} (hα : α < 1)
    (hcov : ∀ i, ENNReal.ofReal (1 - α) ≤ coverage (ρ i) lo hi) :
    expLength μ lo hi = ⊤ := by
  by_contra hlen
  have h0 := iInf_coverage_eq_zero ρ μ hfst m s hs hcond hsup hlo hhi hlen
  have hle : ENNReal.ofReal (1 - α) ≤ ⨅ i, coverage (ρ i) lo hi := le_iInf hcov
  rw [h0, le_zero_iff] at hle
  exact absurd (ENNReal.ofReal_eq_zero.mp hle) (not_le.mpr (by linarith))

/-!
## Reconciliation — what `c:honest`'s formalization needs vs. what the paper states

**(a)** already in the text; **(b)** load-bearing, taken as a named hypothesis; **(c)** automatic.

* "Let `Î` be ANY interval rule … computed from the recorded measurements and the archive —
  learned or classical, randomized or not" — **(a/c)**: `Î` is any pair of measurable endpoint
  maps on the abstract information space `𝒲`; randomization folds into `𝒲` as an extra
  truth-independent coordinate.  Only measurability of `lo`, `hi` is consumed (**(c)**: any
  implementable rule is measurable).
* "Its law is the same for every member of `𝒫`" — **(b)**: the hypothesis `hfst` (one shared
  information marginal `μ`).  Its physical justification is `c:nonident`, machine-checked in
  `NonIdentifiability.lean`; the identification of the abstract `𝒲`-marginal with the physical
  record+archive law is the one modeling step not formalized.
* "the truth's conditional density along `v` is at most `(s√(2π))⁻¹` at conditional spread `s`"
  — **(a, proved)**: `gaussianPDFReal_le` / `gaussianReal_spread_apply_le`; the Gaussian-ness of
  the blind fiber under the witness is the named hypothesis `hcond` (**(b)**).
* "the witness … places in `𝒫` truths whose conditional spread along `v` is arbitrarily large"
  — **(b)**: the hypothesis `hsup`; that each such truth keeps the data law fixed is the
  machine-checked `NonIdentifiability.witness_cov_annihilated` / `witness_mean_annihilated`.
* "the coverage bound is the density bound integrated over that law" — **(a, proved)**:
  `coverage_compProd_le` is exactly this integration, via `Measure.compProd_apply`; the
  disintegration `ρ = ρ.fst ⊗ₘ ρ.condKernel` is mathlib's theorem, not an input.
* "`inf_p Pr_p(θ⋆ ∈ Î) ≤ E|Î|/(√(2π)·sup_p s_p) = 0` whenever `E|Î| < ∞`" — **(a, proved)**:
  `coverage_le_of_gaussian_fiber` (the per-member bound) + `iInf_coverage_eq_zero` (the `= 0`
  endpoint, with `sup s = ∞` consumed as `⨅ (s√(2π))⁻¹ = 0`).
* "an interval with coverage `1−α` uniformly over `𝒫` has infinite expected length" —
  **(a, proved)**: `uniform_coverage_infinite_expected_length`, the contrapositive.
* Working in `ℝ≥0∞` (coverage as measure, `E|Î|` as `lintegral`) — **(c)**: the paper's
  `E|Î| < ∞` is `expLength ≠ ⊤`; no integrability side conditions are needed, and the closed
  interval is the strongest case (open/half-open reports have no larger coverage, equal length).
-/

end HonestInterval

/-! ### Axiom audit

Each must print `propext, Classical.choice, Quot.sound` and never `sorryAx`. -/

#print axioms HonestInterval.coverage_compProd_le
#print axioms HonestInterval.coverage_le_of_condKernel_bound
#print axioms HonestInterval.gaussianPDFReal_le
#print axioms HonestInterval.gaussianReal_apply_le_mul_volume
#print axioms HonestInterval.gaussianReal_spread_apply_le
#print axioms HonestInterval.coverage_le_of_gaussian_fiber
#print axioms HonestInterval.iInf_eq_zero_of_le_mul
#print axioms HonestInterval.iInf_gaussian_bound_eq_zero
#print axioms HonestInterval.iInf_coverage_eq_zero
#print axioms HonestInterval.uniform_coverage_infinite_expected_length
