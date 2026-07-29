import BlindFreeze
import Mathlib.MeasureTheory.Constructions.BorelSpace.Order
import Mathlib.MeasureTheory.Group.Arithmetic
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

/-!
# `r:selfcheck` and the sharpness witnesses of the paper

This file machine-checks three items of the paper:

## Part 1 — `r:selfcheck`: SBC's rank is uniform under EVERY exchangeable prior

Simulation-based calibration draws a test truth `θ = f 0`, simulates a measurement, samples `N`
posterior draws `θᵢ = f i.succ`, and ranks the truth among the draws.  The paper's remark
(`r:selfcheck`) rests on one mathematical fact: whenever the `N+1` variables
`(θ, θ₁, …, θ_N)` are **exchangeable** and a.s. pairwise distinct, the rank
`R = #{i : θᵢ < θ}` is **uniform on `{0, …, N}`** — *whatever* the generating prior.

* Exchangeability is taken in its cleanest measure form: the joint law `μ` on `Fin (N+1) → ℝ` is
  invariant under every coordinate permutation, `map (· ∘ σ) μ = μ`.
* A.s. distinctness is `∀ᵐ f ∂μ, Function.Injective f`.
* `sbcRank_uniform` — `μ {sbcRank = k} = (N+1)⁻¹` for every `k ≤ N` (and `= 0` beyond,
  `sbcRank_empty_of_gt`).
* `sbcRank_law_prior_independent` / `sbc_power_eq_size` — the paper's actual use: the rank
  statistic's LAW is the same for every exchangeable a.s.-distinct prior (curated or truth-trained),
  so any rank-based test rejects the curated prior with exactly the probability it rejects the
  oracle: **SBC's power against the freeze equals its size**.

Proof route (finite combinatorics + one pushforward): the position statistic
`rankAmong f i = #{j : f j < f i}` is (a) carried to `rankAmong f (σ i)` by precomposition with a
permutation, so exchangeability makes all `N+1` events `{rankAmong · i = k}` equiprobable (via the
transposition `swap 0 i`); (b) on the distinctness event, `i ↦ rankAmong f i` is a bijection onto
`{0, …, N}`, so those events partition the space; (c) hence each has mass `(N+1)⁻¹`.

## Part 2 — the kinked-penalty sharpness witness of `p:mapnearnull`

One dimension, forward operator `ε > 0`, Gaussian data term, kinked penalty
`R(x) = |x-1| + |x+1|` (convex, NOT strongly convex): `f_y(x) = (y-εx)²/2 + |x-1| + |x+1|`.

* `kink_uniqueMin` — for `|y| < ε` the unique global minimizer is `x̂(y) = y/ε` (on `[-1,1]` the
  penalty is exactly `2`, `kink_penalty_of_abs_le`; outside, the penalty alone exceeds the interior
  minimum).
* `kink_no_collapse` — hence `x̂` sweeps ALL of `(-1,1)` while `|y| < ε`: for every candidate
  collapse point `x₀` and every `r < 1` there is a data value `|y| < ε` whose minimizer sits at
  distance `> r` from `x₀`; the witness set of such `y` has positive Lebesgue measure
  (`kink_witness_posMeasure`) — the manuscript's "a set of positive measure however small `ε`".
* `kink_no_vanishing_bound` / `kink_no_power_bound` — the sharpness: **no** bound
  `|x̂(y) - x₀(ε)| ≤ C‖Av‖^κ = C ε^κ` (nor any vanishing modulus `ω(ε) → 0`) can hold for merely
  convex penalties: eventually as `ε → 0⁺` the bound fails at some `|y| < ε`.  Strong convexity in
  `p:mapnearnull` is therefore the sharp boundary, not a convenience.

## Part 3 — `r:sampler`: the freeze asks little of the archive's sampler

`sampler_freeze` specializes `BlindFreeze.mixture_freeze` to the remark's phrasing: if the legacy
pipeline's sampler returns, at (truth-)a.e. measurement, a draw whose blind-fiber conditional is the
regularizer's `κreg` — exact fiber conditionals, however inaccurate the resolved part `S_R` — then
the pooled (curated) prior still disintegrates with the SAME blind conditional `κreg`
(`sampler_freeze_fiber_conditional`): the freeze holds for any law of the resolved coordinates.

## Honesty

No `sorry`/`admit`/`native_decide`, no new axioms.  Every hypothesis is stated explicitly in the
theorems; none is hidden.  In Part 1 the exchangeability and a.s.-distinctness hypotheses are
exactly the remark's ("whenever the sampler targets the prior that generated the test truth, that
truth and the posterior draws are exchangeable" — the exchangeability itself is a *modelling*
premise there, consumed here as a hypothesis, not proved from a sampler model).  In Part 2 the
minimizer is identified exactly (no hypothesis beyond `ε > 0`, `|y| < ε`).  In Part 3 the "sampler
has exact blind-fiber conditionals" premise is the remark's own displayed hypothesis.
`#print axioms` for every public result confirms `[propext, Classical.choice, Quot.sound]` only.
-/

open MeasureTheory ProbabilityTheory Filter Finset
open scoped ENNReal

namespace SelfCheckAndSharpness

/-! ## Part 1 — `r:selfcheck`: the SBC rank is uniform from exchangeability -/

section SelfCheck

variable {N : ℕ}

/-- The position statistic: how many of the `N+1` coordinates lie strictly below coordinate `i`.
Since `f i < f i` is false, coordinate `i` itself never counts. -/
noncomputable def rankAmong (f : Fin (N + 1) → ℝ) (i : Fin (N + 1)) : ℕ :=
  #{j | f j < f i}

/-- The SBC rank of the test truth `θ = f 0` among the `N` posterior draws `θᵢ = f i.succ`:
`R = #{i : θᵢ < θ}` (the paper's rank statistic in `r:selfcheck`). -/
noncomputable def sbcRank (f : Fin (N + 1) → ℝ) : ℕ :=
  #{i : Fin N | f i.succ < f 0}

/-- The SBC rank **is** the position statistic of coordinate `0`: counting the draws below the
truth equals counting all coordinates below the truth, because the truth is never below itself. -/
theorem sbcRank_eq (f : Fin (N + 1) → ℝ) : sbcRank f = rankAmong f 0 := by
  refine Finset.card_bij (fun i _ => Fin.succ i) ?_ ?_ ?_
  · intro a ha
    rw [Finset.mem_filter] at ha ⊢
    exact ⟨Finset.mem_univ _, ha.2⟩
  · intro a _ b _ h
    exact Fin.succ_injective _ h
  · intro j hj
    rw [Finset.mem_filter] at hj
    have hj0 : j ≠ 0 := fun h => absurd (h ▸ hj.2) (lt_irrefl _)
    obtain ⟨i, hi⟩ := Fin.exists_succ_eq.mpr hj0
    refine ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩, hi⟩
    rw [hi]
    exact hj.2

/-- The rank never exceeds `N` (coordinate `i` itself never counts). -/
theorem rankAmong_le (f : Fin (N + 1) → ℝ) (i : Fin (N + 1)) : rankAmong f i ≤ N := by
  show #{j | f j < f i} ≤ N
  have hsub : ({j | f j < f i} : Finset (Fin (N + 1))) ⊆ Finset.univ.erase i := by
    intro j hj
    rw [Finset.mem_filter] at hj
    rw [Finset.mem_erase]
    refine ⟨?_, Finset.mem_univ _⟩
    intro hji
    rw [hji] at hj
    exact lt_irrefl _ hj.2
  have h1 := Finset.card_le_card hsub
  rwa [Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ, Fintype.card_fin,
    Nat.add_sub_cancel] at h1

theorem sbcRank_le (f : Fin (N + 1) → ℝ) : sbcRank f ≤ N := by
  rw [sbcRank_eq]; exact rankAmong_le f 0

/-- A strictly smaller coordinate has a strictly smaller rank (its below-set is a strict subset). -/
theorem rankAmong_lt_of_lt {f : Fin (N + 1) → ℝ} {i j : Fin (N + 1)} (h : f i < f j) :
    rankAmong f i < rankAmong f j := by
  have hsub : ({l | f l < f i} : Finset (Fin (N + 1)))
      ⊆ ({l | f l < f j} : Finset (Fin (N + 1))) := by
    intro l hl
    rw [Finset.mem_filter] at hl ⊢
    exact ⟨hl.1, hl.2.trans h⟩
  exact Finset.card_lt_card ((Finset.ssubset_iff_of_subset hsub).mpr
    ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ _, h⟩,
      fun hmem => lt_irrefl _ (Finset.mem_filter.mp hmem).2⟩)

/-- On a vector with pairwise-distinct coordinates the rank map is injective. -/
theorem rankAmong_injective {f : Fin (N + 1) → ℝ} (hf : Function.Injective f) :
    Function.Injective (rankAmong f) := by
  intro i j hij
  rcases lt_trichotomy (f i) (f j) with h | h | h
  · exact absurd hij (ne_of_lt (rankAmong_lt_of_lt h))
  · exact hf h
  · exact absurd hij (ne_of_gt (rankAmong_lt_of_lt h))

/-- On a vector with pairwise-distinct coordinates every rank value `k ≤ N` is attained: the rank
map is an injection of `Fin (N+1)` into `{0,…,N}`, hence a bijection. -/
theorem exists_rankAmong_eq {f : Fin (N + 1) → ℝ} (hf : Function.Injective f) {k : ℕ}
    (hk : k ≤ N) : ∃ i, rankAmong f i = k := by
  have hinj : Function.Injective
      (fun i : Fin (N + 1) => (⟨rankAmong f i, Nat.lt_succ_of_le (rankAmong_le f i)⟩ :
        Fin (N + 1))) := by
    intro i j hij
    exact rankAmong_injective hf (congrArg Fin.val hij)
  obtain ⟨i, hi⟩ := (Finite.injective_iff_bijective.mp hinj).2 ⟨k, Nat.lt_succ_of_le hk⟩
  exact ⟨i, congrArg Fin.val hi⟩

/-- The rank as a real-valued sum of indicators (the measurability vehicle). -/
theorem rankAmong_cast (f : Fin (N + 1) → ℝ) (i : Fin (N + 1)) :
    ((rankAmong f i : ℕ) : ℝ) = ∑ j, if f j < f i then (1 : ℝ) else 0 := by
  show ((#{j | f j < f i} : ℕ) : ℝ) = _
  rw [Finset.card_filter]
  push_cast
  rfl

/-- Each rank level set is measurable (as the preimage of a point under a finite sum of
indicators of the open half-space events `{f j < f i}`). -/
theorem measurableSet_rankAmong (i : Fin (N + 1)) (k : ℕ) :
    MeasurableSet {f : Fin (N + 1) → ℝ | rankAmong f i = k} := by
  have hmeas : Measurable fun f : Fin (N + 1) → ℝ => ∑ j, if f j < f i then (1 : ℝ) else 0 :=
    Finset.measurable_sum _ fun j _ =>
      Measurable.ite (measurableSet_lt (measurable_pi_apply j) (measurable_pi_apply i))
        measurable_const measurable_const
  have heq : {f : Fin (N + 1) → ℝ | rankAmong f i = k}
      = (fun f : Fin (N + 1) → ℝ => ∑ j, if f j < f i then (1 : ℝ) else 0) ⁻¹' {(k : ℝ)} := by
    ext f
    simp only [Set.mem_setOf_eq, Set.mem_preimage, Set.mem_singleton_iff, ← rankAmong_cast]
    exact ⟨fun h => by rw [h], fun h => Nat.cast_injective h⟩
  rw [heq]
  exact hmeas (measurableSet_singleton _)

/-- The rank statistic is a measurable function into `ℕ`. -/
theorem measurable_sbcRank : Measurable (sbcRank : (Fin (N + 1) → ℝ) → ℕ) := by
  have h : (sbcRank : (Fin (N + 1) → ℝ) → ℕ) = fun f => rankAmong f 0 := funext sbcRank_eq
  rw [h]
  exact measurable_to_nat fun f => measurableSet_rankAmong 0 (rankAmong f 0)

/-- Precomposing the vector with a permutation carries the position statistic along:
`rankAmong (f ∘ σ) i = rankAmong f (σ i)`. -/
theorem rankAmong_comp_perm (σ : Equiv.Perm (Fin (N + 1))) (f : Fin (N + 1) → ℝ)
    (i : Fin (N + 1)) : rankAmong (f ∘ σ) i = rankAmong f (σ i) := by
  show #{j | (f ∘ σ) j < (f ∘ σ) i} = #{j | f j < f (σ i)}
  exact Finset.card_equiv σ fun j => by
    simp [Function.comp]

/-- The distinctness event: all `N+1` coordinates pairwise distinct. -/
def distinctSet (N : ℕ) : Set (Fin (N + 1) → ℝ) := {f | Function.Injective f}

theorem measurableSet_distinctSet : MeasurableSet (distinctSet N) := by
  have h : distinctSet N
      = ⋂ (i) (j) (_ : i ≠ j), ({f : Fin (N + 1) → ℝ | f i < f j} ∪ {f | f j < f i}) := by
    ext f
    simp only [distinctSet, Set.mem_setOf_eq, Set.mem_iInter, Set.mem_union]
    constructor
    · intro hf i j hij
      rcases lt_trichotomy (f i) (f j) with h | h | h
      · exact Or.inl h
      · exact absurd (hf h) hij
      · exact Or.inr h
    · intro h a b hab
      by_contra hne
      rcases h a b hne with h' | h' <;> simp only [hab, lt_self_iff_false] at h'
  rw [h]
  exact MeasurableSet.iInter fun i => MeasurableSet.iInter fun j => MeasurableSet.iInter fun _ =>
    (measurableSet_lt (measurable_pi_apply i) (measurable_pi_apply j)).union
      (measurableSet_lt (measurable_pi_apply j) (measurable_pi_apply i))

/-- **Exchangeability equidistributes the `N+1` position events.**  If the joint law is invariant
under every coordinate permutation, then for each `k` the events `{rankAmong · i = k}` have the
same probability for every coordinate `i` (via the transposition `swap 0 i`). -/
theorem measure_rankAmong_eq (μ : Measure (Fin (N + 1) → ℝ))
    (hexch : ∀ σ : Equiv.Perm (Fin (N + 1)), Measure.map (fun f => f ∘ σ) μ = μ)
    (i : Fin (N + 1)) (k : ℕ) :
    μ {f | rankAmong f i = k} = μ {f | rankAmong f 0 = k} := by
  have hmeasσ : ∀ σ : Equiv.Perm (Fin (N + 1)),
      Measurable fun f : Fin (N + 1) → ℝ => f ∘ σ := fun σ =>
    measurable_pi_lambda _ fun j => measurable_pi_apply (σ j)
  have hpre : (fun f : Fin (N + 1) → ℝ => f ∘ Equiv.swap 0 i) ⁻¹' {g | rankAmong g 0 = k}
      = {f | rankAmong f i = k} := by
    ext f
    simp only [Set.mem_preimage, Set.mem_setOf_eq, rankAmong_comp_perm, Equiv.swap_apply_left]
  calc μ {f | rankAmong f i = k}
      = μ ((fun f : Fin (N + 1) → ℝ => f ∘ Equiv.swap 0 i) ⁻¹' {g | rankAmong g 0 = k}) := by
        rw [hpre]
    _ = (Measure.map (fun f : Fin (N + 1) → ℝ => f ∘ Equiv.swap 0 i) μ)
          {g | rankAmong g 0 = k} :=
        (Measure.map_apply (hmeasσ _) (measurableSet_rankAmong 0 k)).symm
    _ = μ {g | rankAmong g 0 = k} := by rw [hexch (Equiv.swap 0 i)]

/-- **`r:selfcheck`, the core fact: the SBC rank is uniform on `{0,…,N}`.**

For `N+1` exchangeable (permutation-invariant joint law) and a.s. pairwise-distinct real random
variables `(θ, θ₁, …, θ_N)`, the rank `R = #{i : θᵢ < θ}` satisfies
`μ {R = k} = (N+1)⁻¹` for every `k ≤ N`, whatever the prior that generated the draws. -/
theorem rankAmong_zero_uniform (μ : Measure (Fin (N + 1) → ℝ)) [IsProbabilityMeasure μ]
    (hexch : ∀ σ : Equiv.Perm (Fin (N + 1)), Measure.map (fun f => f ∘ σ) μ = μ)
    (hdist : ∀ᵐ f ∂μ, Function.Injective f) {k : ℕ} (hk : k ≤ N) :
    μ {f | rankAmong f 0 = k} = ((N : ℝ≥0∞) + 1)⁻¹ := by
  classical
  -- the distinctness event carries full mass
  have hDmeas : MeasurableSet (distinctSet N) := measurableSet_distinctSet
  have hDc : μ (distinctSet N)ᶜ = 0 := by
    rw [distinctSet, Set.compl_setOf]
    exact ae_iff.mp hdist
  have hDone : μ (distinctSet N) = 1 := by
    have h := measure_add_measure_compl (μ := μ) hDmeas
    rwa [hDc, add_zero, measure_univ] at h
  -- the N+1 position events, restricted to the distinctness event, partition it
  have hEmeas : ∀ i : Fin (N + 1),
      MeasurableSet ({f | rankAmong f i = k} ∩ distinctSet N) := fun i =>
    (measurableSet_rankAmong i k).inter hDmeas
  have hdisj : Pairwise (Function.onFun Disjoint
      fun i : Fin (N + 1) => {f | rankAmong f i = k} ∩ distinctSet N) := by
    intro i j hij
    simp only [Function.onFun]
    rw [Set.disjoint_left]
    rintro f ⟨hfi, hfD⟩ ⟨hfj, -⟩
    exact hij (rankAmong_injective hfD (hfi.trans hfj.symm))
  have hcover : (⋃ i : Fin (N + 1), {f | rankAmong f i = k} ∩ distinctSet N)
      = distinctSet N := by
    apply Set.Subset.antisymm
    · exact Set.iUnion_subset fun i => Set.inter_subset_right
    · intro f hf
      obtain ⟨i, hi⟩ := exists_rankAmong_eq hf hk
      exact Set.mem_iUnion.mpr ⟨i, hi, hf⟩
  have hsum : ∑' i : Fin (N + 1), μ ({f | rankAmong f i = k} ∩ distinctSet N) = 1 := by
    rw [← measure_iUnion hdisj hEmeas, hcover, hDone]
  -- dropping the full-mass event and using exchangeability, all summands are the k-mass at 0
  have hEi : ∀ i : Fin (N + 1),
      μ ({f | rankAmong f i = k} ∩ distinctSet N) = μ {f | rankAmong f 0 = k} := by
    intro i
    have h := measure_inter_add_sdiff (μ := μ) {f | rankAmong f i = k} hDmeas
    have h2 : μ ({f | rankAmong f i = k} \ distinctSet N) = 0 :=
      le_antisymm (le_trans (measure_mono fun x hx => hx.2) hDc.le) zero_le
    rw [h2, add_zero] at h
    rw [h]
    exact measure_rankAmong_eq μ hexch i k
  have hsum' : ((N : ℝ≥0∞) + 1) * μ {f | rankAmong f 0 = k} = 1 := by
    have hconst : ∑' i : Fin (N + 1), μ ({f | rankAmong f i = k} ∩ distinctSet N)
        = ∑ _i : Fin (N + 1), μ {f | rankAmong f 0 = k} := by
      rw [tsum_fintype]
      exact Finset.sum_congr rfl fun i _ => hEi i
    rw [hconst, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      Nat.cast_add, Nat.cast_one] at hsum
    exact hsum
  -- solve for the mass
  have hne : ((N : ℝ≥0∞) + 1) ≠ 0 :=
    (lt_of_lt_of_le zero_lt_one le_add_self).ne'
  have htop : ((N : ℝ≥0∞) + 1) ≠ ⊤ :=
    ENNReal.add_ne_top.mpr ⟨ENNReal.natCast_ne_top N, ENNReal.one_ne_top⟩
  calc μ {f | rankAmong f 0 = k} = 1 * μ {f | rankAmong f 0 = k} := (one_mul _).symm
    _ = (((N : ℝ≥0∞) + 1)⁻¹ * ((N : ℝ≥0∞) + 1)) * μ {f | rankAmong f 0 = k} := by
        rw [ENNReal.inv_mul_cancel hne htop]
    _ = ((N : ℝ≥0∞) + 1)⁻¹ * (((N : ℝ≥0∞) + 1) * μ {f | rankAmong f 0 = k}) := by
        rw [mul_assoc]
    _ = ((N : ℝ≥0∞) + 1)⁻¹ * 1 := by rw [hsum']
    _ = ((N : ℝ≥0∞) + 1)⁻¹ := mul_one _

/-- **`r:selfcheck` in the paper's rank: SBC's rank statistic is uniform.**  For exchangeable,
a.s. pairwise-distinct `(θ, θ₁, …, θ_N)` the rank `R = #{i : θᵢ < θ}` of the test truth among the
posterior draws is uniform on `{0, …, N}`. -/
theorem sbcRank_uniform (μ : Measure (Fin (N + 1) → ℝ)) [IsProbabilityMeasure μ]
    (hexch : ∀ σ : Equiv.Perm (Fin (N + 1)), Measure.map (fun f => f ∘ σ) μ = μ)
    (hdist : ∀ᵐ f ∂μ, Function.Injective f) {k : ℕ} (hk : k ≤ N) :
    μ {f | sbcRank f = k} = ((N : ℝ≥0∞) + 1)⁻¹ := by
  have h : {f : Fin (N + 1) → ℝ | sbcRank f = k} = {f | rankAmong f 0 = k} :=
    Set.ext fun f => by simp only [Set.mem_setOf_eq, sbcRank_eq]
  rw [h]
  exact rankAmong_zero_uniform μ hexch hdist hk

/-- Ranks beyond `N` are impossible. -/
theorem sbcRank_empty_of_gt {k : ℕ} (hk : N < k) :
    {f : Fin (N + 1) → ℝ | sbcRank f = k} = ∅ := by
  ext f
  simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
  intro hf
  exact absurd hk (Nat.not_lt.mpr (hf ▸ sbcRank_le f))

/-- **The rank law does not depend on which exchangeable prior generated the draws.**  Any two
exchangeable a.s.-distinct joint laws — e.g. the SBC run under the curated prior and under a
truth-trained oracle — push forward to the SAME distribution of the rank statistic (the uniform
one).  This is the precise content of "SBC prices a prior's consistency with its own posterior,
never its fidelity to the truth". -/
theorem sbcRank_law_prior_independent
    (μ₁ μ₂ : Measure (Fin (N + 1) → ℝ)) [IsProbabilityMeasure μ₁] [IsProbabilityMeasure μ₂]
    (hexch₁ : ∀ σ : Equiv.Perm (Fin (N + 1)), Measure.map (fun f => f ∘ σ) μ₁ = μ₁)
    (hdist₁ : ∀ᵐ f ∂μ₁, Function.Injective f)
    (hexch₂ : ∀ σ : Equiv.Perm (Fin (N + 1)), Measure.map (fun f => f ∘ σ) μ₂ = μ₂)
    (hdist₂ : ∀ᵐ f ∂μ₂, Function.Injective f) :
    Measure.map sbcRank μ₁ = Measure.map sbcRank μ₂ := by
  apply Measure.ext_of_singleton
  intro k
  rw [Measure.map_apply measurable_sbcRank (measurableSet_singleton k),
    Measure.map_apply measurable_sbcRank (measurableSet_singleton k)]
  have hpre : (sbcRank : (Fin (N + 1) → ℝ) → ℕ) ⁻¹' {k}
      = {f : Fin (N + 1) → ℝ | sbcRank f = k} := rfl
  rw [hpre]
  by_cases hk : k ≤ N
  · rw [sbcRank_uniform μ₁ hexch₁ hdist₁ hk, sbcRank_uniform μ₂ hexch₂ hdist₂ hk]
  · rw [sbcRank_empty_of_gt (not_le.mp hk), measure_empty, measure_empty]

/-- **SBC's power against the freeze equals its size.**  Any rank-based test — any rejection
region `T` of rank values — rejects with the SAME probability under every exchangeable
a.s.-distinct generating prior.  In particular a curated (frozen-blind) prior and a truth-trained
oracle are rejected with identical probability: the check cannot see the freeze. -/
theorem sbc_power_eq_size (T : Set ℕ)
    (μ₁ μ₂ : Measure (Fin (N + 1) → ℝ)) [IsProbabilityMeasure μ₁] [IsProbabilityMeasure μ₂]
    (hexch₁ : ∀ σ : Equiv.Perm (Fin (N + 1)), Measure.map (fun f => f ∘ σ) μ₁ = μ₁)
    (hdist₁ : ∀ᵐ f ∂μ₁, Function.Injective f)
    (hexch₂ : ∀ σ : Equiv.Perm (Fin (N + 1)), Measure.map (fun f => f ∘ σ) μ₂ = μ₂)
    (hdist₂ : ∀ᵐ f ∂μ₂, Function.Injective f) :
    μ₁ (sbcRank ⁻¹' T) = μ₂ (sbcRank ⁻¹' T) := by
  have hT : MeasurableSet T := MeasurableSet.of_discrete
  rw [← Measure.map_apply measurable_sbcRank hT, ← Measure.map_apply measurable_sbcRank hT,
    sbcRank_law_prior_independent μ₁ μ₂ hexch₁ hdist₁ hexch₂ hdist₂]

end SelfCheck

/-! ## Part 2 — the kinked-penalty sharpness witness of `p:mapnearnull` -/

section KinkedPenalty

/-- The 1-D kinked-penalty objective: Gaussian data term for the scalar operator `ε`, penalty
`|x-1| + |x+1|` (convex, kinks at `±1`, NOT strongly convex — its flat middle is the witness). -/
noncomputable def kinkObj (ε y x : ℝ) : ℝ :=
  (y - ε * x) ^ 2 / 2 + (|x - 1| + |x + 1|)

/-- Between the kinks the penalty is exactly `2`. -/
theorem kink_penalty_of_abs_le {x : ℝ} (hx : |x| ≤ 1) : |x - 1| + |x + 1| = 2 := by
  rw [abs_le] at hx
  rw [abs_of_nonpos (by linarith), abs_of_nonneg (by linarith)]
  ring

/-- The interior minimum value: at `x = y/ε` the data term vanishes and the penalty is `2`. -/
theorem kinkObj_at_min {ε y : ℝ} (hε : 0 < ε) (hy : |y| < ε) : kinkObj ε y (y / ε) = 2 := by
  have ht : |y / ε| < 1 := by
    rw [abs_div, abs_of_pos hε]
    exact (div_lt_one hε).mpr hy
  have h0 : y - ε * (y / ε) = 0 := by
    rw [mul_comm ε (y / ε), div_mul_cancel₀ y hε.ne', sub_self]
  show (y - ε * (y / ε)) ^ 2 / 2 + (|y / ε - 1| + |y / ε + 1|) = 2
  rw [kink_penalty_of_abs_le ht.le, h0]
  norm_num

/-- **The unique global minimizer for `|y| < ε` is `x̂(y) = y/ε`** — strictly below every other
candidate.  On `[-1,1]` the penalty is constant (`= 2`) so the data term decides; outside, the
penalty alone already exceeds the interior minimum `2`. -/
theorem kink_uniqueMin {ε y : ℝ} (hε : 0 < ε) (hy : |y| < ε) :
    ∀ x, x ≠ y / ε → kinkObj ε y (y / ε) < kinkObj ε y x := by
  intro x hx
  have hεne : ε ≠ 0 := hε.ne'
  rw [kinkObj_at_min hε hy]
  by_cases h1 : x ≤ 1
  · by_cases h2 : -1 ≤ x
    · -- the flat stretch: the data term is strictly positive off the minimizer
      have hxabs : |x| ≤ 1 := abs_le.mpr ⟨h2, h1⟩
      have hne : y - ε * x ≠ 0 := by
        intro h
        exact hx (by rw [eq_div_iff hεne]; linear_combination -h)
      have hsq : 0 < (y - ε * x) ^ 2 := sq_pos_iff.mpr hne
      show (2 : ℝ) < (y - ε * x) ^ 2 / 2 + (|x - 1| + |x + 1|)
      rw [kink_penalty_of_abs_le hxabs]
      linarith
    · -- left of the kink at −1: the penalty is −2x > 2
      rw [not_le] at h2
      show (2 : ℝ) < (y - ε * x) ^ 2 / 2 + (|x - 1| + |x + 1|)
      rw [abs_of_nonpos (by linarith : x - 1 ≤ 0), abs_of_nonpos (by linarith : x + 1 ≤ 0)]
      have hsq := sq_nonneg (y - ε * x)
      linarith
  · -- right of the kink at 1: the penalty is 2x > 2
    rw [not_le] at h1
    show (2 : ℝ) < (y - ε * x) ^ 2 / 2 + (|x - 1| + |x + 1|)
    rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ x - 1), abs_of_nonneg (by linarith : (0 : ℝ) ≤ x + 1)]
    have hsq := sq_nonneg (y - ε * x)
    linarith

/-- The non-strict version: `y/ε` is a global minimizer. -/
theorem kink_isMin {ε y : ℝ} (hε : 0 < ε) (hy : |y| < ε) (x : ℝ) :
    kinkObj ε y (y / ε) ≤ kinkObj ε y x := by
  rcases eq_or_ne x (y / ε) with rfl | hx
  · exact le_refl _
  · exact (kink_uniqueMin hε hy x hx).le

/-- **The minimizer sweeps all of `(-1,1)` however small `ε`: no collapse point exists.**  For
every candidate point `x₀` and every `r < 1` there is a measurement `|y| < ε` whose unique global
minimizer `y/ε` sits at distance `> r` from `x₀`.  (Compare `p:map` at `ε = 0`: a fixed selection
reports one point — the near-null spread is discontinuous at the kernel.) -/
theorem kink_no_collapse {ε : ℝ} (hε : 0 < ε) (x₀ : ℝ) {r : ℝ} (hr₀ : 0 ≤ r) (hr₁ : r < 1) :
    ∃ y, |y| < ε ∧ r < |y / ε - x₀| ∧
      ∀ x, x ≠ y / ε → kinkObj ε y (y / ε) < kinkObj ε y x := by
  have key : ∀ t : ℝ, |t| < 1 → r < |t - x₀| →
      ∃ y, |y| < ε ∧ r < |y / ε - x₀| ∧
        ∀ x, x ≠ y / ε → kinkObj ε y (y / ε) < kinkObj ε y x := by
    intro t htabs htfar
    have hdiv : ε * t / ε = t := mul_div_cancel_left₀ t hε.ne'
    have hyabs : |ε * t| < ε := by
      rw [abs_mul, abs_of_pos hε]
      calc ε * |t| < ε * 1 := mul_lt_mul_of_pos_left htabs hε
        _ = ε := mul_one ε
    exact ⟨ε * t, hyabs, by rw [hdiv]; exact htfar, kink_uniqueMin hε hyabs⟩
  by_cases h : 0 ≤ x₀
  · refine key (-((r + 1) / 2)) ?_ ?_
    · rw [abs_neg, abs_of_nonneg (by linarith)]; linarith
    · rw [abs_of_nonpos (by linarith)]; linarith
  · rw [not_le] at h
    refine key ((r + 1) / 2) ?_ ?_
    · rw [abs_of_nonneg (by linarith)]; linarith
    · rw [abs_of_nonneg (by linarith)]; linarith

/-- The witnessing measurements form a set of **positive Lebesgue measure** (it is open and
nonempty) — the manuscript's "for `y` in a set of positive measure however small `ε`". -/
theorem kink_witness_posMeasure {ε : ℝ} (hε : 0 < ε) (x₀ : ℝ) {r : ℝ} (hr₀ : 0 ≤ r)
    (hr₁ : r < 1) : 0 < volume {y : ℝ | |y| < ε ∧ r < |y / ε - x₀|} := by
  have hopen : IsOpen {y : ℝ | |y| < ε ∧ r < |y / ε - x₀|} := by
    rw [Set.setOf_and]
    exact (isOpen_lt continuous_abs continuous_const).inter
      (isOpen_lt continuous_const
        (continuous_abs.comp ((continuous_id.div_const ε).sub continuous_const)))
  obtain ⟨y, hy1, hy2, -⟩ := kink_no_collapse hε x₀ hr₀ hr₁
  exact hopen.measure_pos volume ⟨y, hy1, hy2⟩

/-- **No vanishing modulus can bound the minimizer's excursion.**  For ANY putative bound
`ω(ε) → 0` as `ε → 0⁺` and ANY selection of collapse points `ε ↦ x₀(ε)`, eventually (as `ε → 0⁺`)
there is a measurement `|y| < ε` whose unique global minimizer violates
`|x̂(y) - x₀(ε)| ≤ ω(ε)`.  Merely convex penalties admit no collapse rate. -/
theorem kink_no_vanishing_bound (ω : ℝ → ℝ)
    (hω : Tendsto ω (nhdsWithin 0 (Set.Ioi 0)) (nhds 0)) (x₀ : ℝ → ℝ) :
    ∀ᶠ ε in nhdsWithin 0 (Set.Ioi 0), ∃ y, |y| < ε ∧ ω ε < |y / ε - x₀ ε| ∧
      ∀ x, x ≠ y / ε → kinkObj ε y (y / ε) < kinkObj ε y x := by
  have h1 : ∀ᶠ ε in nhdsWithin (0 : ℝ) (Set.Ioi 0), ω ε < 1 :=
    Filter.Tendsto.eventually_lt_const zero_lt_one hω
  have h2 : ∀ᶠ ε in nhdsWithin (0 : ℝ) (Set.Ioi 0), ε ∈ Set.Ioi (0 : ℝ) :=
    self_mem_nhdsWithin
  filter_upwards [h1, h2] with ε hω1 hε
  obtain ⟨y, hy1, hy2, hy3⟩ := kink_no_collapse (Set.mem_Ioi.mp hε) (x₀ ε)
    (le_max_right (ω ε) 0) (max_lt hω1 one_pos)
  exact ⟨y, hy1, lt_of_le_of_lt (le_max_left (ω ε) 0) hy2, hy3⟩

/-- **The sharpness of `p:mapnearnull`: no bound `C‖Av‖^κ` can hold for merely convex penalties.**
Here `‖Av‖ = ε`, and for EVERY constant `C`, EVERY exponent `κ > 0`, and EVERY selection of
collapse points `x₀(ε)`: eventually as `ε → 0⁺` some measurement `|y| < ε` has its unique global
minimizer at distance `> C ε^κ` from `x₀(ε)`.  Strong convexity of the penalty sections — exactly
what `MapNearNullTube.pinning_tube` consumes — is the sharp boundary. -/
theorem kink_no_power_bound (C : ℝ) {κ : ℝ} (hκ : 0 < κ) (x₀ : ℝ → ℝ) :
    ∀ᶠ ε in nhdsWithin 0 (Set.Ioi 0), ∃ y, |y| < ε ∧ C * ε ^ κ < |y / ε - x₀ ε| ∧
      ∀ x, x ≠ y / ε → kinkObj ε y (y / ε) < kinkObj ε y x := by
  have h0 : Tendsto (fun ε : ℝ => ε ^ κ) (nhds 0) (nhds ((0 : ℝ) ^ κ)) :=
    Real.continuousAt_rpow_const 0 κ (Or.inr hκ.le)
  rw [Real.zero_rpow hκ.ne'] at h0
  have h1 : Tendsto (fun ε : ℝ => C * ε ^ κ) (nhdsWithin 0 (Set.Ioi 0)) (nhds (C * 0)) :=
    Filter.Tendsto.const_mul C
      (h0.mono_left (nhdsWithin_le_nhds (s := Set.Ioi 0)))
  rw [mul_zero] at h1
  exact kink_no_vanishing_bound (fun ε => C * ε ^ κ) h1 x₀

end KinkedPenalty

/-! ## Part 3 — `r:sampler`: the freeze asks little of the archive's sampler -/

section SamplerFreeze

variable {Y R B : Type*} [MeasurableSpace Y] [MeasurableSpace R] [MeasurableSpace B]

/-- **`r:sampler`: an approximate sampler with exact blind-fiber conditionals still freezes.**

If the legacy pipeline's sampler returns, at (truth-)a.e. measurement `y`, a draw whose law
disintegrates as `S y = S_R y ⊗ₘ κreg` — resolved part `S_R y` arbitrary (however inaccurate),
blind-fiber conditional exactly the regularizer's `κreg` — then the pooled curated prior is
`(truth.bind S_R) ⊗ₘ κreg`: the SAME blind conditional `κreg`, only the resolved marginal mixed.
This is `BlindFreeze.mixture_freeze` ("a mixture of kernels with a common fiber conditional keeps
that conditional") in the archive-sampler phrasing of the remark. -/
theorem sampler_freeze (truth : Measure Y) [SFinite truth]
    (S : Y → Measure (R × B)) (S_R : Kernel Y R) [IsSFiniteKernel S_R]
    (κreg : Kernel R B) [IsSFiniteKernel κreg]
    (hS : ∀ᵐ y ∂truth, S y = S_R y ⊗ₘ κreg) :
    truth.bind S = (truth.bind S_R) ⊗ₘ κreg := by
  rw [Measure.bind_congr_right (f := S) (g := fun y => S_R y ⊗ₘ κreg) hS]
  exact BlindFreeze.mixture_freeze truth S_R κreg

/-- The remark as the paper uses it: the curated prior built from such a sampler **disintegrates
with the regularizer's own blind-fiber conditional** — the freeze, for ANY law of the resolved
coordinates.  (When the regularizer is `reg = νR ⊗ₘ κreg`, the shared kernel `κreg` here IS the
regularizer's fiber conditional.) -/
theorem sampler_freeze_fiber_conditional (truth : Measure Y) [SFinite truth]
    (S : Y → Measure (R × B)) (S_R : Kernel Y R) [IsSFiniteKernel S_R]
    (κreg : Kernel R B) [IsSFiniteKernel κreg]
    (hS : ∀ᵐ y ∂truth, S y = S_R y ⊗ₘ κreg) :
    ∃ m : Measure R, truth.bind S = m ⊗ₘ κreg :=
  ⟨truth.bind S_R, sampler_freeze truth S S_R κreg hS⟩

end SamplerFreeze

end SelfCheckAndSharpness

-- #print axioms audit (r:selfcheck — SBC rank uniformity)
#print axioms SelfCheckAndSharpness.sbcRank_eq
#print axioms SelfCheckAndSharpness.rankAmong_le
#print axioms SelfCheckAndSharpness.sbcRank_le
#print axioms SelfCheckAndSharpness.rankAmong_lt_of_lt
#print axioms SelfCheckAndSharpness.rankAmong_injective
#print axioms SelfCheckAndSharpness.exists_rankAmong_eq
#print axioms SelfCheckAndSharpness.measurableSet_rankAmong
#print axioms SelfCheckAndSharpness.measurable_sbcRank
#print axioms SelfCheckAndSharpness.rankAmong_comp_perm
#print axioms SelfCheckAndSharpness.measure_rankAmong_eq
#print axioms SelfCheckAndSharpness.rankAmong_zero_uniform
#print axioms SelfCheckAndSharpness.sbcRank_uniform
#print axioms SelfCheckAndSharpness.sbcRank_empty_of_gt
#print axioms SelfCheckAndSharpness.sbcRank_law_prior_independent
#print axioms SelfCheckAndSharpness.sbc_power_eq_size

-- #print axioms audit (p:mapnearnull sharpness — the kinked-penalty witness)
#print axioms SelfCheckAndSharpness.kink_penalty_of_abs_le
#print axioms SelfCheckAndSharpness.kinkObj_at_min
#print axioms SelfCheckAndSharpness.kink_uniqueMin
#print axioms SelfCheckAndSharpness.kink_isMin
#print axioms SelfCheckAndSharpness.kink_no_collapse
#print axioms SelfCheckAndSharpness.kink_witness_posMeasure
#print axioms SelfCheckAndSharpness.kink_no_vanishing_bound
#print axioms SelfCheckAndSharpness.kink_no_power_bound

-- #print axioms audit (r:sampler — the archive-sampler freeze)
#print axioms SelfCheckAndSharpness.sampler_freeze
#print axioms SelfCheckAndSharpness.sampler_freeze_fiber_conditional
