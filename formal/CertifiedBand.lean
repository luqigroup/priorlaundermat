import SelfCheckAndSharpness
import Mathlib.Data.Fin.Tuple.Sort
import Mathlib.Order.Interval.Finset.Fin

/-!
# `p:certify` — the certified blind interval (split-conformal validity)

This file machine-checks the paper's Proposition `p:certify`:

> *Let `x₁, …, x_k` be references exchangeable with the deployment truth and held out of the
> archive; score a candidate value by the prior's standardized blind residual
> `η(θ) = |θ − m_ρ(x_R)| / s_ρ`.  The band `{θ : η(θ) ≤ η₍⌈(k+1)(1−α)⌉₎}` is an interval, and it
> covers `θ⋆` with probability at least `1 − α` for every prior and every truth, with no
> assumption beyond exchangeability.*

## Decomposition

* **The rank/quantile lemma** (`le_orderStat_iff`, `measure_rank_le`,
  `score_quantile_coverage`): for `k+1` exchangeable, a.s. pairwise-distinct real scores the
  test score falls at or below the `j`-th order statistic of the `k` calibration scores with
  probability exactly `(j+1)/(k+1)` (0-indexed `j`), hence `≥ 1−α` once
  `(k+1)(1−α) ≤ j+1`.  The exchangeability-rank engine is **reused from
  `SelfCheckAndSharpness`** (`rankAmong_zero_uniform` — the SBC rank-uniformity theorem): the
  tie-free set identity `{test ≤ η₍ⱼ₎} = {rank ≤ j−1}` proved here reduces conformal coverage to
  exactly that rank law.
* **The band is an interval** (`mem_band_iff`, `band_eq_Icc`): `η(θ) = |θ − m|/s` with `s > 0`
  is quasi-convex, and `{θ : η(θ) ≤ c} = Icc (m − c·s) (m + c·s)` — proved as a set equality,
  unconditional in `c` (an empty `Icc` when `c < 0`).
* **Coverage of the band** (`coverage_of_exchangeable`, `certified_blind_band`, `certify`):
  combining the two.  `coverage_of_exchangeable` is the "for every prior, every truth" master:
  it quantifies over an *arbitrary* measurable score `η : X → ℝ` on an arbitrary underlying
  space and an *arbitrary* exchangeable joint law of `(θ⋆, x₁, …, x_k)` — the prior enters only
  through `η` (via `m_ρ, s_ρ`) and the argument never uses its values, which is exactly the
  paper's "the score's dependence on the prior is immaterial since the references are held
  out".  `certify` is the verbatim endpoint: the paper's index `⌈(k+1)(1−α)⌉` (1-indexed),
  the `Icc` band, coverage `≥ 1 − α`.
* **Finiteness threshold** (`paperIndex_le_iff`, `paperIndex_gt_of_lt`,
  `trivial_band_covers`): `⌈(k+1)(1−α)⌉ ≤ k ↔ ⌈1/α⌉ − 1 ≤ k` (the band is finite once
  `k ≥ ⌈1/α⌉ − 1`); below the threshold the prescribed order statistic does not exist
  (`k < ⌈(k+1)(1−α)⌉`) and the convention "the band is all of `ℝ`" covers trivially — infinite
  rather than wrong.
* **Conditional validity** (`certified_blind_band_conditional`): the secondary claim.  The
  paper's resolved–blind independence is consumed here as a NAMED HYPOTHESIS — "for a.e.
  deployment datum `y` the conditional law of `(θ⋆, x₁, …, x_k)` given `y` is still
  exchangeable (and score-distinct)" — which is precisely what that independence buys; it is
  **not derived** from an independence structure here.  Given it, coverage holds conditionally
  at a.e. `y`, with data-dependent score parameters `m(y), s(y)`.

## Conventions

Exchangeability is the measure form used by `SelfCheckAndSharpness`: the joint law on
`Fin (k+1) → ℝ` is invariant under precomposition with every coordinate permutation.  The TEST
coordinate is slot `0` (the imported engine's convention); the paper lists the test score last —
under exchangeability the labelling is immaterial (any slot's rank has the same law, which is the
engine's own first lemma).  Order statistics are 0-indexed here (`orderStat g j` is the
`(j+1)`-th smallest of `g`), and the paper's 1-indexed `η₍⌈(k+1)(1−α)⌉₎` is
`orderStat _ ⟨paperIndex k α − 1, _⟩`.

## Honesty — named hypotheses vs discharged

No `sorry`/`admit`/`native_decide`, no new axioms.  Named hypotheses (each a *modelling*
premise, flagged where it appears):

1. `hexch` — exchangeability of the references with the deployment truth: the proposition's own
   stated assumption ("no assumption beyond exchangeability" — this is that assumption).
2. `hdist` — a.s. pairwise distinctness of the `k+1` SCORES (not the coordinates: distinct
   truths can collide in `|θ − m|/s`).  This is the tie-handling device, identical to
   `SelfCheckAndSharpness`'s; it holds e.g. whenever the joint law of the scores has a density.
   With ties the standard conservative (`≤`-rank) treatment would still give `≥ 1−α`; that
   refinement is not formalized here.
3. In `certified_blind_band_conditional` only: the conditional exchangeability/distinctness
   given the deployment datum — the formal residue of the paper's resolved–blind independence.

Everything else is discharged: the rank law (imported theorem, not axiom), the order-statistic /
rank set identity, the interval form, the ceiling algebra of the finiteness threshold, and the
coverage arithmetic.  `#print axioms` for every public result confirms
`[propext, Classical.choice, Quot.sound]` only.
-/

open MeasureTheory ProbabilityTheory Finset
open scoped ENNReal

namespace CertifiedBand

open SelfCheckAndSharpness

variable {k : ℕ}

/-! ## Part 1 — order statistics and the rank identity -/

/-- The `(j+1)`-th smallest of the `k` calibration scores (0-indexed `j`): sort the tuple and
read off position `j`. -/
noncomputable def orderStat (g : Fin k → ℝ) (j : Fin k) : ℝ :=
  (g ∘ Tuple.sort g) j

/-- For a MONOTONE tuple `h`, sitting at or below position `j`'s value is the same as at most
`j` entries lying strictly below `t`.  Tie-free: no distinctness is needed for this identity. -/
theorem le_monotone_iff_count {h : Fin k → ℝ} (hmono : Monotone h) (t : ℝ) (j : Fin k) :
    t ≤ h j ↔ #{p | h p < t} ≤ (j : ℕ) := by
  constructor
  · intro ht
    have hsub : ({p | h p < t} : Finset (Fin k)) ⊆ Finset.Iio j := by
      intro p hp
      rw [Finset.mem_filter] at hp
      rw [Finset.mem_Iio]
      by_contra hle
      exact absurd (hp.2.trans_le (ht.trans (hmono (not_lt.mp hle)))) (lt_irrefl _)
    calc #{p | h p < t} ≤ #(Finset.Iio j) := Finset.card_le_card hsub
      _ = (j : ℕ) := Fin.card_Iio j
  · intro hcard
    by_contra hlt
    rw [not_le] at hlt
    have hsub : Finset.Iic j ⊆ ({p | h p < t} : Finset (Fin k)) := by
      intro p hp
      rw [Finset.mem_Iic] at hp
      rw [Finset.mem_filter]
      exact ⟨Finset.mem_univ _, (hmono hp).trans_lt hlt⟩
    have h1 := Finset.card_le_card hsub
    rw [Fin.card_Iic] at h1
    omega

/-- **The order-statistic / count identity** (tie-free): `t ≤ η₍ⱼ₊₁₎` iff at most `j` of the
calibration scores lie strictly below `t`. -/
theorem le_orderStat_iff (g : Fin k → ℝ) (t : ℝ) (j : Fin k) :
    t ≤ orderStat g j ↔ #{i | g i < t} ≤ (j : ℕ) := by
  have hcount : #{p | (g ∘ Tuple.sort g) p < t} = #{i | g i < t} :=
    Finset.card_equiv (Tuple.sort g) fun p => by simp [Function.comp]
  have h := le_monotone_iff_count (Tuple.monotone_sort g) t j
  rw [hcount] at h
  exact h

/-- The conformal event IS a rank event: the test score (slot `0`) sits at or below the `j`-th
order statistic of the `k` calibration scores iff its rank among all `k+1` scores is at most
`j`.  (`rankAmong` is `SelfCheckAndSharpness`'s position statistic.) -/
theorem le_orderStat_iff_rank (f : Fin (k + 1) → ℝ) (j : Fin k) :
    f 0 ≤ orderStat (fun i => f i.succ) j ↔ rankAmong f 0 ≤ (j : ℕ) := by
  refine (le_orderStat_iff _ _ _).trans ?_
  rw [← sbcRank_eq]
  exact Iff.rfl

/-- The rank sublevel event as a finite disjoint union of the rank level sets. -/
theorem rank_le_eq_biUnion (j : ℕ) :
    {f : Fin (k + 1) → ℝ | rankAmong f 0 ≤ j}
      = ⋃ m ∈ Finset.range (j + 1), {f : Fin (k + 1) → ℝ | rankAmong f 0 = m} := by
  ext f
  simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_range, Nat.lt_succ_iff, exists_prop]
  exact ⟨fun h => ⟨rankAmong f 0, h, rfl⟩, fun ⟨m, hm, he⟩ => le_of_eq_of_le he hm⟩

theorem measurableSet_rank_le (j : ℕ) :
    MeasurableSet {f : Fin (k + 1) → ℝ | rankAmong f 0 ≤ j} := by
  rw [rank_le_eq_biUnion]
  exact (Finset.range (j + 1)).measurableSet_biUnion fun m _ => measurableSet_rankAmong 0 m

theorem measurableSet_le_orderStat (j : Fin k) :
    MeasurableSet {f : Fin (k + 1) → ℝ | f 0 ≤ orderStat (fun i => f i.succ) j} := by
  have h : {f : Fin (k + 1) → ℝ | f 0 ≤ orderStat (fun i => f i.succ) j}
      = {f | rankAmong f 0 ≤ (j : ℕ)} := Set.ext fun f => le_orderStat_iff_rank f j
  rw [h]
  exact measurableSet_rank_le j

/-! ## Part 2 — the rank/quantile coverage law -/

/-- **The rank sublevel law**: under exchangeability + a.s. distinctness the test score's rank
is uniform on `{0,…,k}` (the REUSED engine `rankAmong_zero_uniform`), so
`μ {rank ≤ j} = (j+1)/(k+1)` — exact, not just a bound. -/
theorem measure_rank_le (μ : Measure (Fin (k + 1) → ℝ)) [IsProbabilityMeasure μ]
    (hexch : ∀ σ : Equiv.Perm (Fin (k + 1)), Measure.map (fun f => f ∘ σ) μ = μ)
    (hdist : ∀ᵐ f ∂μ, Function.Injective f) {j : ℕ} (hj : j ≤ k) :
    μ {f | rankAmong f 0 ≤ j} = ((j : ℝ≥0∞) + 1) / ((k : ℝ≥0∞) + 1) := by
  have hd : (↑(Finset.range (j + 1)) : Set ℕ).PairwiseDisjoint
      (fun m => {f : Fin (k + 1) → ℝ | rankAmong f 0 = m}) := by
    intro m _ m' _ hne
    simp only [Function.onFun]
    rw [Set.disjoint_left]
    intro f hf hf'
    rw [Set.mem_setOf_eq] at hf hf'
    exact hne (hf.symm.trans hf')
  rw [rank_le_eq_biUnion,
    measure_biUnion_finset hd fun m _ => measurableSet_rankAmong 0 m]
  have heach : ∀ m ∈ Finset.range (j + 1),
      μ {f : Fin (k + 1) → ℝ | rankAmong f 0 = m} = ((k : ℝ≥0∞) + 1)⁻¹ := by
    intro m hm
    rw [Finset.mem_range, Nat.lt_succ_iff] at hm
    exact rankAmong_zero_uniform μ hexch hdist (hm.trans hj)
  rw [Finset.sum_congr rfl heach, Finset.sum_const, Finset.card_range, nsmul_eq_mul,
    div_eq_mul_inv]
  push_cast
  ring

/-- Cast bookkeeping: `ofReal (n + 1) = n + 1` from `ℝ` to `ℝ≥0∞`. -/
theorem ofReal_natCast_succ (n : ℕ) : ENNReal.ofReal ((n : ℝ) + 1) = (n : ℝ≥0∞) + 1 := by
  rw [show ((n : ℝ) + 1) = ((n + 1 : ℕ) : ℝ) by push_cast; ring, ENNReal.ofReal_natCast]
  push_cast
  ring

/-- **The rank/quantile lemma (the heart of split conformal).**  For `k+1` exchangeable,
a.s. pairwise-distinct real scores (test in slot `0`), any 0-indexed order-statistic position
`j` with `(k+1)(1−α) ≤ j+1` gives
`Pr(test ≤ η₍ⱼ₊₁₎) = (j+1)/(k+1) ≥ 1−α`. -/
theorem score_quantile_coverage (μ : Measure (Fin (k + 1) → ℝ)) [IsProbabilityMeasure μ]
    (hexch : ∀ σ : Equiv.Perm (Fin (k + 1)), Measure.map (fun f => f ∘ σ) μ = μ)
    (hdist : ∀ᵐ f ∂μ, Function.Injective f)
    {α : ℝ} {j : Fin k} (hj : ((k : ℝ) + 1) * (1 - α) ≤ ((j : ℕ) : ℝ) + 1) :
    ENNReal.ofReal (1 - α) ≤ μ {f | f 0 ≤ orderStat (fun i => f i.succ) j} := by
  have hset : {f : Fin (k + 1) → ℝ | f 0 ≤ orderStat (fun i => f i.succ) j}
      = {f | rankAmong f 0 ≤ (j : ℕ)} := Set.ext fun f => le_orderStat_iff_rank f j
  rw [hset, measure_rank_le μ hexch hdist (le_of_lt j.isLt)]
  have hk1 : (0 : ℝ) < (k : ℝ) + 1 := by positivity
  have h1 : (1 - α : ℝ) ≤ (((j : ℕ) : ℝ) + 1) / ((k : ℝ) + 1) := by
    rw [le_div_iff₀ hk1]
    nlinarith [hj]
  have h2 := ENNReal.ofReal_le_ofReal h1
  rwa [ENNReal.ofReal_div_of_pos hk1, ofReal_natCast_succ, ofReal_natCast_succ] at h2

/-! ## Part 3 — "for every prior": pullback along an arbitrary score -/

/-- **Conformal coverage for ANY score of ANY exchangeable vector** — the "for every prior and
every truth" master.  `X` is an arbitrary underlying space (truth objects), `ν` any exchangeable
joint law of `(θ⋆, x₁, …, x_k)` (test in slot `0`), and `η` an ARBITRARY measurable score — in
the paper `η = |⟨·⟩_blind − m_ρ(x_R)|/s_ρ`, built from the prior; nothing about `η`'s values is
used, which is why the guarantee is prior-free.  NAMED HYPOTHESIS `hdist`: the scores are a.s.
pairwise distinct (tie handling; see module docstring). -/
theorem coverage_of_exchangeable {X : Type*} [MeasurableSpace X]
    (ν : Measure (Fin (k + 1) → X)) [IsProbabilityMeasure ν]
    (hexch : ∀ σ : Equiv.Perm (Fin (k + 1)), Measure.map (fun x => x ∘ σ) ν = ν)
    (η : X → ℝ) (hη : Measurable η)
    (hdist : ∀ᵐ x ∂ν, Function.Injective fun i => η (x i))
    {α : ℝ} {j : Fin k} (hj : ((k : ℝ) + 1) * (1 - α) ≤ ((j : ℕ) : ℝ) + 1) :
    ENNReal.ofReal (1 - α) ≤ ν {x | η (x 0) ≤ orderStat (fun i => η (x i.succ)) j} := by
  have hTmeas : Measurable fun (x : Fin (k + 1) → X) (i : Fin (k + 1)) => η (x i) :=
    measurable_pi_lambda _ fun i => hη.comp (measurable_pi_apply i)
  haveI : IsProbabilityMeasure (ν.map fun x i => η (x i)) :=
    Measure.isProbabilityMeasure_map hTmeas.aemeasurable
  have hexch' : ∀ σ : Equiv.Perm (Fin (k + 1)),
      Measure.map (fun f => f ∘ σ) (ν.map fun x i => η (x i)) = ν.map fun x i => η (x i) := by
    intro σ
    have hσR : Measurable fun f : Fin (k + 1) → ℝ => f ∘ σ :=
      measurable_pi_lambda _ fun i => measurable_pi_apply (σ i)
    have hσX : Measurable fun x : Fin (k + 1) → X => x ∘ σ :=
      measurable_pi_lambda _ fun i => measurable_pi_apply (σ i)
    calc Measure.map (fun f => f ∘ σ) (ν.map fun x i => η (x i))
        = ν.map ((fun f : Fin (k + 1) → ℝ => f ∘ σ) ∘ fun x i => η (x i)) :=
          Measure.map_map hσR hTmeas
      _ = (ν.map fun x : Fin (k + 1) → X => x ∘ σ).map (fun x i => η (x i)) :=
          (Measure.map_map hTmeas hσX).symm
      _ = ν.map fun x i => η (x i) := by rw [hexch σ]
  have hdist' : ∀ᵐ g ∂(ν.map fun x i => η (x i)), Function.Injective g :=
    (MeasureTheory.ae_map_iff hTmeas.aemeasurable
      (measurableSet_distinctSet (N := k))).mpr hdist
  have hcov := score_quantile_coverage (ν.map fun x i => η (x i)) hexch' hdist' hj
  rwa [Measure.map_apply hTmeas (measurableSet_le_orderStat j)] at hcov

/-! ## Part 4 — the band is an interval -/

/-- The standardized blind residual's sublevel condition IS an interval condition:
`|θ − m|/s ≤ c ↔ θ ∈ [m − c·s, m + c·s]` (for `s > 0`; when `c < 0` both sides are false —
the `Icc` is empty, never wrong). -/
theorem mem_band_iff {s : ℝ} (hs : 0 < s) (m c θ : ℝ) :
    |θ - m| / s ≤ c ↔ θ ∈ Set.Icc (m - c * s) (m + c * s) := by
  rw [Set.mem_Icc, div_le_iff₀ hs, abs_le]
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨by linarith, by linarith⟩

/-- **The band is an interval**: as a set equality,
`{θ : |θ − m|/s ≤ c} = Icc (m − c·s) (m + c·s)` — the quasi-convexity of `η` in `θ`, made
explicit. -/
theorem band_eq_Icc {s : ℝ} (hs : 0 < s) (m c : ℝ) :
    {θ : ℝ | |θ - m| / s ≤ c} = Set.Icc (m - c * s) (m + c * s) :=
  Set.ext fun θ => mem_band_iff hs m c θ

/-! ## Part 5 — the certified blind band covers -/

/-- **`p:certify`, general quantile position**: for any exchangeable joint law of the blind
coordinates `(θ⋆, θ₁, …, θ_k)` (test slot `0`), any score parameters `m, s` (in the paper:
`m_ρ(x_R), s_ρ` — the prior's; arbitrary here, hence "whatever the prior believes"), and any
0-indexed order-statistic position `j` with `(k+1)(1−α) ≤ j+1`: the interval band
`[m − η₍ⱼ₊₁₎·s, m + η₍ⱼ₊₁₎·s]` built from the calibration scores covers the deployment truth
`θ⋆` with probability at least `1 − α`.  NAMED HYPOTHESES: `hexch` (the proposition's own
exchangeability premise), `hdist` (a.s. distinct scores — tie handling). -/
theorem certified_blind_band (ν : Measure (Fin (k + 1) → ℝ)) [IsProbabilityMeasure ν]
    (hexch : ∀ σ : Equiv.Perm (Fin (k + 1)), Measure.map (fun x => x ∘ σ) ν = ν)
    (m s : ℝ) (hs : 0 < s)
    (hdist : ∀ᵐ x ∂ν, Function.Injective fun i => |x i - m| / s)
    {α : ℝ} {j : Fin k} (hj : ((k : ℝ) + 1) * (1 - α) ≤ ((j : ℕ) : ℝ) + 1) :
    ENNReal.ofReal (1 - α) ≤ ν {x | x 0 ∈ Set.Icc
      (m - orderStat (fun i => |x i.succ - m| / s) j * s)
      (m + orderStat (fun i => |x i.succ - m| / s) j * s)} := by
  have hη : Measurable fun θ : ℝ => |θ - m| / s :=
    (Continuous.div_const ((continuous_id.sub continuous_const).abs) s).measurable
  have h : ENNReal.ofReal (1 - α)
      ≤ ν {x | |x 0 - m| / s ≤ orderStat (fun i => |x i.succ - m| / s) j} :=
    coverage_of_exchangeable ν hexch (fun θ => |θ - m| / s) hη hdist hj
  have hset : {x : Fin (k + 1) → ℝ | |x 0 - m| / s ≤ orderStat (fun i => |x i.succ - m| / s) j}
      = {x | x 0 ∈ Set.Icc (m - orderStat (fun i => |x i.succ - m| / s) j * s)
          (m + orderStat (fun i => |x i.succ - m| / s) j * s)} :=
    Set.ext fun x => mem_band_iff hs m _ (x 0)
  rwa [hset] at h

/-! ## Part 6 — the paper's quantile index and the finiteness threshold -/

/-- The paper's 1-indexed quantile position `⌈(k+1)(1−α)⌉`. -/
noncomputable def paperIndex (k : ℕ) (α : ℝ) : ℕ :=
  ⌈((k : ℝ) + 1) * (1 - α)⌉₊

/-- The paper's index is at least `1` whenever `α < 1` (the 0-indexed position
`paperIndex − 1` is well-formed). -/
theorem one_le_paperIndex (k : ℕ) {α : ℝ} (hα1 : α < 1) : 1 ≤ paperIndex k α :=
  Nat.ceil_pos.mpr (mul_pos (by positivity) (by linarith))

/-- **The finiteness threshold**: the prescribed order statistic exists among the `k`
calibration scores (`⌈(k+1)(1−α)⌉ ≤ k`) precisely when `k ≥ ⌈1/α⌉ − 1`. -/
theorem paperIndex_le_iff (k : ℕ) {α : ℝ} (hα0 : 0 < α) :
    paperIndex k α ≤ k ↔ ⌈(1 / α : ℝ)⌉₊ - 1 ≤ k := by
  unfold paperIndex
  rw [Nat.ceil_le]
  constructor
  · intro h
    have h1 : (1 / α : ℝ) ≤ (k : ℝ) + 1 := by
      rw [div_le_iff₀ hα0]
      nlinarith
    have h2 : ⌈(1 / α : ℝ)⌉₊ ≤ k + 1 := by
      rw [Nat.ceil_le]
      push_cast
      exact h1
    omega
  · intro h
    have h2 : ⌈(1 / α : ℝ)⌉₊ ≤ k + 1 := by omega
    have h1 : (1 / α : ℝ) ≤ (k : ℝ) + 1 := by
      have h3 := Nat.ceil_le.mp h2
      push_cast at h3
      exact h3
    rw [div_le_iff₀ hα0] at h1
    nlinarith

/-- Below the threshold the prescribed order statistic does NOT exist: `k < ⌈1/α⌉ − 1` forces
`⌈(k+1)(1−α)⌉ > k`.  The paper's convention there is the infinite band — see
`trivial_band_covers`: infinite rather than wrong. -/
theorem paperIndex_gt_of_lt {α : ℝ} (hα0 : 0 < α) {k : ℕ}
    (hk : k < ⌈(1 / α : ℝ)⌉₊ - 1) : k < paperIndex k α := by
  by_contra h
  rw [not_lt] at h
  have h2 := (paperIndex_le_iff k hα0).mp h
  omega

/-- The infinite band (`band = univ`, the convention below the finiteness threshold) trivially
covers at any level: `ν(univ) = 1 ≥ 1 − α`. -/
theorem trivial_band_covers {Ω : Type*} [MeasurableSpace Ω] (ν : Measure Ω)
    [IsProbabilityMeasure ν] {α : ℝ} (hα0 : 0 ≤ α) :
    ENNReal.ofReal (1 - α) ≤ ν Set.univ := by
  rw [measure_univ]
  exact ENNReal.ofReal_le_one.mpr (by linarith)

/-- The paper's 0-indexed position fits among the `k` calibration scores once
`k ≥ ⌈1/α⌉ − 1`. -/
theorem paperIndex_sub_one_lt {α : ℝ} (hα0 : 0 < α) (hα1 : α < 1) {k : ℕ}
    (hk : ⌈(1 / α : ℝ)⌉₊ - 1 ≤ k) : paperIndex k α - 1 < k := by
  have h1 := one_le_paperIndex k hα1
  have h2 := (paperIndex_le_iff k hα0).mpr hk
  omega

/-- **`p:certify`, verbatim endpoint.**  References exchangeable with the deployment truth,
scores a.s. distinct, `0 < α < 1`, and `k ≥ ⌈1/α⌉ − 1` (the finiteness threshold): the band
`{θ : |θ − m|/s ≤ η₍⌈(k+1)(1−α)⌉₎}` — here written as the interval it equals
(`band_eq_Icc`) — covers the deployment truth `θ⋆ = x 0` with probability at least `1 − α`,
for EVERY `m, s` (every prior) and EVERY exchangeable truth law. -/
theorem certify (ν : Measure (Fin (k + 1) → ℝ)) [IsProbabilityMeasure ν]
    (hexch : ∀ σ : Equiv.Perm (Fin (k + 1)), Measure.map (fun x => x ∘ σ) ν = ν)
    (m s : ℝ) (hs : 0 < s)
    (hdist : ∀ᵐ x ∂ν, Function.Injective fun i => |x i - m| / s)
    {α : ℝ} (hα0 : 0 < α) (hα1 : α < 1) (hk : ⌈(1 / α : ℝ)⌉₊ - 1 ≤ k) :
    ENNReal.ofReal (1 - α) ≤ ν {x | x 0 ∈ Set.Icc
      (m - orderStat (fun i => |x i.succ - m| / s)
        ⟨paperIndex k α - 1, paperIndex_sub_one_lt hα0 hα1 hk⟩ * s)
      (m + orderStat (fun i => |x i.succ - m| / s)
        ⟨paperIndex k α - 1, paperIndex_sub_one_lt hα0 hα1 hk⟩ * s)} := by
  apply certified_blind_band ν hexch m s hs hdist
  have h1 := one_le_paperIndex k hα1
  have hle : ((k : ℝ) + 1) * (1 - α) ≤ (paperIndex k α : ℝ) := Nat.le_ceil _
  rw [Nat.cast_sub h1, Nat.cast_one]
  linarith

/-! ## Part 7 — conditional validity (the flagged secondary claim) -/

/-- **Conditional validity, as a hypothesis-based corollary.**  The paper's secondary claim:
under resolved–blind independence the guarantee holds conditionally on the deployment datum.
The independence enters ONLY through the NAMED HYPOTHESES `hexch`/`hdist`: for a.e. deployment
datum `y`, the conditional law `κ y` of `(θ⋆, θ₁, …, θ_k)` is still exchangeable with a.s.
distinct scores — which is exactly what resolved–blind independence buys (conditioning on the
resolved data does not break the blind coordinates' exchangeability).  That implication is NOT
derived here; given it, coverage holds at a.e. `y`, with data-dependent score parameters
`m y, s y` (the paper's `m_ρ(x_R), s_ρ`). -/
theorem certified_blind_band_conditional {Y : Type*} [MeasurableSpace Y]
    (π : Measure Y) (κ : Y → Measure (Fin (k + 1) → ℝ))
    (hprob : ∀ᵐ y ∂π, IsProbabilityMeasure (κ y))
    (hexch : ∀ᵐ y ∂π, ∀ σ : Equiv.Perm (Fin (k + 1)),
      Measure.map (fun x => x ∘ σ) (κ y) = κ y)
    (m s : Y → ℝ) (hs : ∀ᵐ y ∂π, 0 < s y)
    (hdist : ∀ᵐ y ∂π, ∀ᵐ x ∂(κ y), Function.Injective fun i => |x i - m y| / s y)
    {α : ℝ} {j : Fin k} (hj : ((k : ℝ) + 1) * (1 - α) ≤ ((j : ℕ) : ℝ) + 1) :
    ∀ᵐ y ∂π, ENNReal.ofReal (1 - α) ≤ κ y {x | x 0 ∈ Set.Icc
      (m y - orderStat (fun i => |x i.succ - m y| / s y) j * s y)
      (m y + orderStat (fun i => |x i.succ - m y| / s y) j * s y)} := by
  filter_upwards [hprob, hexch, hs, hdist] with y h1 h2 h3 h4
  haveI := h1
  exact certified_blind_band (κ y) h2 (m y) (s y) h3 h4 hj

end CertifiedBand

-- #print axioms audit (Part 1 — order statistics and the rank identity)
#print axioms CertifiedBand.le_monotone_iff_count
#print axioms CertifiedBand.le_orderStat_iff
#print axioms CertifiedBand.le_orderStat_iff_rank
#print axioms CertifiedBand.rank_le_eq_biUnion
#print axioms CertifiedBand.measurableSet_rank_le
#print axioms CertifiedBand.measurableSet_le_orderStat

-- #print axioms audit (Part 2 — the rank/quantile coverage law)
#print axioms CertifiedBand.measure_rank_le
#print axioms CertifiedBand.ofReal_natCast_succ
#print axioms CertifiedBand.score_quantile_coverage

-- #print axioms audit (Part 3 — "for every prior")
#print axioms CertifiedBand.coverage_of_exchangeable

-- #print axioms audit (Part 4 — the band is an interval)
#print axioms CertifiedBand.mem_band_iff
#print axioms CertifiedBand.band_eq_Icc

-- #print axioms audit (Part 5 — the certified blind band covers)
#print axioms CertifiedBand.certified_blind_band

-- #print axioms audit (Part 6 — the paper index and the finiteness threshold)
#print axioms CertifiedBand.one_le_paperIndex
#print axioms CertifiedBand.paperIndex_le_iff
#print axioms CertifiedBand.paperIndex_gt_of_lt
#print axioms CertifiedBand.trivial_band_covers
#print axioms CertifiedBand.paperIndex_sub_one_lt
#print axioms CertifiedBand.certify

-- #print axioms audit (Part 7 — conditional validity)
#print axioms CertifiedBand.certified_blind_band_conditional
