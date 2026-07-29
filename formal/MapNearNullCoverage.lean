/-
# `p:mapnearnull` (ii), the coverage tail: the narrowest conditional credible interval
covers with probability at most `2 B⋆ ‖A v‖ E‖d‖ / (α μ_v)`

This module machine-checks the LAST unchecked clause of Proposition `p:mapnearnull` (ii):

> "against a truth whose fiber conditional given the readout has density at most `B⋆`, the
> narrowest conditional `1−α` credible interval covers with probability at most
> `2 B⋆ ‖A v‖ E‖d‖ / (α μ_v)`: the collapse of `p:map`, recovered at rate `‖A v‖` as the
> illumination vanishes."

The paper's proof has four steps, and each is a theorem here:

1. **Conditional Markov** (`markov_good_event`): under any fiber probability law, a nonnegative
   integrable variable `D` satisfies `Pr(D ≤ E[D]/α) ≥ 1 − α`.  Proved from mathlib's `ℝ≥0∞`
   Markov inequality (`meas_ge_le_lintegral_div`) — no measurability of `D` beyond integrability
   is consumed, and the `E[D] = 0` endpoint (`D = 0` a.s.) is handled exactly.
2. **The exhibited interval** (`interval_mass_of_tube`, `radius_interval_credible`): the pinning
   tube `|vᵀx̂ − ψ(Π x̂)| ≤ ‖A v‖ ‖d‖ / μ_v` — the clause of `p:mapnearnull` (i) already
   machine-checked in `MapNearNullTube.pinning_tube` / `MapNearNullSharp.pinning_tube_kinked` —
   plus step 1 place conditional mass at least `1 − α` on
   `ψ(Π x̂) ± ‖A v‖ · E[‖d‖ | Π x̂] / (α μ_v)`, per readout fiber.
3. **The narrowest interval is no wider** (`narrowest_length_le`) and **the density bound**
   (`coverage_le_of_length`, reusing `HonestInterval.coverage_compProd_le` — the wave-4-checked
   inequality "coverage ≤ B⋆ · expected length" — verbatim on the truth's fiber conditional).
4. **The tower property** (`lintegral_fiber_mean`): `E[E[‖d‖ | Π x̂]] = E‖d‖`, as the
   `compProd` Fubini identity, averaging the conditional bound to the stated
   `2 B⋆ ‖A v‖ E‖d‖ / (α μ_v)`.

`narrowest_credible_coverage_le` chains all four; `coverage_zero_of_kernel` is the `A v = 0`
endpoint (coverage exactly `0` — `p:map`'s collapse recovered); the `_condKernel` variant
replaces the abstract archive fiber kernel by mathlib's canonical disintegration of the archive
joint law, so the conditional's existence is a theorem, not an input.

## Formal setting

`𝒲` is the readout space — the remaining coordinates `Π x̂` (everything the conditioning sees).
The archive's joint law of `(readout, vᵀx̂, ‖d‖)` is disintegrated as `μ ⊗ₘ ν` with `μ` the
readout law and `ν : Kernel 𝒲 (ℝ × ℝ)` the fiber law of the pair `(archived blind coordinate,
data-gradient norm)` given the readout; `∫ p, p.2 ∂(ν w)` IS the paper's `E[‖d‖ | Π x̂ = w]`.
The truth's blind coordinate shares the readout and has fiber conditional `τ : Kernel 𝒲 ℝ`;
its coverage of an interval rule is `HonestInterval.coverage (μ ⊗ₘ τ) lo hi`, exactly as in
`c:honest`.

## "Narrowest": the formalization choice (stated per the protocol)

The narrowest conditional `1−α` credible interval is consumed through its DEFINING minimality,
as the hypothesis `hmin`: at a.e. readout `w`, the reported interval `[lo w, hi w]` is no longer
than ANY closed interval carrying conditional archive mass `≥ 1−α` at `w`
(`ν w (Prod.fst ⁻¹' Icc a b)` is that mass — `fiber_mass_eq_map_fst` identifies it with the
fiber posterior marginal).  This is the honest minimal reading of the paper's sentence:
* the general claim "any `1−α` credible interval is no wider than a given one" is FALSE — only
  minimality transfers the exhibited interval's width;
* quantifying over closed intervals loses nothing (an open or half-open credible interval's
  closure has no smaller mass and the same length);
* the rule's own credibility and the EXISTENCE of a minimizer are not needed for the upper
  bound, so neither is assumed — the bound holds for every minimal-length rule if any exists.

## Honesty — named hypotheses (flagged)

No `sorry`/`admit`; `#print axioms` on every public theorem lists only `propext`,
`Classical.choice`, `Quot.sound`.  Following the conventions of `MapNearNullSharp` (which takes
the same tube and the same integrability as hypotheses for the conditional-variance clause),
the model inputs are named hypotheses rather than re-derivations:

* `htube` — the pinning tube `|vᵀx̂ − ψ(Π x̂)| ≤ ‖A v‖ ‖d‖ / μ_v` a.e. under the archive law.
  Pathwise (for every measurement and minimizer) this is the machine-checked
  `MapNearNullTube.pinning_tube` / `MapNearNullSharp.pinning_tube_kinked`; its lift to an a.e.
  statement about a measurable minimizer selection is the modeling step, exactly as in
  `MapNearNullSharp.condVar_tube_paper`.
* `hD` — `E‖d‖ < ∞` (first moment; part (ii)'s variance clause needed the second).  The paper's
  objective-comparison argument bounding it for the Gaussian data term is closed in
  `RidgeMarginal.lean` (`misfit_comparison → expected_grad_sq_lt_top`), which discharges the
  second moment and hence this first-moment hypothesis.
* `hnn` — `‖d‖ ≥ 0` a.e.: automatic for a norm, carried abstractly since the fiber coordinate
  here is any real `D`.
* `hB` — the truth's fiber conditional is dominated by `B⋆ · Lebesgue`: the paper's
  "`B⋆`-bounded conditional density", the same named modeling hypothesis as `c:honest`'s
  `hcond` (`HonestInterval`), of which only the domination is consumed.
* `hmin` — the minimality above: the DEFINITION of "narrowest", not a lemma to discharge.

Everything else — conditional Markov, the exhibited interval's conditional mass, the length
comparison, the density-bound coverage inequality, the tower property, the endpoint — is proved.
-/
import HonestInterval

namespace MapNearNullCoverage

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

/-! ## Step 1: conditional Markov, per readout fiber -/

/-- **Conditional Markov inequality, one fiber.**  Under a probability law `P`, a nonnegative
integrable variable `f` satisfies `Pr(f ≤ E[f]/α) ≥ 1 − α` for every `α > 0`.  Applied fiberwise
to the archive's conditional law given the readout, this is the paper's "the conditional Markov
inequality"; the `E[f] = 0` endpoint (where `f = 0` a.s. and the good event has full mass) is
handled exactly, so no positivity of the conditional mean is assumed. -/
theorem markov_good_event {γ : Type*} [MeasurableSpace γ] (P : Measure γ)
    [IsProbabilityMeasure P] {f : γ → ℝ} {α : ℝ} (hα : 0 < α)
    (hf : Integrable f P) (hnn : ∀ᵐ x ∂P, 0 ≤ f x) :
    ENNReal.ofReal (1 - α) ≤ P {x | f x ≤ (∫ y, f y ∂P) / α} := by
  have hnn' : 0 ≤ᵐ[P] f := hnn
  rcases (integral_nonneg_of_ae hnn').lt_or_eq with hm | hm
  · -- positive mean: honest Markov
    set m : ℝ := ∫ y, f y ∂P with hm_def
    -- the bad event has mass at most `α`
    have hbad : P {x | f x ≤ m / α}ᶜ ≤ ENNReal.ofReal α := by
      have hsub : {x | f x ≤ m / α}ᶜ
          ⊆ {x | ENNReal.ofReal (m / α) ≤ ENNReal.ofReal (f x)} := by
        intro x hx
        simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_le] at hx
        exact ENNReal.ofReal_le_ofReal hx.le
      have hg : AEMeasurable (fun x => ENNReal.ofReal (f x)) P :=
        hf.aemeasurable.ennreal_ofReal
      have hε0 : ENNReal.ofReal (m / α) ≠ 0 :=
        (ENNReal.ofReal_pos.mpr (div_pos hm hα)).ne'
      have hlint : ∫⁻ x, ENNReal.ofReal (f x) ∂P = ENNReal.ofReal m :=
        (ofReal_integral_eq_lintegral_ofReal hf hnn').symm
      have hdivR : m / (m / α) = α := by
        rw [div_div_eq_mul_div, mul_comm, mul_div_assoc, div_self hm.ne', mul_one]
      calc P {x | f x ≤ m / α}ᶜ
          ≤ P {x | ENNReal.ofReal (m / α) ≤ ENNReal.ofReal (f x)} := measure_mono hsub
        _ ≤ (∫⁻ x, ENNReal.ofReal (f x) ∂P) / ENNReal.ofReal (m / α) :=
            meas_ge_le_lintegral_div hg hε0 ENNReal.ofReal_ne_top
        _ = ENNReal.ofReal m / ENNReal.ofReal (m / α) := by rw [hlint]
        _ = ENNReal.ofReal (m / (m / α)) := (ENNReal.ofReal_div_of_pos (div_pos hm hα)).symm
        _ = ENNReal.ofReal α := by rw [hdivR]
    -- subadditivity (no measurability of the event needed) closes the complement
    have huniv : (1 : ℝ≥0∞) ≤ P {x | f x ≤ m / α} + P {x | f x ≤ m / α}ᶜ := by
      calc (1 : ℝ≥0∞) = P Set.univ := measure_univ.symm
        _ = P ({x | f x ≤ m / α} ∪ {x | f x ≤ m / α}ᶜ) := by rw [Set.union_compl_self]
        _ ≤ P {x | f x ≤ m / α} + P {x | f x ≤ m / α}ᶜ := measure_union_le _ _
    have h1 : ENNReal.ofReal (1 - α) = 1 - ENNReal.ofReal α := by
      rw [ENNReal.ofReal_sub 1 hα.le, ENNReal.ofReal_one]
    rw [h1]
    refine tsub_le_iff_right.mpr ?_
    calc (1 : ℝ≥0∞) ≤ P {x | f x ≤ m / α} + P {x | f x ≤ m / α}ᶜ := huniv
      _ ≤ P {x | f x ≤ m / α} + ENNReal.ofReal α := add_le_add le_rfl hbad
  · -- zero mean: `f = 0` a.s., the good event has full mass
    have hf0 : f =ᵐ[P] 0 := (integral_eq_zero_iff_of_nonneg_ae hnn' hf).mp hm.symm
    have hfull : ∀ᵐ x ∂P, x ∈ {x | f x ≤ (∫ y, f y ∂P) / α} := by
      filter_upwards [hf0] with x hx
      have hx0 : f x = 0 := by simpa using hx
      simp only [hx0, ← hm, zero_div, le_refl]
    calc ENNReal.ofReal (1 - α) ≤ 1 := ENNReal.ofReal_le_one.mpr (by linarith)
      _ = P Set.univ := measure_univ.symm
      _ ≤ P {x | f x ≤ (∫ y, f y ∂P) / α} :=
          measure_mono_ae (hfull.mono fun x hx _ => hx)

/-! ## Step 2: the exhibited interval carries conditional mass at least `1 − α` -/

/-- Interpretation glue: the mass the fiber law `P` of `(vᵀx̂, ‖d‖)` puts on
`Prod.fst ⁻¹' Icc a b` IS the fiber posterior marginal mass of the interval `[a, b]` — the
"conditional mass" of the paper's credible-interval sentence. -/
theorem fiber_mass_eq_map_fst (P : Measure (ℝ × ℝ)) (a b : ℝ) :
    P.map Prod.fst (Set.Icc a b) = P (Prod.fst ⁻¹' Set.Icc a b) :=
  Measure.map_apply measurable_fst measurableSet_Icc

/-- **The exhibited interval, one fiber.**  If under the fiber law `P` of the pair
`(archived coordinate, gradient norm)` the pinning tube `|X − ψ| ≤ c · D` holds a.e. (the
machine-checked `p:mapnearnull` (i) with `c = ‖A v‖/μ_v`) and `D ≥ 0` is integrable, then the
interval `ψ ± c · E[D]/α` carries mass at least `1 − α` of the archived coordinate: Markov
confines `D`, the tube converts that into confinement of `X`. -/
theorem interval_mass_of_tube (P : Measure (ℝ × ℝ)) [IsProbabilityMeasure P]
    {α c ψ : ℝ} (hα : 0 < α) (hc : 0 ≤ c)
    (hD : Integrable (fun p : ℝ × ℝ => p.2) P)
    (hnn : ∀ᵐ p ∂P, 0 ≤ p.2)
    (htube : ∀ᵐ p ∂P, |p.1 - ψ| ≤ c * p.2) :
    ENNReal.ofReal (1 - α)
      ≤ P (Prod.fst ⁻¹' Set.Icc (ψ - c * (∫ p, p.2 ∂P) / α)
                                 (ψ + c * (∫ p, p.2 ∂P) / α)) := by
  refine (markov_good_event P hα hD hnn).trans (measure_mono_ae ?_)
  filter_upwards [htube] with q hq hqm
  have hq2 : q.2 ≤ (∫ p, p.2 ∂P) / α := hqm
  show q ∈ Prod.fst ⁻¹' Set.Icc (ψ - c * (∫ p, p.2 ∂P) / α) (ψ + c * (∫ p, p.2 ∂P) / α)
  simp only [Set.mem_preimage, Set.mem_Icc]
  have h1 : c * q.2 ≤ c * ((∫ p, p.2 ∂P) / α) := mul_le_mul_of_nonneg_left hq2 hc
  have h2 : c * ((∫ p, p.2 ∂P) / α) = c * (∫ p, p.2 ∂P) / α := (mul_div_assoc _ _ _).symm
  have h3 := abs_le.mp (hq.trans (h1.trans_eq h2))
  exact ⟨by linarith [h3.1], by linarith [h3.2]⟩

section Archive

variable {𝒲 : Type*} [MeasurableSpace 𝒲]
variable {G : Type*} [NormedAddCommGroup G]

/-- **The paper's exhibited credible interval, at the archive.**  For the archive fiber kernel
`ν` of `(vᵀx̂, ‖d‖)` given the readout, with the pinning tube of `p:mapnearnull` (i) holding
a.e. under the archive law `μ ⊗ₘ ν` and `E‖d‖ < ∞`, at a.e. readout `w` the interval

    `ψ(w) ± ‖A v‖ · E[‖d‖ | w] / (α μ_v)`

carries conditional mass at least `1 − α` of the archived blind coordinate — the paper's
"(i) and the conditional Markov inequality place conditional mass at least `1−α` on
`ψ(Π x̂) ± ‖A v‖ E[‖d‖ | Π x̂]/(α μ_v)`", with `E[‖d‖ | w] = ∫ p, p.2 ∂(ν w)`. -/
theorem radius_interval_credible
    (μ : Measure 𝒲) [SFinite μ] (ν : Kernel 𝒲 (ℝ × ℝ)) [IsMarkovKernel ν]
    (Av : G) {mu α : ℝ} (hmu : 0 < mu) (hα : 0 < α) (ψ : 𝒲 → ℝ)
    (htube : ∀ᵐ q ∂(μ ⊗ₘ ν), |q.2.1 - ψ q.1| ≤ ‖Av‖ * q.2.2 / mu)
    (hnn : ∀ᵐ q ∂(μ ⊗ₘ ν), 0 ≤ q.2.2)
    (hD : Integrable (fun q : 𝒲 × ℝ × ℝ => q.2.2) (μ ⊗ₘ ν)) :
    ∀ᵐ w ∂μ, ENNReal.ofReal (1 - α)
      ≤ ν w (Prod.fst ⁻¹' Set.Icc
          (ψ w - ‖Av‖ * (∫ p, p.2 ∂(ν w)) / (α * mu))
          (ψ w + ‖Av‖ * (∫ p, p.2 ∂(ν w)) / (α * mu))) := by
  have htube' := Measure.ae_ae_of_ae_compProd htube
  have hnn' := Measure.ae_ae_of_ae_compProd hnn
  have hint : ∀ᵐ w ∂μ, Integrable (fun p : ℝ × ℝ => p.2) (ν w) :=
    ((Measure.integrable_compProd_iff hD.aestronglyMeasurable).mp hD).1
  filter_upwards [htube', hnn', hint] with w hw1 hw2 hw3
  have hc : (0 : ℝ) ≤ ‖Av‖ / mu := div_nonneg (norm_nonneg _) hmu.le
  have hw2' : ∀ᵐ p ∂(ν w), 0 ≤ p.2 := hw2
  have htw : ∀ᵐ p ∂(ν w), |p.1 - ψ w| ≤ ‖Av‖ / mu * p.2 := by
    filter_upwards [hw1] with p hp
    have hp' : |p.1 - ψ w| ≤ ‖Av‖ * p.2 / mu := hp
    exact hp'.trans_eq (by ring)
  have key := interval_mass_of_tube (ν w) hα hc hw3 hw2' htw
  have hrw : ‖Av‖ / mu * (∫ p, p.2 ∂(ν w)) / α
      = ‖Av‖ * (∫ p, p.2 ∂(ν w)) / (α * mu) := by
    rw [div_mul_eq_mul_div, div_div, mul_comm mu α]
  rwa [hrw] at key

/-! ## Step 3: the narrowest credible interval is no wider, and the density bound -/

/-- **The narrowest conditional credible interval is no wider than the exhibited one.**
If at a.e. readout the rule `[lo w, hi w]` is of minimal length among closed intervals carrying
conditional archive mass at least `1 − α` (`hmin` — the DEFINING property of the narrowest
conditional `1−α` credible interval; see the module docstring for why this is the honest
reading), then its length is at most `2 ‖A v‖ E[‖d‖ | w] / (α μ_v)` a.e. — the paper's "so the
narrowest `1−α` credible interval is no wider". -/
theorem narrowest_length_le
    (μ : Measure 𝒲) [SFinite μ] (ν : Kernel 𝒲 (ℝ × ℝ)) [IsMarkovKernel ν]
    (Av : G) {mu α : ℝ} (hmu : 0 < mu) (hα : 0 < α) (ψ : 𝒲 → ℝ)
    (htube : ∀ᵐ q ∂(μ ⊗ₘ ν), |q.2.1 - ψ q.1| ≤ ‖Av‖ * q.2.2 / mu)
    (hnn : ∀ᵐ q ∂(μ ⊗ₘ ν), 0 ≤ q.2.2)
    (hD : Integrable (fun q : 𝒲 × ℝ × ℝ => q.2.2) (μ ⊗ₘ ν))
    {lo hi : 𝒲 → ℝ}
    (hmin : ∀ᵐ w ∂μ, ∀ a b : ℝ,
      ENNReal.ofReal (1 - α) ≤ ν w (Prod.fst ⁻¹' Set.Icc a b) → hi w - lo w ≤ b - a) :
    ∀ᵐ w ∂μ, hi w - lo w ≤ 2 * (‖Av‖ * (∫ p, p.2 ∂(ν w)) / (α * mu)) := by
  filter_upwards [radius_interval_credible μ ν Av hmu hα ψ htube hnn hD, hmin] with w hw hmw
  have h := hmw _ _ hw
  linarith [h]

/-- **The density bound on an interval rule of bounded length** — the wave-4-checked
`HonestInterval.coverage_compProd_le` (truth-conditional density at most `B⋆` gives coverage at
most `B⋆ ·` expected length), specialized to a rule a.e. no longer than `L`. -/
theorem coverage_le_of_length
    (μ : Measure 𝒲) [SFinite μ] (τ : Kernel 𝒲 ℝ) [IsSFiniteKernel τ]
    {B : ℝ≥0∞} (hB : ∀ᵐ w ∂μ, ∀ t : Set ℝ, τ w t ≤ B * volume t)
    {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi)
    {L : 𝒲 → ℝ} (hlen : ∀ᵐ w ∂μ, hi w - lo w ≤ L w) :
    HonestInterval.coverage (μ ⊗ₘ τ) lo hi ≤ B * ∫⁻ w, ENNReal.ofReal (L w) ∂μ := by
  refine (HonestInterval.coverage_compProd_le μ τ hB hlo hhi).trans (mul_le_mul' le_rfl ?_)
  calc HonestInterval.expLength μ lo hi
      = ∫⁻ w, ENNReal.ofReal (hi w - lo w) ∂μ := rfl
    _ ≤ ∫⁻ w, ENNReal.ofReal (L w) ∂μ := by
        refine lintegral_mono_ae ?_
        filter_upwards [hlen] with w hw
        exact ENNReal.ofReal_le_ofReal hw

/-! ## Step 4: the tower property -/

/-- **The tower property** `E[E[‖d‖ | Π x̂]] = E‖d‖`, in the disintegrated picture: integrating
the fiber mean of the gradient norm over the readout law recovers its mean under the archive
joint law (`compProd` Fubini, `Measure.lintegral_compProd`). -/
theorem lintegral_fiber_mean
    (μ : Measure 𝒲) [SFinite μ] (ν : Kernel 𝒲 (ℝ × ℝ)) [IsSFiniteKernel ν]
    (hD : Integrable (fun q : 𝒲 × ℝ × ℝ => q.2.2) (μ ⊗ₘ ν))
    (hnn : ∀ᵐ q ∂(μ ⊗ₘ ν), 0 ≤ q.2.2) :
    ∫⁻ w, ENNReal.ofReal (∫ p, p.2 ∂(ν w)) ∂μ
      = ENNReal.ofReal (∫ q, q.2.2 ∂(μ ⊗ₘ ν)) := by
  have hnnE : 0 ≤ᵐ[μ ⊗ₘ ν] fun q : 𝒲 × ℝ × ℝ => q.2.2 := hnn
  rw [ofReal_integral_eq_lintegral_ofReal hD hnnE,
    Measure.lintegral_compProd (measurable_snd.snd.ennreal_ofReal)]
  refine lintegral_congr_ae ?_
  have hint : ∀ᵐ w ∂μ, Integrable (fun p : ℝ × ℝ => p.2) (ν w) :=
    ((Measure.integrable_compProd_iff hD.aestronglyMeasurable).mp hD).1
  have hnn' := Measure.ae_ae_of_ae_compProd hnn
  filter_upwards [hint, hnn'] with w hw1 hw2
  exact ofReal_integral_eq_lintegral_ofReal hw1 hw2

/-! ## The coverage tail of `p:mapnearnull` (ii) -/

/-- **`p:mapnearnull` (ii), the coverage tail.**  Archive fiber kernel `ν` of
`(vᵀx̂, ‖d‖)` given the readout; truth fiber conditional `τ` given the same readout, dominated
by `B⋆ · Lebesgue` (the "`B⋆`-bounded conditional density"); the pinning tube of the
machine-checked `p:mapnearnull` (i) a.e. under the archive law; `E‖d‖ < ∞`; and a narrowest
conditional `1−α` credible interval rule (`hmin`, its defining minimality).  Then the rule's
coverage of the truth is at most

    `B⋆ · 2 ‖A v‖ E‖d‖ / (α μ_v)`,

the paper's `2 B⋆ ‖A v‖ E‖d‖ / (α μ_v)`: conditional Markov confines the fiber, the narrowest
credible interval is no wider than the confining interval, the bounded conditional density
converts length into coverage, and the tower property averages over the readout. -/
theorem narrowest_credible_coverage_le
    (μ : Measure 𝒲) [SFinite μ]
    (ν : Kernel 𝒲 (ℝ × ℝ)) [IsMarkovKernel ν]
    (τ : Kernel 𝒲 ℝ) [IsSFiniteKernel τ]
    {B : ℝ≥0∞} (hB : ∀ᵐ w ∂μ, ∀ t : Set ℝ, τ w t ≤ B * volume t)
    (Av : G) {mu α : ℝ} (hmu : 0 < mu) (hα : 0 < α) (_hα1 : α < 1)
    (ψ : 𝒲 → ℝ)
    (htube : ∀ᵐ q ∂(μ ⊗ₘ ν), |q.2.1 - ψ q.1| ≤ ‖Av‖ * q.2.2 / mu)
    (hnn : ∀ᵐ q ∂(μ ⊗ₘ ν), 0 ≤ q.2.2)
    (hD : Integrable (fun q : 𝒲 × ℝ × ℝ => q.2.2) (μ ⊗ₘ ν))
    {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi)
    (hmin : ∀ᵐ w ∂μ, ∀ a b : ℝ,
      ENNReal.ofReal (1 - α) ≤ ν w (Prod.fst ⁻¹' Set.Icc a b) → hi w - lo w ≤ b - a) :
    HonestInterval.coverage (μ ⊗ₘ τ) lo hi
      ≤ B * ENNReal.ofReal (2 * ‖Av‖ * (∫ q, q.2.2 ∂(μ ⊗ₘ ν)) / (α * mu)) := by
  have hlen := narrowest_length_le μ ν Av hmu hα ψ htube hnn hD hmin
  refine (coverage_le_of_length μ τ hB hlo hhi hlen).trans
    (mul_le_mul' le_rfl (le_of_eq ?_))
  have hcnn : (0 : ℝ) ≤ 2 * ‖Av‖ / (α * mu) := by positivity
  calc ∫⁻ w, ENNReal.ofReal (2 * (‖Av‖ * (∫ p, p.2 ∂(ν w)) / (α * mu))) ∂μ
      = ∫⁻ w, ENNReal.ofReal (2 * ‖Av‖ / (α * mu))
          * ENNReal.ofReal (∫ p, p.2 ∂(ν w)) ∂μ := by
        refine lintegral_congr fun w => ?_
        rw [← ENNReal.ofReal_mul hcnn]
        congr 1
        ring
    _ = ENNReal.ofReal (2 * ‖Av‖ / (α * mu))
          * ∫⁻ w, ENNReal.ofReal (∫ p, p.2 ∂(ν w)) ∂μ :=
        lintegral_const_mul' _ _ ENNReal.ofReal_ne_top
    _ = ENNReal.ofReal (2 * ‖Av‖ / (α * mu))
          * ENNReal.ofReal (∫ q, q.2.2 ∂(μ ⊗ₘ ν)) := by
        rw [lintegral_fiber_mean μ ν hD hnn]
    _ = ENNReal.ofReal (2 * ‖Av‖ * (∫ q, q.2.2 ∂(μ ⊗ₘ ν)) / (α * mu)) := by
        rw [← ENNReal.ofReal_mul hcnn, div_mul_eq_mul_div]

/-- **The kernel endpoint: `p:map`'s collapse recovered.**  On a genuine kernel direction
(`A v = 0`) the coverage bound closes to zero exactly: the narrowest conditional credible
interval covers the truth with probability `0` — the zero-coverage collapse of `p:map`,
reached at rate `‖A v‖` as the illumination vanishes. -/
theorem coverage_zero_of_kernel
    (μ : Measure 𝒲) [SFinite μ]
    (ν : Kernel 𝒲 (ℝ × ℝ)) [IsMarkovKernel ν]
    (τ : Kernel 𝒲 ℝ) [IsSFiniteKernel τ]
    {B : ℝ≥0∞} (hB : ∀ᵐ w ∂μ, ∀ t : Set ℝ, τ w t ≤ B * volume t)
    (Av : G) {mu α : ℝ} (hmu : 0 < mu) (hα : 0 < α) (hα1 : α < 1)
    (ψ : 𝒲 → ℝ)
    (htube : ∀ᵐ q ∂(μ ⊗ₘ ν), |q.2.1 - ψ q.1| ≤ ‖Av‖ * q.2.2 / mu)
    (hnn : ∀ᵐ q ∂(μ ⊗ₘ ν), 0 ≤ q.2.2)
    (hD : Integrable (fun q : 𝒲 × ℝ × ℝ => q.2.2) (μ ⊗ₘ ν))
    {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi)
    (hmin : ∀ᵐ w ∂μ, ∀ a b : ℝ,
      ENNReal.ofReal (1 - α) ≤ ν w (Prod.fst ⁻¹' Set.Icc a b) → hi w - lo w ≤ b - a)
    (hker : Av = 0) :
    HonestInterval.coverage (μ ⊗ₘ τ) lo hi = 0 := by
  have h := narrowest_credible_coverage_le μ ν τ hB Av hmu hα hα1 ψ
    htube hnn hD hlo hhi hmin
  rw [hker] at h
  simp only [norm_zero, mul_zero, zero_mul, zero_div, ENNReal.ofReal_zero] at h
  exact le_zero_iff.mp h

/-- **The coverage tail with mathlib's canonical disintegration.**  Same statement as
`narrowest_credible_coverage_le`, with the archive fiber kernel supplied by mathlib's
`ρ.condKernel` for the archive joint law `ρ` of `(readout, vᵀx̂, ‖d‖)` — its existence and
validity (`ρ.fst ⊗ₘ ρ.condKernel = ρ`) is a THEOREM (`Measure.disintegrate`), not an input, so
the tube, sign, and moment hypotheses read directly against the archive law `ρ`. -/
theorem narrowest_credible_coverage_le_condKernel
    (ρ : Measure (𝒲 × (ℝ × ℝ))) [IsFiniteMeasure ρ]
    (τ : Kernel 𝒲 ℝ) [IsSFiniteKernel τ]
    {B : ℝ≥0∞} (hB : ∀ᵐ w ∂ρ.fst, ∀ t : Set ℝ, τ w t ≤ B * volume t)
    (Av : G) {mu α : ℝ} (hmu : 0 < mu) (hα : 0 < α) (hα1 : α < 1)
    (ψ : 𝒲 → ℝ)
    (htube : ∀ᵐ q ∂ρ, |q.2.1 - ψ q.1| ≤ ‖Av‖ * q.2.2 / mu)
    (hnn : ∀ᵐ q ∂ρ, 0 ≤ q.2.2)
    (hD : Integrable (fun q : 𝒲 × ℝ × ℝ => q.2.2) ρ)
    {lo hi : 𝒲 → ℝ} (hlo : Measurable lo) (hhi : Measurable hi)
    (hmin : ∀ᵐ w ∂ρ.fst, ∀ a b : ℝ,
      ENNReal.ofReal (1 - α) ≤ ρ.condKernel w (Prod.fst ⁻¹' Set.Icc a b)
        → hi w - lo w ≤ b - a) :
    HonestInterval.coverage (ρ.fst ⊗ₘ τ) lo hi
      ≤ B * ENNReal.ofReal (2 * ‖Av‖ * (∫ q, q.2.2 ∂ρ) / (α * mu)) := by
  have hd : ρ.fst ⊗ₘ ρ.condKernel = ρ := ρ.disintegrate ρ.condKernel
  have htube' : ∀ᵐ q ∂(ρ.fst ⊗ₘ ρ.condKernel),
      |q.2.1 - ψ q.1| ≤ ‖Av‖ * q.2.2 / mu := by rw [hd]; exact htube
  have hnn' : ∀ᵐ q ∂(ρ.fst ⊗ₘ ρ.condKernel), 0 ≤ q.2.2 := by rw [hd]; exact hnn
  have hD' : Integrable (fun q : 𝒲 × ℝ × ℝ => q.2.2) (ρ.fst ⊗ₘ ρ.condKernel) := by
    rw [hd]; exact hD
  have h := narrowest_credible_coverage_le ρ.fst ρ.condKernel τ hB Av hmu hα hα1 ψ
    htube' hnn' hD' hlo hhi hmin
  rwa [hd] at h

end Archive

/-!
## Reconciliation — what the formalization consumes vs. what the paper states

**(a)** proved here; **(b)** load-bearing named hypothesis; **(c)** automatic/bookkeeping.

* "(i) and the conditional Markov inequality place conditional mass at least `1−α` on
  `ψ(Π x̂) ± ‖A v‖ E[‖d‖ | Π x̂]/(α μ_v)`" — **(a)**: `markov_good_event` +
  `interval_mass_of_tube` + `radius_interval_credible`; (i) itself enters as `htube` **(b)**,
  in exactly the display `MapNearNullTube.pinning_tube` / `MapNearNullSharp.pinning_tube_kinked`
  machine-check pathwise.
* "so the narrowest `1−α` credible interval is no wider" — **(a)**: `narrowest_length_le`,
  with "narrowest" consumed through its defining per-fiber minimality `hmin` **(b/c)** (see
  the module docstring for the formalization choice).
* "a `B⋆`-bounded conditional density gives it conditional probability at most
  `2 B⋆ ‖A v‖ E[‖d‖ | Π x̂]/(α μ_v)`" — **(a)**: `coverage_le_of_length`, REUSING the
  wave-4-checked `HonestInterval.coverage_compProd_le`; the density bound on the truth's fiber
  conditional is `hB` **(b)**, the same modeling hypothesis as `c:honest`.
* "and the tower property averages this to the stated bound" — **(a)**:
  `lintegral_fiber_mean` (`compProd` Fubini) inside `narrowest_credible_coverage_le`.
* "the collapse of `p:map`, recovered at rate `‖A v‖` as the illumination vanishes" — **(a)**:
  the bound is linear in `‖A v‖`, and `coverage_zero_of_kernel` is the exact `A v = 0` endpoint.
* `E‖d‖ < ∞` (`hD`) — **(b)**: the paper's objective-comparison boundedness argument is not
  formalized, matching the caveat `MapNearNullSharp` records for `E‖d‖²`.
-/

end MapNearNullCoverage

-- #print axioms audit (p:mapnearnull (ii), the coverage tail)
#print axioms MapNearNullCoverage.markov_good_event
#print axioms MapNearNullCoverage.fiber_mass_eq_map_fst
#print axioms MapNearNullCoverage.interval_mass_of_tube
#print axioms MapNearNullCoverage.radius_interval_credible
#print axioms MapNearNullCoverage.narrowest_length_le
#print axioms MapNearNullCoverage.coverage_le_of_length
#print axioms MapNearNullCoverage.lintegral_fiber_mean
#print axioms MapNearNullCoverage.narrowest_credible_coverage_le
#print axioms MapNearNullCoverage.coverage_zero_of_kernel
#print axioms MapNearNullCoverage.narrowest_credible_coverage_le_condKernel
