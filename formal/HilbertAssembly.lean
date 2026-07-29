import BlindFreeze
import Mathlib.Probability.Kernel.Disintegration.StandardBorel
import Mathlib.Probability.Kernel.Disintegration.Unique
import Mathlib.MeasureTheory.Constructions.Polish.Basic
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Topology.MetricSpace.Polish
import Mathlib.Probability.Kernel.Basic

/-!
# `r:hilbert` assembled, and `r:ensemble`'s K-mixture (eq:pooled)

This file discharges two "the disintegration is an input rather than a theorem" gaps of
the paper's formal appendix, using mathlib's canonical disintegration
(`MeasureTheory.Measure.condKernel`, existence `Measure.disintegrate`, a.e. uniqueness
`ProbabilityTheory.eq_condKernel_of_measure_eq_compProd`) instead of hypothesizing the
disintegration `reg = νR ⊗ₘ κreg` as `BlindFreeze.curatedPrior_blind_freeze` did.

## Part 1 — `r:hilbert` on a standard Borel split (the core)

For `ρ` a finite measure on `R × B` (`B` standard Borel, `R` any measurable space) and a
**fiber-constant** reweight `g ∘ fst`:

* `resolved_disintegration_exists` — the disintegration along the resolved coordinate EXISTS
  (mathlib's `condKernel`); nothing is assumed.
* `withDensity_fiberConst_eq_compProd_condKernel` — the reweighted measure disintegrates with
  ρ's OWN canonical fiber kernel: `ρ.withDensity (g ∘ fst) = (…).fst ⊗ₘ ρ.condKernel`.
* `condKernel_withDensity_fiberConst` — hence its canonical conditional kernel agrees with `ρ`'s
  a.e.: the freeze, at the level of mathlib's canonical disintegration.
* `curatedPrior_condKernel_freeze` — assembled with Theorem 1 Part A and the derived
  fiber-constancy of the EM reweight (`BlindFreeze.emReweight_factors`): for ANY probability
  regularizer `ρ` on `R × B` (no disintegration hypothesized), the curated prior
  `T[ρ]` disintegrates with `ρ.condKernel`, and `T[ρ].condKernel = ρ.condKernel` a.e.

## Part 2 — `r:hilbert` on the Hilbert carrier

`X` a separable Hilbert space (complete, second countable, Borel), `A : X →L[ℝ] Y` a bounded
operator, blind subspace `ker A` (closed), resolved subspace `(ker A)ᗮ`:

* `resolvedBlindSplit A : ((ker A)ᗮ × ker A) ≃ᵐ X` — the measurable (indeed continuous-linear)
  resolved ⊕ blind splitting, built from `Submodule.isTopCompl_orthogonal`.
* `hilbert_freeze_assembled` — the paper's `r:hilbert`(i) in full: for a likelihood kernel that
  factors through the operator (`A x = A x' → κ x = κ x'`), there EXISTS a Markov fiber kernel
  `κc` such that BOTH the regularizer and the curated prior `T[reg]`, transported through the
  split, disintegrate over the resolved coordinate with the SAME `κc` — and any other
  disintegration of the curated prior agrees with `κc` a.e. (the a.e. uniqueness clause).
  Existence comes from `Measure.condKernel` (X Polish ⇒ the blind subspace is standard Borel),
  NOT from a hypothesis.

## Part 3 — `r:ensemble`'s K-mixture (eq:pooled)

`ι` a finite index of legacy methods; method `i` contributes the graph measure
`(μ.withDensity (p i)) ⊗ₘ deterministic (b i)` — readout density `p i` against a common
reference `μ` on the resolved space, conditional-mode atom `b i (x_R)` on the blind space:

* `pooledFiberKernel` — the K-atom kernel `x_R ↦ ∑ i, (p i x_R / ∑ j, p j x_R) • δ_{b i x_R}`
  with the readout-tilted weights of eq:pooled (0/0 = 0 convention where all densities vanish).
* `pooledArchive_eq_compProd` / `pooledArchive_fst` / `pooledArchive_disintegration` — the pooled
  archive `∑ i, graph i` disintegrates over the readout with exactly this kernel.
* `pooledUniform_condKernel` — the paper's uniform mixture `(1/K) ∑ i, graph i`: its canonical
  conditional kernel IS the K-atom law of eq:pooled, a.e. (uniqueness via `condKernel` again);
  the `1/K` cancels into the readout-tilted weights, exactly as the paper says.
* `pooledFiberKernel_apply_univ` — the weights sum to 1 wherever the pooled readout density is
  positive (and finite): the K-atom law is a probability there.
* `pooledUniform_fiber_atomic` — zero conditional width per atom, read off: a.e. readout, the
  pooled fiber conditional puts NO mass outside the K reconstruction atoms `{b i (x_R)}`.

## Honesty

No `sorry`/`admit`; `#print axioms` at the end lists only `propext`, `Classical.choice`,
`Quot.sound`.  All regularity is explicit:
finite/probability measures where mathlib's `condKernel` needs them; `∀ i r, p i r ≠ ∞`
(densities finite — automatic for actual densities of finite measures, taken pointwise here so
the kernel identity holds everywhere, not just a.e.); `Nonempty ι` (at least one method) for the
uniform mixture's finiteness.  The Hilbert statement is on a genuine separable Hilbert carrier
(complete + second countable + Borel σ-algebra), with the likelihood factoring hypothesis
`A x = A x' → κ x = κ x'` — the paper's `p(y∣x) = ℓ(y, A x)` — and full-support/domination
hypotheses `h_dom`, `h_ac` exactly as in Theorem 1 Part A.
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace HilbertAssembly

/-! ## Part 1 — the freeze against mathlib's canonical disintegration (standard Borel core) -/

section StandardBorelCore

variable {R B : Type*} [MeasurableSpace R] [MeasurableSpace B]
  [StandardBorelSpace B] [Nonempty B]

/-- **The disintegration along the resolved coordinate exists** (`r:hilbert`'s input, now a
theorem): any finite measure on `R × B` with `B` standard Borel is the composition of its
resolved marginal with its canonical fiber kernel.  This is mathlib's `Measure.condKernel`;
nothing is hypothesized. -/
theorem resolved_disintegration_exists (ρ : Measure (R × B)) [IsFiniteMeasure ρ] :
    ρ = ρ.fst ⊗ₘ ρ.condKernel :=
  (ρ.disintegrate ρ.condKernel).symm

/-- **A fiber-constant reweight keeps the canonical fiber kernel.** Reweighting `ρ` by a density
`g ∘ fst` of the resolved coordinate alone yields a measure that disintegrates over the resolved
coordinate with `ρ`'s OWN canonical conditional kernel. -/
theorem withDensity_fiberConst_eq_compProd_condKernel
    (ρ : Measure (R × B)) [IsFiniteMeasure ρ] {g : R → ℝ≥0∞} (hg : Measurable g) :
    ρ.withDensity (fun p => g p.1)
      = (ρ.withDensity fun p => g p.1).fst ⊗ₘ ρ.condKernel := by
  have hswap : ρ.withDensity (fun p => g p.1) = (ρ.fst.withDensity g) ⊗ₘ ρ.condKernel :=
    calc ρ.withDensity (fun p => g p.1)
        = (ρ.fst ⊗ₘ ρ.condKernel).withDensity (fun p => g p.1) := by
          rw [ρ.disintegrate ρ.condKernel]
      _ = (ρ.fst.withDensity g) ⊗ₘ ρ.condKernel :=
          BlindFreeze.freeze_general ρ.fst ρ.condKernel hg
  have hfst : (ρ.withDensity fun p => g p.1).fst = ρ.fst.withDensity g := by
    rw [hswap, Measure.fst_compProd]
  rw [hfst]
  exact hswap

/-- **The freeze at the level of canonical disintegrations.** If the fiber-constant reweight of a
finite measure is still finite, its canonical conditional kernel agrees with `ρ`'s a.e. — the
a.e. uniqueness of disintegrations (`eq_condKernel_of_measure_eq_compProd`) applied to the
explicit disintegration above. -/
theorem condKernel_withDensity_fiberConst
    (ρ : Measure (R × B)) [IsFiniteMeasure ρ] {g : R → ℝ≥0∞} (hg : Measurable g)
    [IsFiniteMeasure (ρ.withDensity fun p => g p.1)] :
    ∀ᵐ r ∂(ρ.withDensity fun p => g p.1).fst,
      (ρ.withDensity fun p => g p.1).condKernel r = ρ.condKernel r := by
  filter_upwards [eq_condKernel_of_measure_eq_compProd ρ.condKernel
    (withDensity_fiberConst_eq_compProd_condKernel ρ hg)] with r hr
  exact hr.symm

/-- **`r:hilbert` assembled on a standard Borel split — no disintegration hypothesized.**

For ANY probability regularizer `ρ` on `R × B` and a likelihood kernel reaching `x` only through
the resolved coordinate, the curated prior `T[ρ]` (Theorem 1 Part A: the data-averaged posterior)
(a) disintegrates over the resolved coordinate with `ρ`'s canonical fiber kernel, and
(b) its OWN canonical fiber kernel agrees with `ρ`'s a.e.:
`T[ρ](· ∣ x_R) = ρ(· ∣ x_R)` — the freeze, with the disintegration's existence supplied by
mathlib's `condKernel` and its a.e. uniqueness by `eq_condKernel_of_measure_eq_compProd`.
This discharges the "existence of the disintegration is an input rather than a theorem" caveat of
`BlindFreeze.curatedPrior_blind_freeze` (which took `ρ = νR ⊗ₘ κreg` as a hypothesis). -/
theorem curatedPrior_condKernel_freeze
    {𝓧 : Type*} [MeasurableSpace 𝓧] [Nonempty R] [StandardBorelSpace R]
    [MeasurableSpace.CountableOrCountablyGenerated (R × B) 𝓧]
    (κ : Kernel (R × B) 𝓧) [IsMarkovKernel κ]
    (ρ ν : Measure (R × B)) [IsProbabilityMeasure ρ] [IsProbabilityMeasure ν]
    (hκ : ∀ x x' : R × B, x.1 = x'.1 → κ x = κ x')
    (h_dom : ∀ᵐ x ∂ρ, κ x ≪ κ ∘ₘ ρ)
    (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ ρ) :
    PriorLaundering.curatedPrior κ ρ ν
        = (PriorLaundering.curatedPrior κ ρ ν).fst ⊗ₘ ρ.condKernel ∧
      ∀ᵐ r ∂(PriorLaundering.curatedPrior κ ρ ν).fst,
        (PriorLaundering.curatedPrior κ ρ ν).condKernel r = ρ.condKernel r := by
  obtain ⟨f, hf, hfeq⟩ := BlindFreeze.emReweight_factors κ ρ ν hκ h_ac
  have hq : PriorLaundering.curatedPrior κ ρ ν = ρ.withDensity fun p => f p.1 := by
    rw [PriorLaundering.pLaw_partA κ ρ ν h_dom h_ac, hfeq]
  have hcomp : PriorLaundering.curatedPrior κ ρ ν
      = (PriorLaundering.curatedPrior κ ρ ν).fst ⊗ₘ ρ.condKernel := by
    rw [hq]
    exact withDensity_fiberConst_eq_compProd_condKernel ρ hf
  refine ⟨hcomp, ?_⟩
  filter_upwards [eq_condKernel_of_measure_eq_compProd ρ.condKernel hcomp] with r hr
  exact hr.symm

end StandardBorelCore

/-! ## Part 2 — the Hilbert carrier -/

section EmReweightGeneral

/- The two `emReweight` facts of `BlindFreeze`, restated for an arbitrary measurable carrier
(they never used the product structure): measurability, and constancy on likelihood fibers. -/

variable {Ω 𝓧 : Type*} [MeasurableSpace Ω] [MeasurableSpace 𝓧]
  [MeasurableSpace.CountableOrCountablyGenerated Ω 𝓧]
  (κ : Kernel Ω 𝓧) [IsMarkovKernel κ]
  (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]

omit [IsProbabilityMeasure μ] in
/-- Measurability of the EM reweight on an arbitrary measurable carrier. -/
lemma measurable_emReweight_general : Measurable (PriorLaundering.emReweight κ μ ν) := by
  unfold PriorLaundering.emReweight
  exact (Kernel.measurable_rnDeriv κ (Kernel.const _ (κ ∘ₘ μ))).lintegral_kernel_prod_right'
    (κ := Kernel.const _ (κ ∘ₘ ν))

omit [IsProbabilityMeasure ν] in
/-- The EM reweight is constant on likelihood fibers: `κ x = κ x'` gives
`g(x) = g(x')` (the RN-derivative depends on `x` only through `κ x`). -/
lemma emReweight_congr (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ) {x x' : Ω} (hxx : κ x = κ x') :
    PriorLaundering.emReweight κ μ ν x = PriorLaundering.emReweight κ μ ν x' := by
  unfold PriorLaundering.emReweight
  refine lintegral_congr_ae (h_ac.ae_le ?_)
  have hx := Kernel.rnDeriv_eq_rnDeriv_measure (κ := κ) (η := Kernel.const Ω (κ ∘ₘ μ)) (a := x)
  have hx' := Kernel.rnDeriv_eq_rnDeriv_measure (κ := κ) (η := Kernel.const Ω (κ ∘ₘ μ)) (a := x')
  rw [Kernel.const_apply] at hx hx'
  filter_upwards [hx, hx'] with y hy hy'
  rw [hy, hy', hxx]

end EmReweightGeneral

section MapWithDensity

/-- Transport of a `withDensity` along a measurable equivalence: pushing forward
`μ · (w ∘ φ)` by `φ` is `(φ_* μ) · w`. -/
lemma map_withDensity_measurableEquiv {Z W : Type*} [MeasurableSpace Z] [MeasurableSpace W]
    (μ : Measure Z) (φ : Z ≃ᵐ W) {w : W → ℝ≥0∞} (hw : Measurable w) :
    (μ.withDensity fun z => w (φ z)).map φ = (μ.map φ).withDensity w := by
  ext s hs
  rw [Measure.map_apply φ.measurable hs, withDensity_apply _ (φ.measurable hs),
    withDensity_apply _ hs, setLIntegral_map hs hw φ.measurable]

end MapWithDensity

section HilbertCarrier

variable {X Y : Type*}
  [NormedAddCommGroup X] [InnerProductSpace ℝ X] [CompleteSpace X] [SecondCountableTopology X]
  [MeasurableSpace X] [BorelSpace X]
  [NormedAddCommGroup Y] [NormedSpace ℝ Y]

/-- The resolved ⊕ blind splitting of the Hilbert carrier, as a continuous linear equivalence
`((ker A)ᗮ × ker A) ≃L[ℝ] X`, `(r, b) ↦ r + b`.  The kernel is closed (`A` bounded), hence
complete (`ContinuousLinearMap.completeSpace_ker`), hence admits an orthogonal projection;
`Submodule.isTopCompl_orthogonal` provides the topological complementarity. -/
noncomputable def resolvedBlindSplitL (A : X →L[ℝ] Y) :
    ((A.ker)ᗮ × A.ker) ≃L[ℝ] X :=
  Submodule.prodEquivOfIsTopCompl _ _
    (Submodule.isTopCompl_orthogonal A.ker).symm

/-- The resolved ⊕ blind splitting as a **measurable** equivalence (Borel σ-algebras on both
sides; the split is a homeomorphism). -/
noncomputable def resolvedBlindSplit (A : X →L[ℝ] Y) :
    ((A.ker)ᗮ × A.ker) ≃ᵐ X :=
  (resolvedBlindSplitL A).toHomeomorph.toMeasurableEquiv

@[simp] lemma resolvedBlindSplit_apply (A : X →L[ℝ] Y)
    (p : (A.ker)ᗮ × A.ker) :
    resolvedBlindSplit A p = (p.1 : X) + (p.2 : X) :=
  rfl

/-- The operator does not see the blind component of the split: `A (r + b) = A r`. -/
lemma apply_resolvedBlindSplit (A : X →L[ℝ] Y)
    (p : (A.ker)ᗮ × A.ker) :
    A (resolvedBlindSplit A p) = A (p.1 : X) := by
  rw [resolvedBlindSplit_apply, map_add, A.apply_val_ker p.2, add_zero]

/-- The second (blind) coordinate of the split is `x` minus its resolved component: the
splitting is the `resolved ⊕ ker A` decomposition of the paper. -/
lemma resolvedBlindSplit_symm_blind (A : X →L[ℝ] Y) (x : X) :
    x - (((resolvedBlindSplit A).symm x).1 : X) ∈ A.ker := by
  have hx : resolvedBlindSplit A ((resolvedBlindSplit A).symm x) = x :=
    (resolvedBlindSplit A).apply_symm_apply x
  rw [resolvedBlindSplit_apply] at hx
  have hb : ((((resolvedBlindSplit A).symm x).2 : X))
      = x - (((resolvedBlindSplit A).symm x).1 : X) := eq_sub_of_add_eq' hx
  rw [← hb]
  exact (((resolvedBlindSplit A).symm x).2).2

set_option maxHeartbeats 1000000 in
/-- **`r:hilbert`(i), assembled on the Hilbert carrier.**

`X` a separable Hilbert space, `A : X →L[ℝ] Y` bounded, likelihood kernel `κ` factoring through
the operator (`A x = A x' → κ x = κ x'` — the paper's `p(y∣x) = ℓ(y, A x)`), `reg` and `truth`
Borel probability measures, with the domination hypotheses of Theorem 1 Part A.  Then there is a
Markov kernel `κc` on the resolved ⊕ blind split such that

* `reg` (transported through the split) disintegrates over the resolved coordinate as
  `reg = reg_R ⊗ₘ κc` — the existence the paper takes from Polishness, here from
  `Measure.condKernel` (the blind subspace `ker A` is closed in Polish `X`, hence standard
  Borel);
* the curated prior `T[reg]` disintegrates with the **same** fiber kernel `κc` — the freeze
  `q(dx_B ∣ x_R) = reg(dx_B ∣ x_R)`, verbatim;
* any finite kernel disintegrating `T[reg]` agrees with `κc` a.e. — the paper's "almost-
  everywhere uniqueness of disintegrations on a Polish space", from
  `eq_condKernel_of_measure_eq_compProd`. -/
theorem hilbert_freeze_assembled
    {𝓧 : Type*} [MeasurableSpace 𝓧] [MeasurableSpace.CountableOrCountablyGenerated X 𝓧]
    (A : X →L[ℝ] Y) (κ : Kernel X 𝓧) [IsMarkovKernel κ]
    (reg ν : Measure X) [IsProbabilityMeasure reg] [IsProbabilityMeasure ν]
    (hκ : ∀ x x' : X, A x = A x' → κ x = κ x')
    (h_dom : ∀ᵐ x ∂reg, κ x ≪ κ ∘ₘ reg)
    (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ reg) :
    ∃ κc : Kernel (↥(A.ker)ᗮ) (↥A.ker), IsMarkovKernel κc ∧
      reg.map (resolvedBlindSplit A).symm
        = (reg.map (resolvedBlindSplit A).symm).fst ⊗ₘ κc ∧
      (PriorLaundering.curatedPrior κ reg ν).map (resolvedBlindSplit A).symm
        = ((PriorLaundering.curatedPrior κ reg ν).map (resolvedBlindSplit A).symm).fst ⊗ₘ κc ∧
      ∀ κ' : Kernel (↥(A.ker)ᗮ) (↥A.ker), IsFiniteKernel κ' →
        (PriorLaundering.curatedPrior κ reg ν).map (resolvedBlindSplit A).symm
          = ((PriorLaundering.curatedPrior κ reg ν).map (resolvedBlindSplit A).symm).fst ⊗ₘ κ' →
        κ' =ᵐ[((PriorLaundering.curatedPrior κ reg ν).map (resolvedBlindSplit A).symm).fst] κc := by
  -- the blind subspace is a closed subspace of a Polish space: complete, Polish, standard Borel
  haveI : PolishSpace (↥A.ker) := A.isClosed_ker.polishSpace
  haveI : Nonempty (↥A.ker) := ⟨0⟩
  set e : ((A.ker)ᗮ × A.ker) ≃ᵐ X := resolvedBlindSplit A with he
  set q : Measure X := PriorLaundering.curatedPrior κ reg ν with hqdef
  haveI : IsProbabilityMeasure (reg.map e.symm) :=
    Measure.isProbabilityMeasure_map e.symm.measurable.aemeasurable
  haveI : IsProbabilityMeasure (q.map e.symm) :=
    Measure.isProbabilityMeasure_map e.symm.measurable.aemeasurable
  -- the reweight factors through the resolved coordinate of the split
  have hgmeas : Measurable (PriorLaundering.emReweight κ reg ν) :=
    measurable_emReweight_general κ reg ν
  have hfmeas : Measurable fun r : ↥(A.ker)ᗮ =>
      PriorLaundering.emReweight κ reg ν (e (r, 0)) :=
    hgmeas.comp (e.measurable.comp (measurable_id.prodMk measurable_const))
  have hfactor : ∀ x : X, PriorLaundering.emReweight κ reg ν x
      = PriorLaundering.emReweight κ reg ν (e (((e.symm x).1), 0)) := by
    intro x
    refine emReweight_congr κ reg ν h_ac (hκ _ _ ?_)
    have h1 : A x = A (((e.symm x).1 : X)) := by
      conv_lhs => rw [← e.apply_symm_apply x]
      exact apply_resolvedBlindSplit A (e.symm x)
    have h2 : A (e (((e.symm x).1), 0)) = A (((e.symm x).1 : X)) :=
      apply_resolvedBlindSplit A ((e.symm x).1, 0)
    rw [h1, h2]
  -- the curated prior, transported, is a fiber-constant reweight of the transported reg
  have hwmeas : Measurable fun p : ↥(A.ker)ᗮ × ↥A.ker =>
      PriorLaundering.emReweight κ reg ν (e (p.1, 0)) :=
    hfmeas.comp measurable_fst
  have hmap : q.map e.symm = (reg.map e.symm).withDensity
      (fun p => PriorLaundering.emReweight κ reg ν (e (p.1, 0))) := by
    have hq' : q = reg.withDensity fun x =>
        (fun p : ↥(A.ker)ᗮ × ↥A.ker =>
          PriorLaundering.emReweight κ reg ν (e (p.1, 0))) (e.symm x) := by
      rw [hqdef, PriorLaundering.pLaw_partA κ reg ν h_dom h_ac]
      congr 1
      funext x
      exact hfactor x
    rw [hq']
    -- hoist the application: solving the implicit density from `hwmeas` (first-order) rather
    -- than against the goal (higher-order) keeps the unifier linear
    have happ := map_withDensity_measurableEquiv reg e.symm hwmeas
    exact happ
  have hcomp : q.map e.symm = (q.map e.symm).fst ⊗ₘ (reg.map e.symm).condKernel := by
    rw [hmap]
    have happ := withDensity_fiberConst_eq_compProd_condKernel (reg.map e.symm) hfmeas
    exact happ
  -- the transported regularizer's canonical fiber kernel is the common witness
  refine ⟨(reg.map e.symm).condKernel, inferInstance,
    ((reg.map e.symm).disintegrate _).symm, hcomp, ?_⟩
  intro κ' hκ' hdis
  haveI := hκ'
  filter_upwards [eq_condKernel_of_measure_eq_compProd κ' hdis,
    eq_condKernel_of_measure_eq_compProd (reg.map e.symm).condKernel hcomp]
    with r hr1 hr2
  exact hr1.trans hr2.symm

end HilbertCarrier

/-! ## Part 3 — `r:ensemble`'s K-mixture (eq:pooled) -/

section PooledEnsemble

variable {R B ι : Type*} [MeasurableSpace R] [MeasurableSpace B] [Fintype ι]

/-- **The K-atom fiber kernel of eq:pooled**: at readout `r`, the mixture of the methods'
conditional-mode Diracs `δ_{b i r}` with the readout-tilted weights
`w i r = p i r / ∑ j, p j r` (ENNReal convention `0/0 = 0` where every density vanishes). -/
noncomputable def pooledFiberKernel (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i))
    (b : ι → R → B) (hb : ∀ i, Measurable (b i)) : Kernel R B where
  toFun r := ∑ i, (p i r / ∑ j, p j r) • Measure.dirac (b i r)
  measurable' := by
    refine Measure.measurable_of_measurable_coe _ fun t ht => ?_
    simp only [Measure.finsetSum_apply, Measure.smul_apply, smul_eq_mul,
      Measure.dirac_apply' _ ht]
    refine Finset.measurable_sum _ fun i _ => Measurable.mul ?_
      ((measurable_one.indicator ht).comp (hb i))
    simp only [div_eq_mul_inv]
    exact (hp i).mul (Finset.measurable_sum _ fun j _ => hp j).inv

@[simp] lemma pooledFiberKernel_apply (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i))
    (b : ι → R → B) (hb : ∀ i, Measurable (b i)) (r : R) :
    pooledFiberKernel p hp b hb r = ∑ i, (p i r / ∑ j, p j r) • Measure.dirac (b i r) :=
  rfl

/-- The K-atom kernel is a finite kernel (each weight is at most 1). -/
instance (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i))
    (b : ι → R → B) (hb : ∀ i, Measurable (b i)) :
    IsFiniteKernel (pooledFiberKernel p hp b hb) := by
  refine ⟨⟨Fintype.card ι, ENNReal.natCast_lt_top _, fun r => ?_⟩⟩
  rw [pooledFiberKernel_apply, Measure.finsetSum_apply]
  simp only [Measure.smul_apply, smul_eq_mul, measure_univ, mul_one]
  calc ∑ i, p i r / ∑ j, p j r
      ≤ ∑ _i : ι, (1 : ℝ≥0∞) :=
        Finset.sum_le_sum fun i _ =>
          (ENNReal.div_le_div_right
            (Finset.single_le_sum (f := fun j => p j r) (fun j _ => zero_le)
              (Finset.mem_univ i)) _).trans ENNReal.div_self_le_one
    _ = Fintype.card ι := by simp

/-- **The weights of eq:pooled sum to one** wherever the pooled readout density is positive and
finite: the K-atom fiber law is a probability measure there. -/
theorem pooledFiberKernel_apply_univ (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i))
    (b : ι → R → B) (hb : ∀ i, Measurable (b i)) {r : R}
    (h0 : (∑ j, p j r) ≠ 0) (htop : (∑ j, p j r) ≠ ∞) :
    pooledFiberKernel p hp b hb r Set.univ = 1 := by
  rw [pooledFiberKernel_apply, Measure.finsetSum_apply]
  simp only [Measure.smul_apply, smul_eq_mul, measure_univ, mul_one]
  simp only [div_eq_mul_inv]
  rw [← Finset.sum_mul]
  exact ENNReal.mul_inv_cancel h0 htop

variable (μ : Measure R) [SFinite μ]

/-- **The pooled archive disintegrates with the K-atom kernel** (compProd form): the sum of the
K graph measures — readout density `p i` against `μ`, Dirac fiber at the conditional mode
`b i (x_R)` — equals the pooled readout marginal `μ · (∑ j, p j)` composed with
`pooledFiberKernel`.  The identity is exact (everywhere, not just a.e.) thanks to the pointwise
finiteness hypothesis on the densities. -/
theorem pooledArchive_eq_compProd
    (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i)) (hfin : ∀ i r, p i r ≠ ∞)
    (b : ι → R → B) (hb : ∀ i, Measurable (b i)) :
    (∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i))
      = (μ.withDensity fun r => ∑ j, p j r) ⊗ₘ pooledFiberKernel p hp b hb := by
  have hSmeas : Measurable fun r => ∑ j, p j r := Finset.measurable_sum _ fun j _ => hp j
  have hSne : ∀ r, (∑ j, p j r) ≠ ∞ := fun r =>
    (ENNReal.sum_lt_top.2 fun j _ => lt_top_iff_ne_top.2 (hfin j r)).ne
  ext s hs
  have hDmeas : ∀ i, Measurable fun r =>
      Kernel.deterministic (b i) (hb i) r (Prod.mk r ⁻¹' s) :=
    fun i => Kernel.measurable_kernel_prodMk_left hs
  rw [Measure.finsetSum_apply]
  have hterm : ∀ i, ((μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)) s
      = ∫⁻ r, p i r * Kernel.deterministic (b i) (hb i) r (Prod.mk r ⁻¹' s) ∂μ := by
    intro i
    rw [Measure.compProd_apply hs,
      lintegral_withDensity_eq_lintegral_mul _ (hp i) (hDmeas i)]
    rfl
  simp_rw [hterm]
  rw [← lintegral_finsetSum _ fun i _ => (hp i).mul (hDmeas i)]
  conv_rhs => rw [Measure.compProd_apply hs,
    lintegral_withDensity_eq_lintegral_mul _ hSmeas (Kernel.measurable_kernel_prodMk_left hs)]
  refine lintegral_congr fun r => ?_
  simp only [Pi.mul_apply, pooledFiberKernel_apply, Measure.finsetSum_apply,
    Measure.smul_apply, smul_eq_mul, Kernel.deterministic_apply]
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← mul_assoc, mul_comm (∑ j, p j r) (p i r / ∑ j, p j r),
    ENNReal.div_mul_cancel' (fun h0 => ?_) (fun htop => absurd htop (hSne r))]
  exact le_antisymm
    (h0 ▸ Finset.single_le_sum (f := fun j => p j r) (fun j _ => zero_le) (Finset.mem_univ i))
    zero_le

/-- The pooled archive's readout marginal is the summed density against the reference. -/
theorem pooledArchive_fst
    (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i))
    (b : ι → R → B) (hb : ∀ i, Measurable (b i)) :
    (∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)).fst
      = μ.withDensity fun r => ∑ j, p j r := by
  ext t ht
  rw [Measure.fst_apply ht, Measure.finsetSum_apply, withDensity_apply _ ht]
  have hterm : ∀ i, ((μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i))
      (Prod.fst ⁻¹' t) = ∫⁻ r in t, p i r ∂μ := by
    intro i
    rw [← Measure.fst_apply ht, Measure.fst_compProd, withDensity_apply _ ht]
  simp_rw [hterm]
  exact (lintegral_finsetSum _ fun i _ => hp i).symm

/-- The pooled archive disintegrates over its own readout marginal with the K-atom kernel. -/
theorem pooledArchive_disintegration
    (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i)) (hfin : ∀ i r, p i r ≠ ∞)
    (b : ι → R → B) (hb : ∀ i, Measurable (b i)) :
    (∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i))
      = (∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)).fst
          ⊗ₘ pooledFiberKernel p hp b hb := by
  rw [pooledArchive_fst μ p hp b hb]
  exact pooledArchive_eq_compProd μ p hp hfin b hb

/-- The paper's **uniform** mixture `(1/K) ∑ i` disintegrates with the SAME K-atom kernel: the
`1/K` prefactor cancels into the readout-tilted weights `p i / ∑ j, p j` of eq:pooled. -/
theorem pooledUniform_disintegration
    (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i)) (hfin : ∀ i r, p i r ≠ ∞)
    (b : ι → R → B) (hb : ∀ i, Measurable (b i)) :
    ((Fintype.card ι : ℝ≥0∞)⁻¹
        • ∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i))
      = ((Fintype.card ι : ℝ≥0∞)⁻¹
          • ∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)).fst
          ⊗ₘ pooledFiberKernel p hp b hb := by
  have hfst : ((Fintype.card ι : ℝ≥0∞)⁻¹
        • ∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)).fst
      = (Fintype.card ι : ℝ≥0∞)⁻¹
        • (∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)).fst :=
    Measure.map_smul _ _ _
  rw [hfst, pooledArchive_fst μ p hp b hb, Measure.compProd_smul_left,
    ← pooledArchive_eq_compProd μ p hp hfin b hb]

variable [StandardBorelSpace B] [Nonempty B]

/-- The uniform `1/K` scaling of a finite measure is finite: the scalar `(card ι)⁻¹` is finite
because `ι` is nonempty.  Local instance so that `condKernel` of the uniform mixture elaborates. -/
instance isFiniteMeasure_card_inv_smul {α : Type*} [MeasurableSpace α] [Nonempty ι]
    (ρ : Measure α) [IsFiniteMeasure ρ] :
    IsFiniteMeasure ((Fintype.card ι : ℝ≥0∞)⁻¹ • ρ) :=
  ρ.smul_finite (ENNReal.inv_ne_top.mpr (by exact_mod_cast Fintype.card_ne_zero))

/-- **eq:pooled, at the level of the canonical disintegration**: the uniform pooled archive's
conditional kernel over the readout IS the K-atom law
`∑ i, (p i (x_R) / ∑ j, p j (x_R)) • δ_{b i (x_R)}`, a.e. readout.  A.e. uniqueness of
disintegrations again; `Nonempty ι` (at least one method) and per-method finiteness make the
mixture a finite measure so mathlib's `condKernel` applies. -/
theorem pooledUniform_condKernel [Nonempty ι]
    (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i)) (hfin : ∀ i r, p i r ≠ ∞)
    (b : ι → R → B) (hb : ∀ i, Measurable (b i))
    [∀ i, IsFiniteMeasure (μ.withDensity (p i))] :
    ∀ᵐ r ∂((Fintype.card ι : ℝ≥0∞)⁻¹
        • ∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)).fst,
      ((Fintype.card ι : ℝ≥0∞)⁻¹
        • ∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)).condKernel r
      = ∑ i, (p i r / ∑ j, p j r) • Measure.dirac (b i r) := by
  filter_upwards [eq_condKernel_of_measure_eq_compProd (pooledFiberKernel p hp b hb)
    (pooledUniform_disintegration μ p hp hfin b hb)] with r hr
  rw [← pooledFiberKernel_apply p hp b hb r]
  exact hr.symm

/-- **Zero conditional width per atom, read off**: a.e. readout, the pooled fiber conditional
puts no mass outside the K reconstruction atoms `{b i (x_R) : i}` — each component is a Dirac,
so all pooled blind-fiber spread is inter-method disagreement between atoms, exactly as
`r:ensemble` says. -/
theorem pooledUniform_fiber_atomic [Nonempty ι]
    (p : ι → R → ℝ≥0∞) (hp : ∀ i, Measurable (p i)) (hfin : ∀ i r, p i r ≠ ∞)
    (b : ι → R → B) (hb : ∀ i, Measurable (b i))
    [∀ i, IsFiniteMeasure (μ.withDensity (p i))] :
    ∀ᵐ r ∂((Fintype.card ι : ℝ≥0∞)⁻¹
        • ∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)).fst,
      ((Fintype.card ι : ℝ≥0∞)⁻¹
        • ∑ i, (μ.withDensity (p i)) ⊗ₘ Kernel.deterministic (b i) (hb i)).condKernel r
        (Set.range fun i => b i r)ᶜ = 0 := by
  filter_upwards [pooledUniform_condKernel μ p hp hfin b hb] with r hr
  have hmeas : MeasurableSet ((Set.range fun i => b i r)ᶜ) :=
    (Set.finite_range _).measurableSet.compl
  rw [hr, Measure.finsetSum_apply]
  refine Finset.sum_eq_zero fun i _ => ?_
  rw [Measure.smul_apply, Measure.dirac_apply' _ hmeas,
    Set.indicator_of_notMem (by simp), smul_eq_mul, mul_zero]

end PooledEnsemble

end HilbertAssembly

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms HilbertAssembly.resolved_disintegration_exists
#print axioms HilbertAssembly.withDensity_fiberConst_eq_compProd_condKernel
#print axioms HilbertAssembly.condKernel_withDensity_fiberConst
#print axioms HilbertAssembly.curatedPrior_condKernel_freeze
#print axioms HilbertAssembly.measurable_emReweight_general
#print axioms HilbertAssembly.emReweight_congr
#print axioms HilbertAssembly.map_withDensity_measurableEquiv
#print axioms HilbertAssembly.resolvedBlindSplit_apply
#print axioms HilbertAssembly.apply_resolvedBlindSplit
#print axioms HilbertAssembly.resolvedBlindSplit_symm_blind
#print axioms HilbertAssembly.hilbert_freeze_assembled
#print axioms HilbertAssembly.pooledFiberKernel_apply_univ
#print axioms HilbertAssembly.pooledArchive_eq_compProd
#print axioms HilbertAssembly.pooledArchive_fst
#print axioms HilbertAssembly.pooledArchive_disintegration
#print axioms HilbertAssembly.pooledUniform_disintegration
#print axioms HilbertAssembly.pooledUniform_condKernel
#print axioms HilbertAssembly.pooledUniform_fiber_atomic
