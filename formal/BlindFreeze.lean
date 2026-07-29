import PriorLaundering
import Mathlib.Probability.Kernel.Composition.MeasureCompProd
import Mathlib.Probability.Kernel.MeasurableLIntegral
import Mathlib.LinearAlgebra.Matrix.SchurComplement
import Mathlib.MeasureTheory.Measure.WithDensity

/-!
# Proposition `p:blind` of the paper: the blind-fiber freeze

This file machine-checks the two parts of **Proposition `p:blind`** ("On the blind fibers the learned
prior is the assumption", exact) of the paper.

## Part 1 — the general fiber-conditional freeze (measure-theoretic, `GeneralFreeze`)

The regularizer, disintegrated along the resolved coordinate, is `reg = νR ⊗ₘ κreg`: a resolved
marginal `νR` and a blind fiber-conditional kernel `κreg = reg(· ∣ x_R)`.  The EM reweight
`g(x) = 𝔼_{y∼truth}[p(y∣x)/reg(y)]` is **fiber-constant** — `g(x) = g(x_R)` — because the likelihood
`p(y∣x) = p(y∣F(x))` reaches `x` only through `F(x) = A x_R`.  Multiplying `reg` by a fiber-constant
density therefore leaves the fiber conditional unchanged:

* `freeze_general` — the clean measure-level core:
  `(νR ⊗ₘ κreg).withDensity (fun p => f p.1) = (νR.withDensity f) ⊗ₘ κreg`.
  The conditional kernel `κreg` is **identical** on both sides; only the resolved marginal is
  reweighted (`νR ↦ νR.withDensity f`).  This is exactly `q_reg(· ∣ x_R) = reg(· ∣ x_R)`.
* `curatedPrior_blind_freeze` — reuses Theorem 1's Part A (`PriorLaundering.pLaw_partA`,
  `curatedPrior = reg.withDensity (emReweight)`): if the reweight is a.e. fiber-constant, then
  `curatedPrior = (νR.withDensity f) ⊗ₘ κreg`, the same blind fiber conditional `κreg` as `reg`.

## Part 2 — the Gaussian blind-block precision identity (linear algebra, `GaussianPrecision`)

In the `row(A) ⊕ ker A` split, write the covariance as a block matrix `Σ = [[A,B],[C,D]]` (blocks
resolved `R` and blind `B`).  The blind-block of the **precision** `Σ⁻¹` is the conditional precision
`(x_B ∣ x_R)`, which by the block-inverse (Schur complement) identity equals the inverse of the
conditional covariance:

* `invBlock₂₂_eq_schur_inv` — `(Σ⁻¹).toBlocks₂₂ = (D - C A⁻¹ B)⁻¹` (blind-block of precision =
  inverse Schur complement = conditional precision).  For the orthonormal blind selection `N`,
  `(Σ⁻¹).toBlocks₂₂ = Nᵀ Σ⁻¹ N` (`blindSel_conj`).
* `freeze_precision` — eq:freeze: two Gaussians with the **same blind-fiber conditional covariance**
  (Schur complement) have equal blind-block precisions, `Nᵀ Σ_q⁻¹ N = Nᵀ Σ_reg⁻¹ N` — the freeze
  fed back through the Schur bridge.  The equal-Schur hypothesis is what Part 1 delivers for
  Gaussians (equal conditionals ⇒ equal conditional covariances).

## Honesty

No `sorry`/`admit`.  Genuinely-needed regularity is an explicit hypothesis (σ-finiteness of the
resolved marginal / s-finiteness of the fiber kernel in Part 1; the likelihood reaching `x` only
through the resolved coordinate in the `curatedPrior` corollary — from which the reweight's
fiber-constancy is now DERIVED, not assumed; invertibility of `A = Σ_RR` and `Σ` in Part 2, both
automatic from positive-definiteness).
`#print axioms` on each main theorem lists only `propext`, `Classical.choice`, `Quot.sound`
(no `sorryAx`).  The hypothesis-reconciliation audit is the `Reconciliation` block at the end.
-/

open MeasureTheory ProbabilityTheory Matrix
open scoped ENNReal Matrix

namespace BlindFreeze

/-! ## Part 1 — the general fiber-conditional freeze (measure-theoretic) -/

section GeneralFreeze

variable {R B : Type*} [MeasurableSpace R] [MeasurableSpace B]

/-- **The measure-level freeze.** Multiplying the disintegrated measure `νR ⊗ₘ κ` by a
**fiber-constant** density `f ∘ fst` (a function of the resolved coordinate `x_R = p.1` alone) leaves
the fiber-conditional kernel `κ` unchanged; only the resolved marginal is reweighted:
`(νR ⊗ₘ κ).withDensity (fun p => f p.1) = (νR.withDensity f) ⊗ₘ κ`.

This is exactly `q_reg(· ∣ x_R) = reg(· ∣ x_R)`: the fiber-constant EM reweight cancels in the
conditional. -/
theorem freeze_general (νR : Measure R) [SFinite νR] (κ : Kernel R B) [IsSFiniteKernel κ]
    {f : R → ℝ≥0∞} (hf : Measurable f) :
    (νR ⊗ₘ κ).withDensity (fun p => f p.1) = (νR.withDensity f) ⊗ₘ κ := by
  ext s hs
  rw [withDensity_apply _ hs, Measure.compProd_apply hs,
    lintegral_withDensity_eq_lintegral_mul _ hf (Kernel.measurable_kernel_prodMk_left hs),
    ← lintegral_indicator hs (fun p => f p.1),
    Measure.lintegral_compProd
      ((show Measurable (fun p : R × B => f p.1) from hf.comp measurable_fst).indicator hs)]
  -- Both sides are `∫⁻ r, f r * κ r (mk r ⁻¹' s) ∂νR`; reduce the inner blind integral.
  have hinner : ∀ r, ∫⁻ b, s.indicator (fun p => f p.1) (r, b) ∂(κ r)
      = f r * κ r (Prod.mk r ⁻¹' s) := by
    intro r
    have hpt : ∀ b, s.indicator (fun p => f p.1) (r, b)
        = (Prod.mk r ⁻¹' s).indicator (fun _ => f r) b := by
      intro b
      by_cases hb : (r, b) ∈ s
      · rw [Set.indicator_of_mem hb, Set.indicator_of_mem (show b ∈ Prod.mk r ⁻¹' s from hb)]
      · rw [Set.indicator_of_notMem hb, Set.indicator_of_notMem (show b ∉ Prod.mk r ⁻¹' s from hb)]
    simp_rw [hpt]
    rw [lintegral_indicator_const (hs.preimage measurable_prodMk_left)]
  simp only [hinner, Pi.mul_apply]

/-!
The fiber-constancy of the reweight — the paper's trivial step "since `p(y∣x)=p(y∣A x_R)`, the
reweight `g(x)=g(x_R)`" — is now **derived**, not assumed.  We model the likelihood faithfully as a
kernel reaching `x` only through the resolved coordinate (`κ x = κ x'` whenever `x.1 = x'.1`). -/

/-- Measurability of the EM reweight (mirrors the internal step of Theorem 1's Part A). -/
lemma measurable_emReweight {𝓧 : Type*} [MeasurableSpace 𝓧]
    [MeasurableSpace.CountableOrCountablyGenerated (R × B) 𝓧]
    (κ : Kernel (R × B) 𝓧) [IsMarkovKernel κ] (μ ν : Measure (R × B))
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    Measurable (PriorLaundering.emReweight κ μ ν) := by
  unfold PriorLaundering.emReweight
  exact (Kernel.measurable_rnDeriv κ (Kernel.const _ (κ ∘ₘ μ))).lintegral_kernel_prod_right'
    (κ := Kernel.const _ (κ ∘ₘ ν))

/-- **The EM reweight is fiber-constant.** If the likelihood `κ` reaches `x` only through the resolved
coordinate (`κ x = κ x'` whenever `x.1 = x'.1` — the paper's `p(y∣x)=p(y∣A x_R)`), the reweight takes
equal values on a fiber.  The `κ ∘ₘ μ`-a.e. equality of the RN-derivatives (Kernel-vs-measure
`rnDeriv_eq_rnDeriv_measure`) is moved to the `κ ∘ₘ ν`-integral by `κ ∘ₘ ν ≪ κ ∘ₘ μ`. -/
lemma emReweight_fiber_const {𝓧 : Type*} [MeasurableSpace 𝓧]
    [MeasurableSpace.CountableOrCountablyGenerated (R × B) 𝓧]
    (κ : Kernel (R × B) 𝓧) [IsMarkovKernel κ] (μ ν : Measure (R × B))
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hκ : ∀ x x' : R × B, x.1 = x'.1 → κ x = κ x')
    (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ)
    {x x' : R × B} (hxx : x.1 = x'.1) :
    PriorLaundering.emReweight κ μ ν x = PriorLaundering.emReweight κ μ ν x' := by
  unfold PriorLaundering.emReweight
  refine lintegral_congr_ae (h_ac.ae_le ?_)
  have hx := Kernel.rnDeriv_eq_rnDeriv_measure (κ := κ) (η := Kernel.const (R × B) (κ ∘ₘ μ)) (a := x)
  have hx' := Kernel.rnDeriv_eq_rnDeriv_measure (κ := κ) (η := Kernel.const (R × B) (κ ∘ₘ μ)) (a := x')
  rw [Kernel.const_apply] at hx hx'
  filter_upwards [hx, hx'] with y hy hy'
  rw [hy, hy', hκ x x' hxx]

/-- The reweight **factors through the resolved coordinate**: there is a measurable `f : R → ℝ≥0∞`
with `emReweight κ μ ν = fun p => f p.1`.  This discharges the fiber-constancy the freeze needs. -/
lemma emReweight_factors {𝓧 : Type*} [MeasurableSpace 𝓧] [Nonempty B]
    [MeasurableSpace.CountableOrCountablyGenerated (R × B) 𝓧]
    (κ : Kernel (R × B) 𝓧) [IsMarkovKernel κ] (μ ν : Measure (R × B))
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hκ : ∀ x x' : R × B, x.1 = x'.1 → κ x = κ x')
    (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ) :
    ∃ f : R → ℝ≥0∞, Measurable f ∧ PriorLaundering.emReweight κ μ ν = fun p => f p.1 := by
  classical
  refine ⟨fun r => PriorLaundering.emReweight κ μ ν (r, Classical.arbitrary B),
    (measurable_emReweight κ μ ν).comp (measurable_id.prodMk measurable_const), ?_⟩
  funext p
  exact emReweight_fiber_const κ μ ν hκ h_ac rfl

/-- **The freeze for the curated prior (reusing Theorem 1, Part A) — fully self-contained.**

`curatedPrior = reg.withDensity (emReweight)` (`PriorLaundering.pLaw_partA`).  With the regularizer
disintegrated as `reg = νR ⊗ₘ κreg` and the likelihood reaching `x` only through the resolved
coordinate (`hκ`), the reweight factors as `fun p => f p.1` (`emReweight_factors`), so multiplying by
it — a fiber-constant density — keeps `reg`'s blind fiber conditional `κreg` (`freeze_general`):
`curatedPrior = (νR.withDensity f) ⊗ₘ κreg`, i.e. `q_reg(· ∣ x_R) = reg(· ∣ x_R)`. -/
theorem curatedPrior_blind_freeze
    {𝓧 : Type*} [MeasurableSpace 𝓧] [Nonempty R] [Nonempty B]
    [StandardBorelSpace (R × B)]
    [MeasurableSpace.CountableOrCountablyGenerated (R × B) 𝓧]
    (κ : Kernel (R × B) 𝓧) [IsMarkovKernel κ]
    (νR : Measure R) [IsProbabilityMeasure νR] (κreg : Kernel R B) [IsMarkovKernel κreg]
    (ν : Measure (R × B)) [IsProbabilityMeasure ν]
    (hκ : ∀ x x' : R × B, x.1 = x'.1 → κ x = κ x')
    (h_dom : ∀ᵐ x ∂(νR ⊗ₘ κreg), κ x ≪ κ ∘ₘ (νR ⊗ₘ κreg))
    (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ (νR ⊗ₘ κreg)) :
    ∃ f : R → ℝ≥0∞, Measurable f ∧
      PriorLaundering.curatedPrior κ (νR ⊗ₘ κreg) ν = (νR.withDensity f) ⊗ₘ κreg := by
  obtain ⟨f, hf, hfeq⟩ := emReweight_factors κ (νR ⊗ₘ κreg) ν hκ h_ac
  exact ⟨f, hf, by rw [PriorLaundering.pLaw_partA κ (νR ⊗ₘ κreg) ν h_dom h_ac, hfeq,
    freeze_general νR κreg hf]⟩

/-- **The loop's freeze (general half of `p:recover`): the common fiber conditional stays frozen for
all rounds.** Iterating **any** number of fiber-constant reweights (each `fᵢ ∘ P_R`, one per round)
starting from `νR ⊗ₘ κ` leaves the blind fiber conditional `κ` **identical** — only the resolved
marginal accumulates the reweights.  Since each round's reweight is constant along its operator's
kernel fibers, and the alternating loop's common kernel `⋂ᵢ ker Aᵢ` lies in every one of them, the
common-kernel conditional is frozen throughout — "each round freezes its own kernel, so the
intersection stays frozen" (`p:blind` iterated). -/
theorem freeze_iterate (κ : Kernel R B) [IsSFiniteKernel κ] (fs : List (R → ℝ≥0∞)) :
    (∀ f ∈ fs, Measurable f) → ∀ (νR : Measure R), SFinite νR →
      fs.foldl (fun μ f => μ.withDensity (fun p => f p.1)) (νR ⊗ₘ κ)
        = (fs.foldl (fun ρ f => ρ.withDensity f) νR) ⊗ₘ κ := by
  induction fs with
  | nil => intro _ νR _; rfl
  | cons f fs' ih =>
    intro hfs νR hνR
    haveI := hνR
    simp only [List.foldl_cons]
    rw [freeze_general νR κ (hfs f (List.mem_cons_self ..))]
    exact ih (fun g hg => hfs g (List.mem_cons_of_mem _ hg)) (νR.withDensity f) inferInstance

/-- **The pooled-archive freeze (`r:ensemble`): mixing archive laws with a common blind fiber
conditional keeps that conditional.** If each per-measurement archive law is `ρ y ⊗ₘ κ` — a resolved
part `ρ y` times the **common** blind conditional `κ` — then pooling over the measurements (mixing
against `μ`) gives `(μ.bind ρ) ⊗ₘ κ`: the same fiber conditional `κ`, only the resolved marginal is
mixed.  "A mixture of kernels with a common fiber conditional keeps that conditional." -/
theorem mixture_freeze {D : Type*} [MeasurableSpace D] (μ : Measure D) [SFinite μ]
    (ρ : Kernel D R) [IsSFiniteKernel ρ] (κ : Kernel R B) [IsSFiniteKernel κ] :
    μ.bind (fun y => ρ y ⊗ₘ κ) = (μ.bind ρ) ⊗ₘ κ := by
  have hmeas : Measurable (fun y => ρ y ⊗ₘ κ) := by
    apply Measure.measurable_of_measurable_coe
    intro t ht
    simp_rw [Measure.compProd_apply ht]
    exact ((Kernel.measurable_kernel_prodMk_left ht).comp measurable_snd).lintegral_kernel_prod_right'
      (κ := ρ)
  ext s hs
  rw [Measure.bind_apply hs hmeas.aemeasurable]
  simp_rw [Measure.compProd_apply hs]
  rw [Measure.lintegral_bind ρ.aemeasurable (Kernel.measurable_kernel_prodMk_left hs).aemeasurable]

end GeneralFreeze

/-! ## Part 2 — the Gaussian blind-block precision identity (linear algebra) -/

section GaussianPrecision

variable {R B : Type*} [Fintype R] [Fintype B] [DecidableEq R] [DecidableEq B]

/-- **The blind-block of the precision = inverse Schur complement (the conditional precision).**

For a block covariance `Σ = [[A,B],[C,D]]` with `A` and `Σ` invertible, the bottom-right block of the
inverse precision is the inverse of the Schur complement `D - C A⁻¹ B` (the conditional covariance of
the blind block given the resolved block):
`(Σ⁻¹).toBlocks₂₂ = (D - C A⁻¹ B)⁻¹`.

`A` and `Σ` invertible are automatic when `Σ ≻ 0` (`A = Σ_RR` is a principal submatrix, `Σ` positive
definite); the Schur complement's invertibility is then derived. -/
theorem invBlock₂₂_eq_schur_inv
    (A : Matrix R R ℝ) (Bm : Matrix R B ℝ) (C : Matrix B R ℝ) (D : Matrix B B ℝ)
    [Invertible A] [Invertible (fromBlocks A Bm C D)] :
    (fromBlocks A Bm C D)⁻¹.toBlocks₂₂ = (D - C * A⁻¹ * Bm)⁻¹ := by
  letI := invertibleOfFromBlocks₁₁Invertible A Bm C D
  rw [← invOf_eq_nonsing_inv (fromBlocks A Bm C D), invOf_fromBlocks₁₁_eq, toBlocks_fromBlocks₂₂,
    invOf_eq_nonsing_inv (D - C * ⅟A * Bm), invOf_eq_nonsing_inv A]

/-- The **orthonormal blind selection** `N : Matrix (R ⊕ B) B`, `N i j = ⟦i = inr j⟧`, whose columns
are the standard basis of the blind block `ker A` in the `row(A) ⊕ ker A` split. -/
def blindSel (R B : Type*) [DecidableEq R] [DecidableEq B] : Matrix (R ⊕ B) B ℝ :=
  Matrix.of fun i j => if i = Sum.inr j then 1 else 0

/-- `Nᵀ M N` selects the blind–blind block: `Nᵀ M N = M.toBlocks₂₂`.  So the paper's `Nᵀ Σ⁻¹ N` is
`(Σ⁻¹).toBlocks₂₂`. -/
lemma blindSel_conj (M : Matrix (R ⊕ B) (R ⊕ B) ℝ) :
    (blindSel R B)ᵀ * M * blindSel R B = M.toBlocks₂₂ := by
  have hcol : M * blindSel R B = M.submatrix id Sum.inr := by
    ext k j
    rw [Matrix.mul_apply]
    simp only [blindSel, Matrix.of_apply, Matrix.submatrix_apply, id_eq]
    rw [Finset.sum_eq_single_of_mem (Sum.inr j) (Finset.mem_univ _)
      (fun b _ hb => by rw [if_neg hb, mul_zero]), if_pos rfl, mul_one]
  have hrow : (blindSel R B)ᵀ * (M.submatrix id Sum.inr) = M.toBlocks₂₂ := by
    ext i j
    rw [Matrix.mul_apply]
    simp only [Matrix.transpose_apply, blindSel, Matrix.of_apply, Matrix.submatrix_apply, id_eq,
      Matrix.toBlocks₂₂]
    rw [Finset.sum_eq_single_of_mem (Sum.inr i) (Finset.mem_univ _)
      (fun l _ hl => by rw [if_neg hl, zero_mul]), if_pos rfl, one_mul]
  rw [Matrix.mul_assoc, hcol, hrow]

/-- **The blind-fiber freeze in precision form (eq:freeze).**

Two Gaussian covariances `Σ_q`, `Σ_reg` (block matrices, with the resolved blocks and both full
matrices invertible) that share the **same blind-fiber conditional covariance** — i.e. the same
Schur complement `D - C A⁻¹ B` — have equal blind-block precisions:
`Nᵀ Σ_q⁻¹ N = Nᵀ Σ_reg⁻¹ N`, `range N = ker A`.

The equal-Schur hypothesis is exactly what Part 1's freeze gives in the Gaussian case (equal
blind-fiber conditionals ⇒ equal conditional covariances); the Schur bridge
`invBlock₂₂_eq_schur_inv` turns it into the precision identity the paper states. -/
theorem freeze_precision
    (Aq : Matrix R R ℝ) (Bq : Matrix R B ℝ) (Cq : Matrix B R ℝ) (Dq : Matrix B B ℝ)
    (Ar : Matrix R R ℝ) (Br : Matrix R B ℝ) (Cr : Matrix B R ℝ) (Dr : Matrix B B ℝ)
    [Invertible Aq] [Invertible (fromBlocks Aq Bq Cq Dq)]
    [Invertible Ar] [Invertible (fromBlocks Ar Br Cr Dr)]
    (hSchur : Dq - Cq * Aq⁻¹ * Bq = Dr - Cr * Ar⁻¹ * Br) :
    (blindSel R B)ᵀ * (fromBlocks Aq Bq Cq Dq)⁻¹ * blindSel R B
      = (blindSel R B)ᵀ * (fromBlocks Ar Br Cr Dr)⁻¹ * blindSel R B := by
  rw [blindSel_conj, blindSel_conj, invBlock₂₂_eq_schur_inv, invBlock₂₂_eq_schur_inv, hSchur]

end GaussianPrecision

/-!
## Reconciliation — what `p:blind`'s formalization needs vs. what the paper states

Classification **(a)** already in the text; **(b)** load-bearing but not stated (with the honest
minimal hypothesis); **(c)** automatic in the paper's setting.

### Part 1 (general freeze)

* `reg = νR ⊗ₘ κreg` — **(c)**: the disintegration of the regularizer along the resolved coordinate.
  The paper writes `reg(x_R, x_B)` and `reg(x_B ∣ x_R)`; modelling `reg` as a resolved marginal plus a
  fiber-conditional kernel is exactly this (any standard-Borel `reg` disintegrates so).  The freeze
  conclusion — same kernel `κreg` on both sides — **is** `q_reg(· ∣ x_R) = reg(· ∣ x_R)`.
* `SFinite νR`, `IsSFiniteKernel κreg` — **(b, mild)**: σ-finiteness / s-finiteness, the regularity the
  paper's integrals silently assume; made explicit (`IsProbabilityMeasure`/`IsMarkovKernel` in the
  corollary, which imply these).
* `hκ : ∀ x x', x.1 = x'.1 → κ x = κ x'` (the likelihood reaches `x` only through the resolved
  coordinate, `p(y∣x)=p(y∣A x_R)`) — **(a)**: exactly the paper's standing hypothesis on the noise
  kernel.  From it the reweight's fiber-constancy `emReweight = fun p => f p.1` is now **DERIVED**
  (`emReweight_fiber_const` / `emReweight_factors`), not assumed — closing the layer previously
  flagged: `rnDeriv_eq_rnDeriv_measure` turns the Kernel RN-derivative into the measure one
  `∂(κ x)/∂(κ ∘ₘ μ)`, which depends on `x` only through `κ x`, and `κ ∘ₘ ν ≪ κ ∘ₘ μ` carries the a.e.
  equality into the reweight's integral.  (Needs `[Nonempty B]` to name a fiber representative for
  the explicit `f`; automatic when the blind subspace is nontrivial.)  So Part 1 is fully
  self-contained, matching `p:law`/`p:map`.
* `h_dom`, `h_ac` — **(a)**: the two absolute-continuity conditions of Theorem 1's Part A, the paper's
  implicit full-support (`reg` dominates the data marginals).

### Part 2 (Gaussian blind-block precision)

* `Invertible A` (`A = Σ_RR`), `Invertible Σ` — **(c)**: automatic from `Σ ≻ 0` (`Σ_RR` is a principal
  submatrix of a positive-definite matrix, hence positive definite hence invertible; `Σ` positive
  definite hence invertible).  Stated as explicit `Invertible` instances (the honest minimal need);
  the Schur complement's invertibility is then derived (`invertibleOfFromBlocks₁₁Invertible`), not
  assumed.
* `hSchur : Dq - Cq Aq⁻¹ Bq = Dr - Cr Ar⁻¹ Br` (equal Schur complements = equal blind-fiber
  conditional covariances) — **(a)**: this is the content of Part 1's freeze specialized to Gaussians
  (equal conditionals ⇒ equal conditional covariances).  The paper packages the step "conditional
  precision is exactly the blind block `Nᵀ Σ⁻¹ N`"; the formalization separates the trivial part
  (`Nᵀ Σ⁻¹ N = (Σ⁻¹).toBlocks₂₂`, `blindSel_conj`) from the non-trivial Schur bridge
  (`(Σ⁻¹).toBlocks₂₂ = (Schur)⁻¹`, `invBlock₂₂_eq_schur_inv`) — the latter is the genuine linear
  algebra behind eq:freeze.

Nothing here makes `p:blind` wrong, and the previously-flagged layer is now closed: the passage from
"the likelihood factors through the resolved coordinate" to "the reweight is fiber-constant" is
**derived** (`emReweight_fiber_const`), so both parts stand on explicit, faithful hypotheses only.
The one small surfacing is `[Nonempty B]` (a nontrivial blind subspace), needed only to name a fiber
representative for the explicit density `f`. -/

end BlindFreeze

/-! ### Axiom audit

The doc comment above claims these rest only on Lean's three standard axioms. These commands
are what make that checkable rather than asserted: each must print `propext, Classical.choice,
Quot.sound` and never `sorryAx`. -/

#print axioms BlindFreeze.freeze_iterate
#print axioms BlindFreeze.mixture_freeze
#print axioms BlindFreeze.freeze_precision
