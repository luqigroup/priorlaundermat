import PriorLaundering
import Mathlib.Probability.Kernel.Composition.MeasureComp
import Mathlib.Probability.Kernel.Composition.CompMap
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-!
# Corollary `c:nonident` of the paper: the shortfall is not identifiable

This file machine-checks **Corollary `c:nonident`** ("The shortfall is not identifiable from the
measurements", for any likelihood that factors through the forward operator).

Thesis: the data marginal `truth(y) = ∫ p(y∣A x) truth(x) dx` — and hence the curated archive —
depends on the truth **only through its resolved marginal**.  So two truths that share a resolved
marginal but carry different blind-fiber conditionals are indistinguishable in law and train the
**identical** curated prior; the truth's entire blind-subspace law (in particular `s_⋆`, `δ`, the
ratio `s_reg/s_⋆` of `p:cover`) is non-identifiable.

We model the likelihood faithfully as a kernel reaching `x` only through the resolved coordinate
`x_R = Prod.fst x`: `κ x = η x.1` (i.e. `p(y∣x) = p(y∣x_R) = η(x_R)`), exactly the paper's
`p(y∣x)=p(y∣F(x))` with `F = F ∘ P_R` (`A = A P_R`).

## Results

* `data_marginal_resolved_only` — `κ ∘ₘ ν = η ∘ₘ (ν.map P_R)`: the measurement law is a functional
  of the truth's resolved marginal `P_{R#} truth` alone.  (This is the computation that makes
  `reg(y)` resolved-only in the fixed-point argument for `c:fix`.)
* `data_marginal_eq_of_resolved_eq` — two truths with the same resolved marginal induce the identical
  measurement law.
* `curatedPrior_depends_only_on_data_marginal` / `nonident_curatedPrior` — the curated prior depends
  on the truth only through the data marginal, so two truths with the same resolved marginal train
  the **identical** curated prior: the in-law non-identifiability.
* `witness_cov_annihilated`, `witness_mean_annihilated` — the linear--Gaussian witness (eq:witness):
  since `A N = 0`, every added blind block leaves `A Σ Aᵀ` and `A μ` — hence the Gaussian data
  marginal `𝒩(A μ_⋆, A Σ_⋆ Aᵀ + Γ)` — unchanged, while `s_⋆`, `δ`, and the resolved–blind
  cross-covariance move freely.

## Honesty

No `sorry`/`admit`.  `#print axioms` on each main theorem lists only `propext`, `Classical.choice`,
`Quot.sound` (no `sorryAx`).  Reconciliation audit at the end.
-/

open MeasureTheory ProbabilityTheory Matrix
open scoped ENNReal Matrix

namespace NonIdentifiability

/-! ## The general (measure-level) non-identifiability -/

section General

variable {R B 𝓧 : Type*} [MeasurableSpace R] [MeasurableSpace B] [MeasurableSpace 𝓧]

/-- **The data marginal depends on the truth only through its resolved marginal.**
If the likelihood reaches `x` only through the resolved coordinate (`κ x = η x.1`, i.e.
`p(y∣x)=p(y∣x_R)`), the measurement law `κ ∘ₘ ν` equals `η ∘ₘ (ν.map P_R)` — a functional of the
truth's resolved marginal `ν.map P_R = P_{R#} truth` alone. -/
theorem data_marginal_resolved_only
    (κ : Kernel (R × B) 𝓧) (η : Kernel R 𝓧) (hκ : ∀ x, κ x = η x.1) (ν : Measure (R × B)) :
    κ ∘ₘ ν = η ∘ₘ (ν.map Prod.fst) := by
  ext s hs
  rw [Measure.bind_apply hs κ.aemeasurable, Measure.bind_apply hs η.aemeasurable,
    lintegral_map (η.measurable_coe hs) measurable_fst]
  simp_rw [hκ]

/-- **Two truths with the same resolved marginal induce the identical measurement law.**
The blind-fiber conditional `truth(x_B ∣ x_R)` is free and invisible to the data. -/
theorem data_marginal_eq_of_resolved_eq
    (κ : Kernel (R × B) 𝓧) (η : Kernel R 𝓧) (hκ : ∀ x, κ x = η x.1) (ν ν' : Measure (R × B))
    (h : ν.map Prod.fst = ν'.map Prod.fst) :
    κ ∘ₘ ν = κ ∘ₘ ν' := by
  rw [data_marginal_resolved_only κ η hκ ν, data_marginal_resolved_only κ η hκ ν', h]

end General

/-! ## Identical trained prior (reusing `PriorLaundering.curatedPrior`) -/

section Curated

variable {R B 𝓧 : Type*} [MeasurableSpace R] [MeasurableSpace B] [MeasurableSpace 𝓧]
  [StandardBorelSpace (R × B)] [Nonempty (R × B)]

/-- The **curated prior depends on the truth `ν` only through the data marginal `κ ∘ₘ ν`** (it is
`(κ†reg) ∘ₘ (κ ∘ₘ ν)` by definition). -/
theorem curatedPrior_depends_only_on_data_marginal
    (κ : Kernel (R × B) 𝓧) [IsMarkovKernel κ] (μ ν ν' : Measure (R × B))
    [IsProbabilityMeasure μ] (h : κ ∘ₘ ν = κ ∘ₘ ν') :
    PriorLaundering.curatedPrior κ μ ν = PriorLaundering.curatedPrior κ μ ν' := by
  unfold PriorLaundering.curatedPrior
  rw [h]

/-- **`c:nonident` — the identical trained prior.**  For a likelihood factoring through the resolved
projection, two truths `ν`, `ν'` with the **same resolved marginal** but arbitrary, different
blind-fiber conditionals train the **identical** curated prior.  The coverage shortfall (a functional
of the truth's blind conditional) therefore cannot be identified from the measurements. -/
theorem nonident_curatedPrior
    (κ : Kernel (R × B) 𝓧) [IsMarkovKernel κ] (η : Kernel R 𝓧) (hκ : ∀ x, κ x = η x.1)
    (μ ν ν' : Measure (R × B)) [IsProbabilityMeasure μ]
    (h : ν.map Prod.fst = ν'.map Prod.fst) :
    PriorLaundering.curatedPrior κ μ ν = PriorLaundering.curatedPrior κ μ ν' :=
  curatedPrior_depends_only_on_data_marginal κ μ ν ν'
    (data_marginal_eq_of_resolved_eq κ η hκ ν ν' h)

end Curated

/-! ## The linear--Gaussian witness (eq:witness) -/

section Witness

variable {m n k : Type*} [Fintype n] [Fintype k]

/-- **The witness leaves `A Σ Aᵀ` fixed.** Since `A N = 0` (the added directions are blind), the
symmetric block `N W Nᵀ` of the witness perturbation eq:witness is annihilated:
`A (N W Nᵀ) Aᵀ = 0`.  Hence the Gaussian data covariance `A Σ_⋆ Aᵀ + Γ` is unchanged. -/
theorem witness_cov_annihilated (A : Matrix m n ℝ) (N : Matrix n k ℝ) (hAN : A * N = 0)
    (W : Matrix k k ℝ) :
    A * (N * W * Nᵀ) * Aᵀ = 0 := by
  calc A * (N * W * Nᵀ) * Aᵀ = A * N * (W * (Nᵀ * Aᵀ)) := by simp only [Matrix.mul_assoc]
    _ = 0 := by rw [hAN]; simp

/-- **The witness leaves `A μ` fixed.** Since `A N = 0`, the mean shift `N w` of eq:witness is
annihilated: `A (N w) = 0`. -/
theorem witness_mean_annihilated (A : Matrix m n ℝ) (N : Matrix n k ℝ) (hAN : A * N = 0)
    (w : k → ℝ) :
    A *ᵥ (N *ᵥ w) = 0 := by
  rw [mulVec_mulVec, hAN, zero_mulVec]

end Witness

/-!
## Reconciliation — what `c:nonident`'s formalization needs vs. what the paper states

**(a)** already in the text; **(b)** load-bearing but not stated; **(c)** automatic in the setting.

* `κ x = η x.1` (the likelihood reaches `x` only through the resolved coordinate,
  `p(y∣x)=p(y∣F(x))`, `A = A P_R`) — **(a)**: the corollary's standing hypothesis ("any likelihood
  that factors through the forward operator"; "it rests only on the likelihood's fiber-constancy").
  The whole result flows from it via `data_marginal_resolved_only`.
* the truth `ν`, `ν'` are **arbitrary** measures (no positive-definiteness, no Gaussianity) — the
  general statement holds for any two truths sharing a resolved marginal, matching "arbitrary,
  different blind-fiber conditional kernels".  **(a)**.
* `curatedPrior = (κ†reg) ∘ₘ (κ ∘ₘ ν)` — **(a)**: the definition (eq:curated); the ν-dependence is
  through the data marginal alone, so equal data marginals ⇒ equal trained prior.  Reuses Theorem 1's
  `curatedPrior`; no extra hypothesis.
* the `StandardBorel`/`Nonempty` instances on `R × B` and `IsProbabilityMeasure reg` — **(c)**: the
  standing regularity of Theorem 1's `curatedPrior` (needed to form the Bayes inverse `κ†reg`), not
  new to `c:nonident`.
* Witness (`A N = 0 ⇒` blocks annihilated) — **(a)**: the paper's `A N = 0`, `A P_R = A`; the
  annihilation `A (N W Nᵀ) Aᵀ = 0`, `A (N w) = 0` is the elementary calculation eq:witness rests on.

Nothing surfaced beyond the paper's own hypotheses: `c:nonident` is exactly the resolved-only
dependence of the data marginal (and hence of the curated prior), and it is fully derived here from
the single hypothesis that the likelihood factors through the resolved coordinate.
-/

end NonIdentifiability

/-! ### Axiom audit

The doc comment above claims these rest only on Lean's three standard axioms. These commands
are what make that checkable rather than asserted: each must print `propext, Classical.choice,
Quot.sound` and never `sorryAx`. -/

#print axioms NonIdentifiability.nonident_curatedPrior
#print axioms NonIdentifiability.curatedPrior_depends_only_on_data_marginal
#print axioms NonIdentifiability.witness_cov_annihilated
#print axioms NonIdentifiability.witness_mean_annihilated
