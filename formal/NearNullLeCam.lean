import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Measure.Real

/-!
# Le Cam two-point testing lower bound: the near-null undetectability engine (`p:nearnull`)

The exact non-identifiability `c:nonident` (`NonIdentifiability.lean`) says two truths differing only
on the operator's **kernel** train the identical prior and induce the *identical* data law: no test can
tell them apart at all. The near-null results `p:nearnull` / `p:mapnearnull` are the **quantitative**
version: on a near-null direction (small but nonzero illumination) the two data laws are no longer
identical but *close*, so the shortfall is not undetectable outright yet still needs on the order of
`1/σ²` samples to see — a Le Cam two-point testing lower bound.

This file machine-checks the **logical engine** of that bound, self-contained and sorry-free: if two
data laws `P`, `Q` are `ε`-close (every event's probability differs by at most `ε`, the statistical
total-variation notion), then *every* test has total error at least `1 - ε`. At `ε = 0` this recovers
total indistinguishability — the testing form of `c:nonident`; for `ε` small (the near-null regime) no
test does much better than a coin flip. What the engine reduces the near-null claim to is a single
analytic input — a bound on `ε` for the two near-null Gaussian data laws (a same-covariance Gaussian
KL fed through Pinsker), the χ²-between-Gaussians step mathlib does not yet provide.

A **test** is a measurable set `A` (reject `P`, decide `Q`, on `A`). Its two error probabilities are the
type-I error `P A` (`P` true, decided `Q`) and the type-II error `Q Aᶜ` (`Q` true, decided `P`).

## Results
* `lecam_two_point` — `1 - ε ≤ P.real A + Q.real Aᶜ`: every test's summed error is at least `1 - ε`.
* `lecam_minimax` — `(1 - ε)/2 ≤ max (P.real A) (Q.real Aᶜ)`: no test drives *both* errors below
  `(1 - ε)/2`.
* `lecam_exact` — for `P = Q` (the `c:nonident` endpoint), every test has summed error at least `1`:
  it cannot beat guessing.

## Honesty
No `sorry`/`admit`; `#print axioms` on each result lists only `propext`, `Classical.choice`,
`Quot.sound`.
-/

open MeasureTheory Set

namespace NearNull

variable {Ω : Type*} [MeasurableSpace Ω] {P Q : Measure Ω}
  [IsProbabilityMeasure P] [IsProbabilityMeasure Q] {ε : ℝ}

/-- Complement of a probability, real-valued: `Q.real Aᶜ = 1 - Q.real A`. -/
private theorem probReal_compl {A : Set Ω} (hA : MeasurableSet A) : Q.real Aᶜ = 1 - Q.real A := by
  have h := measureReal_add_measureReal_compl (μ := Q) hA
  rw [probReal_univ] at h
  linarith

/-- **Le Cam two-point testing lower bound.** If `P` and `Q` are `ε`-close — every event's probability
differs by at most `ε` (the total-variation notion of closeness) — then every test `A` has total error
`P.real A + Q.real Aᶜ ≥ 1 - ε`. Detection is impossible below that floor, uniformly over tests. -/
theorem lecam_two_point (hclose : ∀ B, MeasurableSet B → |P.real B - Q.real B| ≤ ε)
    {A : Set Ω} (hA : MeasurableSet A) : 1 - ε ≤ P.real A + Q.real Aᶜ := by
  rw [probReal_compl hA]
  have hb := abs_le.mp (hclose A hA)
  linarith [hb.1]

/-- **Minimax form.** No test can drive *both* the type-I error `P.real A` and the type-II error
`Q.real Aᶜ` below `(1 - ε)/2`: their maximum is at least `(1 - ε)/2`. -/
theorem lecam_minimax (hclose : ∀ B, MeasurableSet B → |P.real B - Q.real B| ≤ ε)
    {A : Set Ω} (hA : MeasurableSet A) : (1 - ε) / 2 ≤ max (P.real A) (Q.real Aᶜ) := by
  have h := lecam_two_point hclose hA
  have h1 : P.real A ≤ max (P.real A) (Q.real Aᶜ) := le_max_left _ _
  have h2 : Q.real Aᶜ ≤ max (P.real A) (Q.real Aᶜ) := le_max_right _ _
  linarith

/-- **The exact endpoint (`c:nonident`).** When the two data laws coincide (`P = Q` — two truths
agreeing on the resolved subspace, so identical in law, `NonIdentifiability.data_marginal_eq_of_resolved_eq`),
every test has total error at least `1`: it can do no better than guessing. This is the `ε = 0` case. -/
theorem lecam_exact (hPQ : P = Q) {A : Set Ω} (hA : MeasurableSet A) :
    1 ≤ P.real A + Q.real Aᶜ := by
  have h := lecam_two_point (P := P) (Q := Q) (ε := 0)
    (fun B _ => by rw [hPQ]; simp) hA
  linarith

end NearNull

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms NearNull.lecam_two_point
#print axioms NearNull.lecam_minimax
#print axioms NearNull.lecam_exact
