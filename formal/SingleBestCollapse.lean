import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.Kernel.Composition.MeasureCompProd
import Mathlib.Probability.Kernel.Composition.Prod
import Mathlib.MeasureTheory.Measure.Prod
import Mathlib.MeasureTheory.Measure.Typeclasses.NoAtoms

/-!
# Theorem `p:map` of the paper: the single-best collapse (linear-algebra core)

This file machine-checks the **linear-algebra core** of Theorem `p:map`
("Single-best curation collapses the blind-fiber uncertainty") of the paper
*Priors learned from legacy reconstructions inherit undetectable overconfidence*, together with the elementary deterministic-collapse statement and the
measure-theoretic zero-coverage mechanism.

The paper's `p:map` has three layers.  We formalize each at the level the task asks for:

## Part 1 — deterministic collapse (elementary, fully proved: `DeterministicCollapse`)

For a forward map `F` that is constant on the fibers `x + N` of a blind subspace `N`
(equivalently, in split coordinates, `F` reads only the resolved coordinate), the data term is
flat on each fiber, so:

* `blind_minimizes_penalty_on_fiber` — **every** global minimizer of `D(y, F ·) + R(·)` has a blind
  component that minimizes the penalty `R` over its own fiber (this is the paper's
  "comparing candidates within one fiber cancels the data term").
* `blind_minimizes_penalty_on_fiber_coset` — the same, stated in the paper's coset form
  `F(x + n) = F(x)`, `n ∈ N`.
* `linear_blind_invariance` — for a **linear** operator `A` with `N = ker A`, the hypothesis
  `F(x+n)=F(x)` is automatic (this is why the paper never states it separately).
* `profiled_is_global_min` — profiling nests the two `argmin`s: pairing the resolved value-function
  minimizer with the per-fiber penalty selection gives a global minimizer.
* `blind_is_function_of_resolved` / `collapse_zero_within_fiber` — under a profiled selection
  `x_B⋆(x_R)` chosen as a function of the resolved coordinate alone, the reconstruction's blind part
  is a function of its resolved part; hence two measurements with the same resolved reconstruction
  give the *same* blind reconstruction: `C^sb = 0` (zero within-fiber width).

## Part 2 — Gaussian sharpening (pure linear algebra, fully proved: `GaussianSharpening`)

With posterior gain `G = Σ_post Aᵀ Γ⁻¹` and posterior precision `Σ_post⁻¹ = Σ_reg⁻¹ + Aᵀ Γ⁻¹ A`:

* `gain_range_meets_ker_zero` — `range G ∩ ker A = {0}` (the appendix's "range meets ker A only at
  0" argument, verbatim: `Σ_post w ∈ ker A`, `w ∈ row A` ⟹ `Σ_post w = Σ_reg w` is blind and
  orthogonal to itself under `Σ_reg ≻ 0`, hence `0`).  This is exactly "the blind-conditional
  variance is zero".
* `rank_gain_eq_rank` — `rank G = rank A`.
* `rank_archiveCov` — the archive covariance `Σ_sb = G S_y Gᵀ` (with `S_y ≻ 0`) has rank exactly
  `rank A`.

## Part 3 — zero coverage (measure-theoretic, fully proved: `Coverage`)

* `coverage_zero` — the **fiber-conditional** `C^sb = 0`: at a fixed resolved coordinate, a report
  that is a measurable function of the measurement never equals a truth whose law is nonatomic and
  independent of the report: `P {ω | θ ω = report ω} = 0`.
* `marginal_coverage_zero` — the **marginal** `C^sb = 𝔼_y[ Pr(θ_⋆ = c(y) ∣ x_R) ] = 0`, the full
  statement: with the truth disintegrated along the resolved coordinate as `π ⊗ₘ (ρ ×ₖ L)` (resolved
  marginal `π`; nonatomic blind fiber-conditional `ρ`; likelihood `L` reading the resolved coordinate
  alone; `θ ⟂ y ∣ x_R` via the product kernel `ρ ×ₖ L`), the tower property gives
  `(π ⊗ₘ (ρ ×ₖ L)) {(x_R,(θ,y)) | θ = c y} = 0` for every measurable reconstruction rule `c`.

## Honesty

No `sorry`/`admit`.  Every unproved fact is an explicit hypothesis.  `#print axioms` on each main
theorem lists only `propext`, `Classical.choice`, `Quot.sound` (no `sorryAx`); two of them
(`blind_is_function_of_resolved`, `collapse_zero_within_fiber`) depend on no axioms at all.  The
hypothesis-reconciliation audit (what each Lean statement needs vs. what `p:map` states) is the
`Reconciliation` block at the end of the file.
-/

open Matrix Module MeasureTheory ProbabilityTheory
open scoped Matrix ENNReal

namespace SingleBestCollapse

/-! ## Part 2 — the Gaussian sharpening (pure linear algebra) -/

section GaussianSharpening

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- `star` is trivial on real vectors: for `v : ι → ℝ`, `star v = v`. -/
private lemma star_real_vec {ι : Type*} (v : ι → ℝ) : star v = v := by
  funext i; simp [Pi.star_apply, star_trivial]

/-- **The appendix's core algebraic identity (the "range meets `ker A` only at `0`" argument).**

If `w = Aᵀ p` lies in `row A = range Aᵀ` and `Σ_post w ∈ ker A`, then `Σ_post w = 0`.

Proof (verbatim from `app:proofs`): from `Σ_post w ∈ ker A` we get `Aᵀ Γ⁻¹ A (Σ_post w) = 0`, and the
posterior precision `Σ_post⁻¹ = Σ_reg⁻¹ + Aᵀ Γ⁻¹ A` then gives `Σ_reg⁻¹ (Σ_post w) = w`, i.e.
`Σ_post w = Σ_reg w`.  This vector is simultaneously blind (`∈ ker A`) and, tested against
`w ∈ (ker A)⊥`, has `⟪w, Σ_reg w⟫ = 0`; positive-definiteness of `Σ_reg` forces `w = 0`. -/
theorem posterior_blind_zero
    (A : Matrix m n ℝ) (Γ : Matrix m m ℝ) (Sreg Spost : Matrix n n ℝ)
    (hSreg : Sreg.PosDef) (hSpost : Spost.PosDef)
    (hrel : Spost⁻¹ = Sreg⁻¹ + Aᵀ * Γ⁻¹ * A)
    {w : n → ℝ} (p : m → ℝ) (hw : w = Aᵀ *ᵥ p)
    (hker : A *ᵥ (Spost *ᵥ w) = 0) :
    Spost *ᵥ w = 0 := by
  have hSpostU : IsUnit Spost.det := (Matrix.isUnit_iff_isUnit_det _).mp hSpost.isUnit
  have hSregU : IsUnit Sreg.det := (Matrix.isUnit_iff_isUnit_det _).mp hSreg.isUnit
  set z := Spost *ᵥ w with hz
  -- `Aᵀ Γ⁻¹ A` annihilates `z` because `A z = 0`.
  have hMz : (Aᵀ * Γ⁻¹ * A) *ᵥ z = 0 := by
    rw [← mulVec_mulVec, ← mulVec_mulVec, hker, mulVec_zero, mulVec_zero]
  -- `Σ_post⁻¹ z = w` (definition of inverse) and, via `hrel`, `Σ_post⁻¹ z = Σ_reg⁻¹ z`.
  have hinv : Spost⁻¹ *ᵥ z = w := by
    rw [hz, mulVec_mulVec, nonsing_inv_mul _ hSpostU, one_mulVec]
  have hw2 : w = Sreg⁻¹ *ᵥ z := by
    rw [← hinv, hrel, add_mulVec, hMz, add_zero]
  -- Hence `z = Σ_reg w`.
  have hz_eq : z = Sreg *ᵥ w := by
    rw [hw2, mulVec_mulVec, mul_nonsing_inv _ hSregU, one_mulVec]
  -- `⟪w, z⟫ = ⟪Aᵀ p, z⟫ = ⟪p, A z⟫ = 0` since `z ∈ ker A`.
  have hwz : w ⬝ᵥ z = 0 := by
    rw [hw, mulVec_transpose, ← dotProduct_mulVec, hker, dotProduct_zero]
  -- So `⟪w, Σ_reg w⟫ = 0`, and `Σ_reg ≻ 0` gives `w = 0`, whence `z = 0`.
  rw [hz_eq] at hwz
  have hw0 : w = 0 := by
    by_contra hne
    have hpos := hSreg.dotProduct_mulVec_pos hne
    rw [star_real_vec] at hpos
    linarith
  rw [hz_eq, hw0, mulVec_zero]

/-- The **posterior gain** `G = Σ_post Aᵀ Γ⁻¹` (the affine map `MAP_reg(y) = G y + (I - GA) μ_reg`
of the Gaussian sharpening). -/
noncomputable def gain (A : Matrix m n ℝ) (Γ : Matrix m m ℝ) (Spost : Matrix n n ℝ) :
    Matrix n m ℝ := Spost * Aᵀ * Γ⁻¹

/-- **`range G ∩ ker A = {0}` (the blind-conditional variance is zero).**

Any element `G u` of the gain's range that lies in the blind subspace `ker A` is `0`.  Because
`range G = Σ_post (row A)`, this is `posterior_blind_zero` applied to `w = Aᵀ (Γ⁻¹ u) ∈ row A`. -/
theorem gain_range_meets_ker_zero
    (A : Matrix m n ℝ) (Γ : Matrix m m ℝ) (Sreg Spost : Matrix n n ℝ)
    (hSreg : Sreg.PosDef) (hSpost : Spost.PosDef)
    (hrel : Spost⁻¹ = Sreg⁻¹ + Aᵀ * Γ⁻¹ * A)
    {u : m → ℝ} (hu : A *ᵥ (gain A Γ Spost *ᵥ u) = 0) :
    gain A Γ Spost *ᵥ u = 0 := by
  -- `G u = Σ_post w` with `w = Aᵀ (Γ⁻¹ u) ∈ range Aᵀ`.
  have hGu : gain A Γ Spost *ᵥ u = Spost *ᵥ (Aᵀ *ᵥ (Γ⁻¹ *ᵥ u)) := by
    rw [gain, ← mulVec_mulVec, ← mulVec_mulVec]
  rw [hGu] at hu ⊢
  exact posterior_blind_zero A Γ Sreg Spost hSreg hSpost hrel (Γ⁻¹ *ᵥ u) rfl hu

/-- Two matrices with the same number of columns and equal `mulVecLin`-kernels have equal rank
(rank–nullity on the common domain). -/
private lemma rank_eq_of_ker_mulVecLin_eq {a b c : Type*} [Fintype a] [Fintype b] [Fintype c]
    (P : Matrix a c ℝ) (Q : Matrix b c ℝ)
    (h : LinearMap.ker P.mulVecLin = LinearMap.ker Q.mulVecLin) :
    P.rank = Q.rank := by
  have hP := LinearMap.finrank_range_add_finrank_ker P.mulVecLin
  have hQ := LinearMap.finrank_range_add_finrank_ker Q.mulVecLin
  have hk : finrank ℝ (LinearMap.ker P.mulVecLin) = finrank ℝ (LinearMap.ker Q.mulVecLin) := by
    rw [h]
  show finrank ℝ (LinearMap.range P.mulVecLin) = finrank ℝ (LinearMap.range Q.mulVecLin)
  omega

/-- **`rank G = rank A`.**  `G = Σ_post Aᵀ Γ⁻¹` differs from `Aᵀ` by invertible factors
(`Σ_post ≻ 0`, `Γ ≻ 0`), which do not change rank; and `rank Aᵀ = rank A`. -/
theorem rank_gain_eq_rank
    (A : Matrix m n ℝ) (Γ : Matrix m m ℝ) (Spost : Matrix n n ℝ)
    (hΓ : Γ.PosDef) (hSpost : Spost.PosDef) :
    (gain A Γ Spost).rank = A.rank := by
  have hΓinv : IsUnit (Γ⁻¹).det := (Matrix.isUnit_iff_isUnit_det _).mp hΓ.inv.isUnit
  have hSp : IsUnit Spost.det := (Matrix.isUnit_iff_isUnit_det _).mp hSpost.isUnit
  rw [gain, rank_mul_eq_left_of_isUnit_det _ _ hΓinv,
      rank_mul_eq_right_of_isUnit_det _ _ hSp, rank_transpose]

omit [DecidableEq m] [DecidableEq n] in
/-- For `S ≻ 0`, the Gram-type matrix `G S Gᵀ` has the same `mulVecLin`-kernel as `Gᵀ`:
`G S Gᵀ x = 0` ⟺ `Gᵀ x = 0`, because `⟪Gᵀx, S Gᵀx⟫ = 0` forces `Gᵀ x = 0` when `S ≻ 0`. -/
private lemma ker_mulVecLin_gram
    (G : Matrix n m ℝ) {S : Matrix m m ℝ} (hS : S.PosDef) :
    LinearMap.ker ((G * S * Gᵀ).mulVecLin) = LinearMap.ker (Gᵀ.mulVecLin) := by
  have hexpand : ∀ x : n → ℝ, (G * S * Gᵀ) *ᵥ x = G *ᵥ (S *ᵥ (Gᵀ *ᵥ x)) := by
    intro x; rw [← mulVec_mulVec, ← mulVec_mulVec]
  ext x
  simp only [LinearMap.mem_ker, Matrix.mulVecLin_apply]
  constructor
  · intro hx
    rw [hexpand] at hx
    -- dot with `x`: `⟪x, G (S (Gᵀ x))⟫ = ⟪Gᵀ x, S (Gᵀ x)⟫ = 0`.
    have hdot : (Gᵀ *ᵥ x) ⬝ᵥ (S *ᵥ (Gᵀ *ᵥ x)) = 0 := by
      have h0 : x ⬝ᵥ (G *ᵥ (S *ᵥ (Gᵀ *ᵥ x))) = 0 := by rw [hx, dotProduct_zero]
      rwa [dotProduct_mulVec, ← mulVec_transpose] at h0
    by_contra hne
    have hpos := hS.dotProduct_mulVec_pos hne
    rw [star_real_vec] at hpos
    linarith
  · intro hx
    rw [hexpand, hx, mulVec_zero, mulVec_zero]

/-- **The archive covariance `Σ_sb = G S_y Gᵀ` has rank exactly `rank A`** (with `S_y ≻ 0`).
Combines `ker_mulVecLin_gram` (so `rank(G S_y Gᵀ) = rank Gᵀ`) with `rank Gᵀ = rank G = rank A`. -/
theorem rank_archiveCov
    (A : Matrix m n ℝ) (Γ : Matrix m m ℝ) (Spost : Matrix n n ℝ) (Sy : Matrix m m ℝ)
    (hΓ : Γ.PosDef) (hSpost : Spost.PosDef) (hSy : Sy.PosDef) :
    (gain A Γ Spost * Sy * (gain A Γ Spost)ᵀ).rank = A.rank := by
  rw [rank_eq_of_ker_mulVecLin_eq _ _ (ker_mulVecLin_gram (gain A Γ Spost) hSy),
      rank_transpose, rank_gain_eq_rank A Γ Spost hΓ hSpost]

end GaussianSharpening

/-! ## Part 1 — the deterministic collapse (elementary) -/

section DeterministicCollapse

/-!
Split-coordinate model: `Rc` are the resolved coordinates, `Bc` the blind coordinates, `Y` the
measurements, `W` the data space.  The forward map `F : Rc → W` reads only the resolved coordinate
(this *is* the invariance `F = F ∘ P_{N⊥}`); the objective at measurement `y` is
`J y (r, b) = D y (F r) + R r b`, a data-fidelity term plus a penalty. -/

variable {Y W Rc Bc : Type*} (F : Rc → W) (D : Y → W → ℝ) (R : Rc → Bc → ℝ)

/-- **Fiber optimality (the collapse's engine).** Any global minimizer `(r, b)` of the objective
`J y = D y (F ·) + R` has a blind component that minimizes the penalty over its own fiber:
`R r b ≤ R r b'` for every `b'`.  The data term, constant on the fiber `{r} × Bc`, cancels. -/
theorem blind_minimizes_penalty_on_fiber
    {y : Y} {r : Rc} {b : Bc}
    (hmin : ∀ r' b', D y (F r) + R r b ≤ D y (F r') + R r' b') :
    ∀ b', R r b ≤ R r b' :=
  fun b' => by have := hmin r b'; linarith

/-- **Profiling nests the two `argmin`s.** If `xB r` minimizes `R r ·` for each `r`, and `rHat`
minimizes the profiled value function `r ↦ D y (F r) + R r (xB r)`, then `(rHat, xB rHat)` is a
global minimizer of `J y`.  (Justifies that the profiled selection is WLOG a genuine reconstruction.) -/
theorem profiled_is_global_min
    (xB : Rc → Bc) (hxB : ∀ r b, R r (xB r) ≤ R r b)
    {y : Y} {rHat : Rc}
    (hres : ∀ r, D y (F rHat) + R rHat (xB rHat) ≤ D y (F r) + R r (xB r)) :
    ∀ r b, D y (F rHat) + R rHat (xB rHat) ≤ D y (F r) + R r b := by
  intro r b
  calc D y (F rHat) + R rHat (xB rHat)
      ≤ D y (F r) + R r (xB r) := hres r
    _ ≤ D y (F r) + R r b := by have := hxB r b; linarith

/-- **The single-best reconstruction under a profiled selection.**  `rHat y` is the resolved
reconstruction; `xB` is the profiled blind selection (chosen as a function of the resolved
coordinate alone).  The reconstruction is `recon y = (rHat y, xB (rHat y))`. -/
def recon (xB : Rc → Bc) (rHat : Y → Rc) : Y → Rc × Bc :=
  fun y => (rHat y, xB (rHat y))

/-- **The blind component is a deterministic function of the resolved component.**
There is a map `φ` (namely the profiled selection `xB`) with `(recon y).blind = φ (recon y).resolved`
for every measurement `y`. -/
theorem blind_is_function_of_resolved (xB : Rc → Bc) (rHat : Y → Rc) :
    ∃ φ : Rc → Bc, ∀ y, (recon xB rHat y).2 = φ ((recon xB rHat y).1) :=
  ⟨xB, fun _ => rfl⟩

/-- **Zero within-fiber width (`C^sb = 0`).** Two measurements yielding the same resolved
reconstruction yield the *same* blind reconstruction: the curated blind law, conditional on the
resolved readout, is a single point. -/
theorem collapse_zero_within_fiber (xB : Rc → Bc) (rHat : Y → Rc) {y y' : Y}
    (h : (recon xB rHat y).1 = (recon xB rHat y').1) :
    (recon xB rHat y).2 = (recon xB rHat y').2 :=
  congrArg xB h

end DeterministicCollapse

section CosetForm

/-!
The paper states the invariance in coset form `F(x + n) = F(x)`, `n ∈ N`, with the *leaf* `x + N`.
Here is that faithful statement and the fact that, for a **linear** operator `A` with `N = ker A`,
the hypothesis is automatic — which is why `p:map` states it only as `F(x+n)=F(x)`. -/

variable {X W Y : Type*} [AddCommGroup X] [Module ℝ X] [AddCommGroup W] [Module ℝ W]
  (F : X → W) (D : Y → W → ℝ) (R : X → ℝ) (N : Submodule ℝ X)

omit [AddCommGroup W] [Module ℝ W] in
/-- **Fiber optimality, coset form.** If the data term is flat on the leaf `x̂ + N`
(`F (x̂ + n) = F x̂` for `n ∈ N`) and `x̂` globally minimizes `D y (F ·) + R`, then `x̂` minimizes the
penalty `R` over its whole fiber: `R x̂ ≤ R (x̂ + n)` for all `n ∈ N`. -/
theorem blind_minimizes_penalty_on_fiber_coset
    (hF : ∀ (x : X) (n : X), n ∈ N → F (x + n) = F x)
    {y : Y} {xhat : X} (hmin : ∀ x, D y (F xhat) + R xhat ≤ D y (F x) + R x) :
    ∀ n ∈ N, R xhat ≤ R (xhat + n) := by
  intro n hn
  have h := hmin (xhat + n)
  rw [hF xhat n hn] at h
  linarith

/-- **The invariance is automatic for a linear operator with `N = ker A`.**
`A (x + n) = A x` whenever `n ∈ ker A`. -/
theorem linear_blind_invariance (A : X →ₗ[ℝ] W) :
    ∀ (x : X) (n : X), n ∈ LinearMap.ker A → A (x + n) = A x := by
  intro x n hn
  rw [map_add, LinearMap.mem_ker.mp hn, add_zero]

end CosetForm

/-! ## Part 3 — the zero-coverage mechanism (measure-theoretic core) -/

section Coverage

/-!
The paper's zero-coverage argument, conditional on the resolved coordinate `x_R`: given `x_R`, the
measurement `y` and the truth's blind functional `θ_⋆ = v⊤ x` are **conditionally independent**,
while the archive's report is a function of `y` alone; if the truth's fiber-conditional law of `θ`
is **nonatomic**, the report (a single point per `y`) covers `θ_⋆` with probability zero.

We build this in two layers, both fully proved:

* `prod_coverage_zero` / `coverage_zero` — the **fiber-conditional** `C^sb = 0`: at a fixed resolved
  coordinate the conditional independence is an actual product law, so a nonatomic truth `θ` is never
  hit by a measurable report.
* `marginal_coverage_zero` — the **marginal** `C^sb = 𝔼_y[ Pr(θ_⋆ = c(y) ∣ x_R) ] = 0`, obtained by
  the tower property (`Measure.compProd_apply`).  The truth is modelled generatively, exactly as the
  paper's: resolved marginal `π`, a blind fiber-conditional kernel `ρ` (nonatomic for every resolved
  coordinate), and a likelihood kernel `L` reading the resolved coordinate alone — with `θ ~ ρ` and
  the measurement `y ~ L` conditionally independent given `x_R`, encoded by the product kernel
  `ρ ×ₖ L`.  For **every** measurable reconstruction rule `c`, the report never equals the truth. -/

/-- **Core.** For a product law of a nonatomic truth `μ` (the blind functional) and any measurement
law `ν`, with a measurable report rule `c`, the report never equals the truth:
`(μ × ν) {(t, y) | t = c y} = 0`.  (Peel the `ν`-coordinate; each `t`-section is the singleton
`{c y}`, null because `μ` is nonatomic.) -/
theorem prod_coverage_zero {𝓨 : Type*} [MeasurableSpace 𝓨]
    (μ : Measure ℝ) (ν : Measure 𝓨) [SFinite μ] [SFinite ν] [NoAtoms μ]
    {c : 𝓨 → ℝ} (hc : Measurable c) :
    (μ.prod ν) {q : ℝ × 𝓨 | q.1 = c q.2} = 0 := by
  have hmeas : MeasurableSet {q : ℝ × 𝓨 | q.1 = c q.2} := by
    have he : {q : ℝ × 𝓨 | q.1 = c q.2}
        = (fun q => (q.1, c q.2)) ⁻¹' {p : ℝ × ℝ | p.1 = p.2} := by ext q; simp
    rw [he]
    exact ((isClosed_eq continuous_fst continuous_snd).measurableSet).preimage
      (measurable_fst.prodMk (hc.comp measurable_snd))
  rw [Measure.prod_apply_symm hmeas]
  have hsec : ∀ y : 𝓨, μ ((fun x => (x, y)) ⁻¹' {q : ℝ × 𝓨 | q.1 = c q.2}) = 0 := by
    intro y
    have hs : (fun x => (x, y)) ⁻¹' {q : ℝ × 𝓨 | q.1 = c q.2} = {c y} := by
      ext x; simp only [Set.mem_preimage, Set.mem_setOf_eq, Set.mem_singleton_iff]
    rw [hs, measure_singleton]
  simp only [hsec, lintegral_zero]

/-- **Zero coverage (fiber-conditional).** If the truth's blind functional `θ` and the archive's
report are independent, `θ` and `report` are measurable, and `θ`'s law is nonatomic, then the report
never equals the truth: `P {ω | θ ω = report ω} = 0`.

A deterministic-per-measurement report cannot cover a truth that genuinely varies along the fiber. -/
theorem coverage_zero
    {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
    {θ report : Ω → ℝ} (hθ : Measurable θ) (hrep : Measurable report)
    (hindep : IndepFun θ report P) [NoAtoms (P.map θ)] :
    P {ω | θ ω = report ω} = 0 := by
  haveI : IsProbabilityMeasure (P.map θ) := Measure.isProbabilityMeasure_map hθ.aemeasurable
  haveI : IsProbabilityMeasure (P.map report) := Measure.isProbabilityMeasure_map hrep.aemeasurable
  have hdiag : MeasurableSet {q : ℝ × ℝ | q.1 = q.2} :=
    (isClosed_eq continuous_fst continuous_snd).measurableSet
  -- The coverage event is the pullback of the diagonal under `(θ, report)`; independence turns the
  -- pushforward into the product law, and `prod_coverage_zero` (with `c = id`) finishes.
  have hset : {ω | θ ω = report ω}
      = (fun ω => (θ ω, report ω)) ⁻¹' {q : ℝ × ℝ | q.1 = q.2} := by
    ext ω; simp only [Set.mem_setOf_eq, Set.mem_preimage]
  rw [hset, ← Measure.map_apply (hθ.prodMk hrep) hdiag,
      hindep.map_prod_eq_prod_map_map hθ.aemeasurable hrep.aemeasurable]
  simpa using prod_coverage_zero (P.map θ) (P.map report) measurable_id

/-- **Marginal zero coverage `C^sb = 0` (the full tower-property statement).**

Model the truth generatively (its disintegration along the resolved coordinate): a resolved marginal
`π`, a blind fiber-conditional `ρ` that is **nonatomic at every resolved coordinate**, and a
likelihood `L` reading the resolved coordinate alone.  Conditional independence of the blind
functional `θ ~ ρ` and the measurement `y ~ L` given `x_R` is encoded by the product kernel `ρ ×ₖ L`,
so the joint truth law is `π ⊗ₘ (ρ ×ₖ L)`.  Then for **every** measurable reconstruction rule `c`
(the report `c(y)` reads the measurement alone), the archive's report never equals the truth:
`(π ⊗ₘ (ρ ×ₖ L)) {(x_R, (θ, y)) | θ = c y} = 0`.

Proof: `Measure.compProd_apply` is the tower property `C^sb = ∫ Pr(· ∣ x_R) dπ`; the inner
per-`x_R` term is `prod_coverage_zero` (the product kernel gives an actual product law at each `x_R`,
and `ρ x_R` is nonatomic), so the integrand is `0` and the average is `0`. -/
theorem marginal_coverage_zero {R 𝓨 : Type*} [MeasurableSpace R] [MeasurableSpace 𝓨]
    (π : Measure R) [IsProbabilityMeasure π]
    (ρ : Kernel R ℝ) (L : Kernel R 𝓨) [IsMarkovKernel ρ] [IsMarkovKernel L]
    (hρ : ∀ r, NoAtoms (ρ r)) {c : 𝓨 → ℝ} (hc : Measurable c) :
    (π ⊗ₘ (ρ ×ₖ L)) {p : R × (ℝ × 𝓨) | p.2.1 = c p.2.2} = 0 := by
  have hmeas : MeasurableSet {p : R × (ℝ × 𝓨) | p.2.1 = c p.2.2} := by
    have he : {p : R × (ℝ × 𝓨) | p.2.1 = c p.2.2}
        = (fun p => (p.2.1, c p.2.2)) ⁻¹' {q : ℝ × ℝ | q.1 = q.2} := by ext p; simp
    rw [he]
    exact ((isClosed_eq continuous_fst continuous_snd).measurableSet).preimage
      ((measurable_fst.comp measurable_snd).prodMk (hc.comp (measurable_snd.comp measurable_snd)))
  rw [Measure.compProd_apply hmeas]
  -- Tower property: the coverage is the `π`-average of the per-`x_R` (fiber-conditional) coverage.
  have hfib : ∀ r, (ρ ×ₖ L) r (Prod.mk r ⁻¹' {p : R × (ℝ × 𝓨) | p.2.1 = c p.2.2}) = 0 := by
    intro r
    haveI := hρ r
    have hpre : Prod.mk r ⁻¹' {p : R × (ℝ × 𝓨) | p.2.1 = c p.2.2}
        = {q : ℝ × 𝓨 | q.1 = c q.2} := by ext q; simp
    rw [hpre, Kernel.prod_apply]
    exact prod_coverage_zero (ρ r) (L r) hc
  simp only [hfib, lintegral_zero]

end Coverage

/-!
## Reconciliation — what the formalization needs vs. what `p:map` states

For each hypothesis of the Lean statements: **(a)** already in `p:map`; **(b)** load-bearing but
missing from the text (with a proposed minimal edit); **(c)** automatic in the paper's setting.
Every hypothesis here is either (a) or (c) — the formalization surfaced *no gap that makes `p:map`
wrong*, but it did sharpen exactly which hypothesis does which work.

### Part 2 (Gaussian sharpening) — the precise dependency the prose blurs

The appendix carries `Σ_reg, Σ_post, Γ, S_y` all as "≻ 0" together.  The machine proof separates
their roles cleanly, which is the main audit finding:

* `hrel : Σ_post⁻¹ = Σ_reg⁻¹ + Aᵀ Γ⁻¹ A` — **(a)**, verbatim the paper's posterior precision.
* `w = Aᵀ p` (i.e. `w ∈ row A`) — **(a)**, the paper's "with `w ∈ row A`".
* `Σ_reg ≻ 0` — **(c)/(a)**: it is the *only* positive-definiteness the blind-variance-collapse
  (`posterior_blind_zero`, `gain_range_meets_ker_zero`) actually uses — precisely at the paper's
  step "`wᵀ Σ_reg w = 0`, so `w = 0`".  A covariance, so assumed nondegenerate; the audit note is
  that this lone `≻ 0` (not the others) is what powers `range G ∩ ker A = {0}`.
* `Σ_post ≻ 0` — **(c)**: the collapse and rank proofs use Σ_post *only through invertibility*
  (`Σ_post⁻¹ Σ_post = I`).  And Σ_post ≻ 0 is anyway *derivable* from `hrel` given `Σ_reg ≻ 0`,
  `Γ ≻ 0` — so the paper's stated "Σ_post ≻ 0" is a consequence, not an independent load-bearing
  assumption.  (The Lean file takes it as a hypothesis but consumes only its invertibility.)
* `Γ ≻ 0` — **(c)**: not needed at all for the variance collapse (there `Γ⁻¹` may be *any* matrix —
  `A z = 0` annihilates `Aᵀ Γ⁻¹ A z` regardless); it is used *only* in `rank_gain_eq_rank`
  (`rank(Aᵀ Γ⁻¹) = rank Aᵀ`).  So Γ's regularity is a rank-claim hypothesis, not a collapse one.
* `S_y ≻ 0` — **(a)**: the paper's "`S_y ≻ 0`", used genuinely (`zᵀ S_y z = 0 ⇒ z = 0`) in the
  rank of `Σ_sb = G S_y Gᵀ`.

No edit to `p:map` is *required*; the optional sharpening the formalization justifies is to say
"`Σ_post` invertible (equivalently, `Σ_reg ≻ 0`, `Γ ≻ 0`)" and to attach `Γ ≻ 0` to the rank claim
rather than the collapse.

### Part 1 (deterministic collapse) — the generality claim, verified

* additive objective `D(y, F x) + R x` with `D` reaching `x` only through `F x` — **(a)**, eq:varest.
* `F` constant on fibers (`F(x+n)=F(x)`, split-coordinate `F` reads only `x_R`) — **(a)**.
* profiled selection `x_B⋆` is a function of the resolved coordinate — **(a)**, the paper's
  "profiled measurable selection chosen as a function of the fiber alone".

The proofs use **no** linearity, Gaussianity, convexity, or alignment — confirming the paper's
"needs neither Gaussianity nor an aligned split", "every penalty `R`, proper or improper".
`linear_blind_invariance` discharges `F(x+n)=F(x)` for linear `A`, `N = ker A` — **(c)**, confirming
the paper never needs to state it.  A genuine (harmless) surfacing: the *determinism itself* is
definitional once the selection is a function of `x_R`; the mathematical content is fiber-optimality
(`blind_minimizes_penalty_on_fiber`) plus the `argmin`-nesting (`profiled_is_global_min`).

### Part 3 (zero coverage) — both the fiber-conditional and the marginal statement

Fiber-conditional (`coverage_zero`):
* `IndepFun θ report` — the paper's conditional independence given `x_R`, taken fiber-conditionally.
* `NoAtoms (P.map θ)` — **(a)**, "fiber-conditional law of `θ` is nonatomic".
* `report` measurable — **(a)**, "for every measurable reconstruction rule `c`".

Marginal (`marginal_coverage_zero`, the full `C^sb = 0`) needs a *model of the truth*, and this is
where the formalization asks for something the paper leaves informal:
* The truth is taken as `π ⊗ₘ (ρ ×ₖ L)` — resolved marginal `π`, blind fiber-conditional kernel `ρ`,
  likelihood kernel `L`.  This **is** the paper's disintegration of the truth along `x_R`; **(c)**,
  a faithful model, not an extra assumption (a standard-Borel truth always disintegrates so).
* `θ ⟂ y ∣ x_R` is encoded structurally by the **product kernel** `ρ ×ₖ L` — **(a)**, exactly the
  paper's "given the resolved coordinates the measurement carries no information about the blind
  ones — `y` and `x_B` are conditionally independent given `x_R`".
* `∀ r, NoAtoms (ρ r)` — **(b, mild)**: the paper says the fiber conditional is nonatomic (and notes
  "an absolutely continuous fiber conditional suffices … for every `v ≠ 0`").  The Lean statement
  needs this **at every** resolved coordinate (not merely `π`-a.e.); a `π`-a.e. version would
  suffice and is the honest minimal hypothesis.  Proposed text note: state nonatomicity of the blind
  fiber conditional for `π`-a.e. `x_R`.
* `c` measurable — **(a)**, "for every measurable reconstruction rule `c` — the variational
  structure decides only where the point sits".

No hidden fact: `marginal_coverage_zero` is the paper's tower-property conclusion in full, obtained
from `Measure.compProd_apply` (the tower) and the per-fiber `prod_coverage_zero`.
-/

end SingleBestCollapse

/-! ### Axiom audit

The doc comment above claims these rest only on Lean's three standard axioms. These commands
are what make that checkable rather than asserted: each must print `propext, Classical.choice,
Quot.sound` and never `sorryAx`. -/

#print axioms SingleBestCollapse.coverage_zero
#print axioms SingleBestCollapse.marginal_coverage_zero
#print axioms SingleBestCollapse.linear_blind_invariance
