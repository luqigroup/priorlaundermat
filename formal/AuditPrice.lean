import Mathlib

/-!
# The price of an audit (Proposition `p:audit`): exact, non-asymptotic content

Machine-checked formalization of the exact content of the paper's Proposition *"The price of an
audit, in ground truth"* (Gaussian fibers, noiseless references).

## Mapping to the paper

The paper works in the blind fiber: `N` spans the blind subspace of dimension `b`, the curated
prior supplies the blind-fiber conditional mean `m_ρ(x_R)` and conditional covariance `Σ_{B|R}`,
and `k` noiseless references supply the truth exactly. We work directly in the `b`-dimensional
blind coordinates `ℝᵇ = EuclideanSpace ℝ (Fin b)` (the paper's `Nᵀx` are exactly these
coordinates), so a fiber residual is a vector of `EuclideanSpace ℝ (Fin b)` and the references
give `k` iid draws.

* **The χ² law** (route (b) of the audit plan): `chiSq n` is DEFINED as the pushforward of the
  `n`-fold product of standard 1-D Gaussians under `x ↦ ∑ xᵢ²`; the theorem
  `stdGaussian_normSq_law` identifies it with the pushforward of mathlib's `stdGaussian` on
  `EuclideanSpace ℝ (Fin n)` under `x ↦ ‖x‖²` — "a sum of independent squares of standard
  normals", which is the definition the paper's proof uses.  The identification of `chiSq n`
  with the Gamma(n/2, rate 1/2) measure (route (a)) is deliberately NOT formalized here.
* **The statistic**: `auditStat u = ∑ᵢ ‖uᵢ‖²` is the paper's `T = ∑ᵢ ‖uᵢ‖²`.
* **Standardization** (the paper's "residuals standardized by prior-supplied quantities"):
  `multivariateGaussian_standardize` — if `x ~ N(m, Σ)` (mathlib's `multivariateGaussian m S`,
  whose square root is `CFC.sqrt S`) and the prior supplies a left inverse `W` of `√Σ`
  (hypothesis `hW : W * CFC.sqrt S = 1`, the paper's `Σ_{B|R}^{-1/2}` — a *model input*, since the
  prior supplies `Σ_{B|R}` exactly), then `u = W(x − m) ~ N(0, I_b)`.
  `multivariateGaussian_standardize_inv` instantiates `W := (CFC.sqrt S)⁻¹` verbatim.
* **The null** ("if the prior's blind report is correct then `uᵢ ~ N(0,I_b)` independently, so
  `T ~ χ²_{kb}`"): `auditStat_null_law` (standardized residuals as the model input) and
  `audit_null_pipeline` (the full pipeline from `k` iid Gaussian fiber residuals through the
  standardization map).  Note mathlib's `multivariateGaussian_zero_one : N(0,1) = stdGaussian`.
* **The alternative** ("a misreport inflating the blind spread by factor `r` throughout the
  subspace"; the paper's proof step "inflating the covariance to `r²Σ` scales each residual by
  `r`, giving `T ~ r²χ²_{kb}`"): `sqrt_sq_smul` (`√(r²Σ) = r√Σ`, the CFC lemma mathlib lacks,
  built here), `multivariateGaussian_isotropic` (`N(0, r²I) = (r •)⋆ stdGaussian`),
  `auditStat_alt_law` and `audit_alt_pipeline`: the law of `T` is the pushforward of
  `chiSq (k*b)` under `t ↦ r²t`.
* **Power** ("the size-α test rejecting at `χ²_{kb,1−α}` has power
  `1−β = Pr(χ²_{kb} > χ²_{kb,1−α}/r²)`"): stated against an ARBITRARY threshold `q` (which
  carries the full mathematical content; the quantile is a name for the `q` with
  `chiSq (k*b) (Ioi q) = α`, supplied as hypothesis `hα` where wanted):
  `audit_power` / `audit_power_pipeline`: `P_alt(T > q) = chiSq (k*b) (Ioi (q / r²))`.
* **Level over the composite null** ("rejection probability increases in the inflation, so the
  test is level α over the composite null"): `audit_rejection_mono` (monotone in `r`),
  `audit_size` (at `r = 1` the rejection probability is exactly `chiSq (k*b) (Ioi q)`), and
  `audit_level` / `audit_level_alpha` (for every `0 < r ≤ 1` the rejection probability is at
  most the size).

## Deliberately not formalized

The display `k ≈ 1 + (z_{1−α}+z_{1−β})²/(2b log² r)` is an ASYMPTOTIC (normal-approximation)
sample-size heuristic; it is out of scope for machine-checking and is intentionally absent.

## Named hypotheses (model inputs, not unproven analytic steps)

* `hW : W * CFC.sqrt S = 1` — the prior-supplied inverse square root `Σ_{B|R}^{-1/2}` (the paper's
  standardization uses exactly this prior-supplied quantity; nothing is estimated).
* `hS : S.PosSemidef` — `Σ_{B|R}` is a covariance matrix (used only where `√(r²Σ) = r√Σ` is
  needed, i.e. in the scaled/alternative pipeline).
* `hr : 0 ≤ r` (laws) resp. `0 < r` (power, where division by `r²` occurs), `hq : 0 ≤ q`
  (thresholds of a χ² test are nonnegative), `hα : chiSq (k*b) (Set.Ioi q) = α` (naming the
  size of the threshold `q`, i.e. `q = χ²_{kb,1−α}`).

There are no other hypotheses: every law identity below is proved, not assumed.

## Honesty

No `sorry`/`admit`/`native_decide`; `#print axioms` for every public theorem (see the end of the
file) lists only `propext`, `Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure ProbabilityTheory Matrix
open scoped NNReal ENNReal MatrixOrder

namespace AuditPrice

/-! ### Measurability helpers -/

private lemma measurable_sumSq {ι : Type*} [Fintype ι] :
    Measurable fun x : ι → ℝ => ∑ i, x i ^ 2 :=
  Finset.measurable_sum _ fun i _ => (measurable_pi_apply i).pow_const 2

private lemma measurable_piSum {k : ℕ} : Measurable fun t : Fin k → ℝ => ∑ i, t i :=
  Finset.measurable_sum _ fun i _ => measurable_pi_apply i

/-! ### The χ² law, route (b): pushforward of iid standard normals under the sum of squares -/

/-- **The χ²(n) law**: the pushforward of the `n`-fold product of standard 1-D Gaussians under
`x ↦ ∑ᵢ xᵢ²`.  This is route (b): "a sum of `n` independent squares of standard normals". -/
noncomputable def chiSq (n : ℕ) : Measure ℝ :=
  (Measure.pi fun _ : Fin n => gaussianReal 0 1).map fun x => ∑ i, x i ^ 2

lemma chiSq_def (n : ℕ) :
    chiSq n = (Measure.pi fun _ : Fin n => gaussianReal 0 1).map fun x => ∑ i, x i ^ 2 := rfl

instance isProbabilityMeasure_chiSq (n : ℕ) : IsProbabilityMeasure (chiSq n) :=
  isProbabilityMeasure_map (measurable_sumSq).aemeasurable

/-- **`χ²(n)` is the law of `‖Z‖²` for `Z` standard Gaussian on `ℝⁿ`**: the definitional bridge
between `chiSq` (a product of 1-D Gaussians pushed through the coordinate sum of squares) and
mathlib's `stdGaussian` on `EuclideanSpace ℝ (Fin n)` pushed through the squared norm. -/
theorem stdGaussian_normSq_law (n : ℕ) :
    (stdGaussian (EuclideanSpace ℝ (Fin n))).map (fun x => ‖x‖ ^ 2) = chiSq n := by
  have htoLp : Measurable (WithLp.toLp 2 : (Fin n → ℝ) → EuclideanSpace ℝ (Fin n)) :=
    (MeasurableEquiv.toLp 2 (Fin n → ℝ)).measurable
  rw [chiSq_def, ← map_pi_eq_stdGaussian,
    Measure.map_map (continuous_norm.pow 2).measurable htoLp]
  have hfun : ((fun x : EuclideanSpace ℝ (Fin n) => ‖x‖ ^ 2) ∘
      (WithLp.toLp 2 : (Fin n → ℝ) → EuclideanSpace ℝ (Fin n))) = fun x => ∑ i, x i ^ 2 := by
    funext x
    simp only [Function.comp_apply]
    rw [EuclideanSpace.real_norm_sq_eq]
  rw [hfun]

/-! ### χ² additivity: independent χ²'s add their degrees of freedom -/

/-- **χ² additivity**: the sum of independent `χ²(m)` and `χ²(n)` variables is `χ²(m+n)` —
"`T` is a sum of independent squares", assembled two blocks at a time. -/
theorem chiSq_add (m n : ℕ) :
    ((chiSq m).prod (chiSq n)).map (fun p : ℝ × ℝ => p.1 + p.2) = chiSq (m + n) := by
  classical
  -- Step 1: `χ²(m+n)` as the sum-of-squares pushforward of the `Fin m ⊕ Fin n`-indexed product.
  have key1 : chiSq (m + n)
      = (Measure.pi fun _ : Fin m ⊕ Fin n => gaussianReal (0 : ℝ) 1).map
          fun y => ∑ j, y j ^ 2 := by
    rw [chiSq_def,
      ← Measure.pi_map_piCongrLeft (finSumFinEquiv (m := m) (n := n))
        (fun _ : Fin (m + n) => gaussianReal (0 : ℝ) 1),
      Measure.map_map measurable_sumSq
        (MeasurableEquiv.piCongrLeft (fun _ : Fin (m + n) => ℝ) finSumFinEquiv).measurable]
    have hfun : ((fun x : Fin (m + n) → ℝ => ∑ i, x i ^ 2) ∘
        ⇑(MeasurableEquiv.piCongrLeft (fun _ : Fin (m + n) => ℝ) finSumFinEquiv))
        = fun y : Fin m ⊕ Fin n → ℝ => ∑ j, y j ^ 2 := by
      funext y
      simp only [Function.comp_apply, MeasurableEquiv.coe_piCongrLeft]
      exact (Fintype.sum_equiv finSumFinEquiv (fun j => y j ^ 2)
        (fun i => (Equiv.piCongrLeft (fun _ : Fin (m + n) => ℝ) finSumFinEquiv y) i ^ 2)
        (fun j => by simp)).symm
    rw [hfun]
  -- Step 2: split the sum-indexed product into the two blocks.
  have key2 : (Measure.pi fun _ : Fin m ⊕ Fin n => gaussianReal (0 : ℝ) 1).map
        (fun y => ∑ j, y j ^ 2)
      = ((Measure.pi fun _ : Fin m => gaussianReal (0 : ℝ) 1).prod
          (Measure.pi fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map
          fun p => (∑ i, p.1 i ^ 2) + ∑ j, p.2 j ^ 2 := by
    have h₂ := (measurePreserving_sumPiEquivProdPi
      (μ := fun _ : Fin m ⊕ Fin n => gaussianReal (0 : ℝ) 1)).map_eq
    have hpair : Measurable fun p : (Fin m → ℝ) × (Fin n → ℝ) =>
        (∑ i, p.1 i ^ 2) + ∑ j, p.2 j ^ 2 :=
      (measurable_sumSq.comp measurable_fst).add (measurable_sumSq.comp measurable_snd)
    rw [← h₂, Measure.map_map hpair
      (MeasurableEquiv.sumPiEquivProdPi (fun _ : Fin m ⊕ Fin n => ℝ)).measurable]
    have hfun : ((fun p : (Fin m → ℝ) × (Fin n → ℝ) => (∑ i, p.1 i ^ 2) + ∑ j, p.2 j ^ 2) ∘
        ⇑(MeasurableEquiv.sumPiEquivProdPi (fun _ : Fin m ⊕ Fin n => ℝ)))
        = fun y : Fin m ⊕ Fin n → ℝ => ∑ j, y j ^ 2 := by
      funext y
      simp only [Function.comp_apply, MeasurableEquiv.coe_sumPiEquivProdPi]
      exact (Fintype.sum_sum_type fun j => y j ^ 2).symm
    rw [hfun]
  -- Step 3: the two-block pushforward is the sum of the two χ² pushforwards.
  have key3 : ((chiSq m).prod (chiSq n)).map (fun p : ℝ × ℝ => p.1 + p.2)
      = ((Measure.pi fun _ : Fin m => gaussianReal (0 : ℝ) 1).prod
          (Measure.pi fun _ : Fin n => gaussianReal (0 : ℝ) 1)).map
          fun p => (∑ i, p.1 i ^ 2) + ∑ j, p.2 j ^ 2 := by
    rw [chiSq_def, chiSq_def,
      Measure.map_prod_map _ _ measurable_sumSq measurable_sumSq,
      Measure.map_map (measurable_fst.add measurable_snd)
        (measurable_sumSq.prodMap measurable_sumSq)]
    rfl
  rw [key3, ← key2]
  exact key1.symm

/-- **Sum of independent χ²'s**: for independent `χ²(dᵢ)` coordinates, the coordinate sum has law
`χ²(∑ᵢ dᵢ)`.  This is the paper's "T is a sum of `kb` independent squares", assembled over the
`k` references. -/
theorem pi_chiSq_sum_law : ∀ (k : ℕ) (d : Fin k → ℕ),
    (Measure.pi fun i => chiSq (d i)).map (fun t => ∑ i, t i) = chiSq (∑ i, d i) := by
  intro k
  induction k with
  | zero =>
    intro d
    have h0 : (∑ i, d i) = 0 := by simp
    rw [h0, Measure.pi_of_empty (fun i : Fin 0 => chiSq (d i)),
      map_dirac' measurable_piSum, chiSq_def,
      Measure.pi_of_empty (fun _ : Fin 0 => gaussianReal (0 : ℝ) 1),
      map_dirac' measurable_sumSq]
    simp
  | succ k ih =>
    intro d
    have hsplit := measurePreserving_piFinSuccAbove (fun i : Fin (k + 1) => chiSq (d i)) 0
    have hsum : (fun t : Fin (k + 1) → ℝ => ∑ i, t i)
        = (fun p : ℝ × (Fin k → ℝ) => p.1 + ∑ j, p.2 j) ∘
          ⇑(MeasurableEquiv.piFinSuccAbove (fun _ : Fin (k + 1) => ℝ) 0) := by
      funext t
      exact Fin.sum_univ_succAbove t 0
    have haddAgg : Measurable fun p : ℝ × (Fin k → ℝ) => p.1 + ∑ j, p.2 j :=
      measurable_fst.add (measurable_piSum.comp measurable_snd)
    rw [hsum, ← Measure.map_map haddAgg
        (MeasurableEquiv.piFinSuccAbove (fun _ : Fin (k + 1) => ℝ) 0).measurable,
      hsplit.map_eq]
    have hstep : (fun p : ℝ × (Fin k → ℝ) => p.1 + ∑ j, p.2 j)
        = (fun q : ℝ × ℝ => q.1 + q.2) ∘ Prod.map id (fun t : Fin k → ℝ => ∑ j, t j) := rfl
    rw [hstep, ← Measure.map_map (measurable_fst.add measurable_snd)
        (measurable_id.prodMap measurable_piSum),
      ← Measure.map_prod_map _ _ measurable_id measurable_piSum, Measure.map_id,
      ih fun j => d ((0 : Fin (k + 1)).succAbove j), chiSq_add]
    congr 1
    exact (Fin.sum_univ_succAbove d 0).symm

/-! ### The audit statistic and its law under the null and the scaled alternative -/

/-- **The audit statistic** `T(u) = ∑ᵢ ‖uᵢ‖²` over the `k` standardized blind-fiber residuals
`uᵢ ∈ ℝᵇ`. -/
noncomputable def auditStat {k b : ℕ} (u : Fin k → EuclideanSpace ℝ (Fin b)) : ℝ :=
  ∑ i, ‖u i‖ ^ 2

theorem measurable_auditStat {k b : ℕ} : Measurable (auditStat (k := k) (b := b)) :=
  Finset.measurable_sum _ fun i _ =>
    (continuous_norm.pow 2).measurable.comp (measurable_pi_apply i)

/-- **The null law** (`uᵢ ~ N(0, I_b)` iid ⇒ `T ~ χ²(k·b)`): if the prior's blind report is
correct then the standardized residuals are iid standard Gaussians on the blind fiber, and the
audit statistic has law `χ²(kb)` — exactly, at every `k`, `b`. -/
theorem auditStat_null_law (k b : ℕ) :
    (Measure.pi fun _ : Fin k => stdGaussian (EuclideanSpace ℝ (Fin b))).map auditStat
      = chiSq (k * b) := by
  have hnormSq : Measurable fun x : EuclideanSpace ℝ (Fin b) => ‖x‖ ^ 2 :=
    (continuous_norm.pow 2).measurable
  haveI : IsProbabilityMeasure
      ((stdGaussian (EuclideanSpace ℝ (Fin b))).map fun x => ‖x‖ ^ 2) :=
    isProbabilityMeasure_map hnormSq.aemeasurable
  have hcomp : (auditStat : (Fin k → EuclideanSpace ℝ (Fin b)) → ℝ)
      = (fun t : Fin k → ℝ => ∑ i, t i) ∘ fun u i => ‖u i‖ ^ 2 := rfl
  have hpiNorm : Measurable fun (u : Fin k → EuclideanSpace ℝ (Fin b)) i => ‖u i‖ ^ 2 :=
    measurable_pi_lambda _ fun i => hnormSq.comp (measurable_pi_apply i)
  rw [hcomp, ← Measure.map_map measurable_piSum hpiNorm,
    Measure.pi_map_pi (fun _ => hnormSq.aemeasurable)]
  simp_rw [stdGaussian_normSq_law b]
  rw [pi_chiSq_sum_law k fun _ => b]
  congr 1
  simp [Finset.card_univ]

/-- **The scaled law** ("inflating the covariance to `r²Σ` scales each residual by `r`"):
if each residual is a standard Gaussian scaled by `r`, the audit statistic has the law of
`r² · χ²(kb)` (the pushforward of `χ²(kb)` under `t ↦ r²t`).  No sign condition on `r` is
needed for the law identity. -/
theorem auditStat_smul_law (k b : ℕ) (r : ℝ) :
    (Measure.pi fun _ : Fin k =>
        (stdGaussian (EuclideanSpace ℝ (Fin b))).map fun x => r • x).map auditStat
      = (chiSq (k * b)).map fun t => r ^ 2 * t := by
  have hsmul : Measurable fun x : EuclideanSpace ℝ (Fin b) => r • x :=
    (continuous_const_smul r).measurable
  haveI : IsProbabilityMeasure
      ((stdGaussian (EuclideanSpace ℝ (Fin b))).map fun x => r • x) :=
    isProbabilityMeasure_map hsmul.aemeasurable
  have hsmulPi : Measurable fun (u : Fin k → EuclideanSpace ℝ (Fin b)) i => r • u i :=
    measurable_pi_lambda _ fun i => hsmul.comp (measurable_pi_apply i)
  rw [← Measure.pi_map_pi
      (μ := fun _ : Fin k => stdGaussian (EuclideanSpace ℝ (Fin b)))
      (f := fun _ x => r • x) (fun _ => hsmul.aemeasurable),
    Measure.map_map measurable_auditStat hsmulPi]
  have h1 : (auditStat ∘ fun (u : Fin k → EuclideanSpace ℝ (Fin b)) i => r • u i)
      = (fun t : ℝ => r ^ 2 * t) ∘ auditStat := by
    funext u
    simp only [Function.comp_apply, auditStat]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
  rw [h1, ← Measure.map_map (measurable_const_mul (r ^ 2)) measurable_auditStat,
    auditStat_null_law]

/-! ### The CFC square-root lemma mathlib lacks, and `N(0, r²I)` as a scaled standard Gaussian -/

/-- **`√(r²·Σ) = r·√Σ`** for `r ≥ 0` and `Σ` positive semidefinite: the matrix square root of the
inflated covariance is the inflated square root.  Built here mathlib-style (missing from the
pinned mathlib); this is the algebraic heart of "inflating the covariance scales the residual". -/
theorem sqrt_sq_smul {ι : Type*} [Fintype ι] [DecidableEq ι] {r : ℝ} (hr : 0 ≤ r)
    {S : Matrix ι ι ℝ} (hS : S.PosSemidef) :
    CFC.sqrt (r ^ 2 • S) = r • CFC.sqrt S := by
  refine CFC.sqrt_unique ?_ ?_
  · rw [smul_mul_assoc, mul_smul_comm, smul_smul, CFC.sqrt_mul_sqrt_self S hS.nonneg]
    congr 1
    ring
  · exact ((Matrix.nonneg_iff_posSemidef.mp (CFC.sqrt_nonneg S)).smul hr).nonneg

/-- **`N(0, r²I_b)` is the `r`-scaled standard Gaussian** on the blind fiber: the alternative's
residual law, written both ways. -/
theorem multivariateGaussian_isotropic (b : ℕ) {r : ℝ} (hr : 0 ≤ r) :
    multivariateGaussian (0 : EuclideanSpace ℝ (Fin b)) (r ^ 2 • 1)
      = (stdGaussian (EuclideanSpace ℝ (Fin b))).map fun x => r • x := by
  have h1 : CFC.sqrt (1 : Matrix (Fin b) (Fin b) ℝ) = 1 :=
    CFC.sqrt_unique (one_mul 1) Matrix.PosSemidef.one.nonneg
  rw [multivariateGaussian, sqrt_sq_smul hr Matrix.PosSemidef.one, h1]
  congr 1
  funext z
  simp only [zero_add, map_smul, map_one, _root_.smul_apply, one_apply_eq_self]

/-- **The alternative law at the standardized residuals** (`uᵢ ~ N(0, r²I_b)` iid ⇒
`T ~ r²·χ²(kb)`): against a misreport inflating the blind spread by factor `r` throughout the
subspace, the audit statistic has the law of `χ²(kb)` pushed forward by `t ↦ r²t`. -/
theorem auditStat_alt_law (k b : ℕ) {r : ℝ} (hr : 0 ≤ r) :
    (Measure.pi fun _ : Fin k =>
        multivariateGaussian (0 : EuclideanSpace ℝ (Fin b)) (r ^ 2 • 1)).map auditStat
      = (chiSq (k * b)).map fun t => r ^ 2 * t := by
  simp_rw [multivariateGaussian_isotropic b hr]
  exact auditStat_smul_law k b r

/-! ### Standardization: the prior-supplied `Σ_{B|R}^{-1/2}` whitens the Gaussian fiber -/

/-- **Standardization** (the paper's "residuals standardized by prior-supplied quantities ⇒
standard normal under the null with nothing estimated"): if the fiber residual has law `N(m, Σ)`
and `W` is a left inverse of the square root `√Σ` (the prior-supplied `Σ_{B|R}^{-1/2}`), then
`u = W(x − m)` is a standard Gaussian.  `hW` is a model input: the prior supplies `Σ_{B|R}`
exactly. -/
theorem multivariateGaussian_standardize {ι : Type*} [Fintype ι] [DecidableEq ι]
    (m : EuclideanSpace ℝ ι) (S W : Matrix ι ι ℝ) (hW : W * CFC.sqrt S = 1) :
    (multivariateGaussian m S).map
        (fun x => toEuclideanCLM (𝕜 := ℝ) W (x - m))
      = stdGaussian (EuclideanSpace ℝ ι) := by
  have hres : Measurable fun x : EuclideanSpace ℝ ι => toEuclideanCLM (𝕜 := ℝ) W (x - m) :=
    ((toEuclideanCLM (𝕜 := ℝ) W).continuous.comp (continuous_id.sub continuous_const)).measurable
  have hfwd : Measurable fun x : EuclideanSpace ℝ ι =>
      m + toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt S) x :=
    (continuous_const.add (toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt S)).continuous).measurable
  rw [multivariateGaussian, Measure.map_map hres hfwd]
  have hfun : ((fun x : EuclideanSpace ℝ ι => toEuclideanCLM (𝕜 := ℝ) W (x - m)) ∘
      fun x => m + toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt S) x) = id := by
    funext z
    simp only [Function.comp_apply, add_sub_cancel_left, id_eq]
    rw [← mul_apply_eq_comp, ← map_mul, hW, map_one, one_apply_eq_self]
  rw [hfun, Measure.map_id]

/-- **Standardization with `Σ^{-1/2}` verbatim**: when the square root is invertible, the paper's
`u = Σ_{B|R}^{-1/2}(x − m)` is a standard Gaussian. -/
theorem multivariateGaussian_standardize_inv {ι : Type*} [Fintype ι] [DecidableEq ι]
    (m : EuclideanSpace ℝ ι) (S : Matrix ι ι ℝ) (h : IsUnit (CFC.sqrt S).det) :
    (multivariateGaussian m S).map
        (fun x => toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt S)⁻¹ (x - m))
      = stdGaussian (EuclideanSpace ℝ ι) :=
  multivariateGaussian_standardize m S _ (Matrix.nonsing_inv_mul _ h)

/-- **Standardizing an inflated fiber** (the alternative, one residual): if the truth's fiber law
is `N(m, r²Σ)` while the report says `Σ`, then the reported standardization `u = W(x − m)` is the
`r`-scaled standard Gaussian — "inflating the covariance to `r²Σ` scales each residual by `r`". -/
theorem multivariateGaussian_scaled_standardize {ι : Type*} [Fintype ι] [DecidableEq ι]
    (m : EuclideanSpace ℝ ι) (S W : Matrix ι ι ℝ) (hS : S.PosSemidef)
    (hW : W * CFC.sqrt S = 1) {r : ℝ} (hr : 0 ≤ r) :
    (multivariateGaussian m (r ^ 2 • S)).map
        (fun x => toEuclideanCLM (𝕜 := ℝ) W (x - m))
      = (stdGaussian (EuclideanSpace ℝ ι)).map fun x => r • x := by
  have hres : Measurable fun x : EuclideanSpace ℝ ι => toEuclideanCLM (𝕜 := ℝ) W (x - m) :=
    ((toEuclideanCLM (𝕜 := ℝ) W).continuous.comp (continuous_id.sub continuous_const)).measurable
  have hfwd : Measurable fun x : EuclideanSpace ℝ ι =>
      m + toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt (r ^ 2 • S)) x :=
    (continuous_const.add (toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt (r ^ 2 • S))).continuous).measurable
  rw [multivariateGaussian, Measure.map_map hres hfwd]
  have hfun : ((fun x : EuclideanSpace ℝ ι => toEuclideanCLM (𝕜 := ℝ) W (x - m)) ∘
      fun x => m + toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt (r ^ 2 • S)) x) = fun x => r • x := by
    funext z
    simp only [Function.comp_apply, add_sub_cancel_left]
    rw [sqrt_sq_smul hr hS, ← mul_apply_eq_comp, ← map_mul, mul_smul_comm, hW,
      map_smul, map_one, _root_.smul_apply, one_apply_eq_self]
  rw [hfun]

/-! ### The full pipelines: k iid Gaussian fiber residuals → standardized → T -/

/-- **The null, full pipeline**: `k` iid fiber residuals `xᵢ ~ N(m, Σ)`, standardized by the
prior-supplied `W = Σ^{-1/2}`, give `T = ∑ᵢ ‖W(xᵢ − m)‖² ~ χ²(kb)` — the proposition's null,
with the standardization step included. -/
theorem audit_null_pipeline (k b : ℕ) (m : EuclideanSpace ℝ (Fin b))
    (S W : Matrix (Fin b) (Fin b) ℝ) (hW : W * CFC.sqrt S = 1) :
    (Measure.pi fun _ : Fin k => multivariateGaussian m S).map
        (fun x => auditStat fun i => toEuclideanCLM (𝕜 := ℝ) W (x i - m))
      = chiSq (k * b) := by
  have hres : Measurable fun x : EuclideanSpace ℝ (Fin b) =>
      toEuclideanCLM (𝕜 := ℝ) W (x - m) :=
    ((toEuclideanCLM (𝕜 := ℝ) W).continuous.comp (continuous_id.sub continuous_const)).measurable
  haveI : IsProbabilityMeasure ((multivariateGaussian m S).map
      fun x => toEuclideanCLM (𝕜 := ℝ) W (x - m)) :=
    isProbabilityMeasure_map hres.aemeasurable
  have hcomp : (fun x : Fin k → EuclideanSpace ℝ (Fin b) =>
        auditStat fun i => toEuclideanCLM (𝕜 := ℝ) W (x i - m))
      = auditStat ∘ fun x i => toEuclideanCLM (𝕜 := ℝ) W (x i - m) := rfl
  have hresPi : Measurable fun (x : Fin k → EuclideanSpace ℝ (Fin b)) i =>
      toEuclideanCLM (𝕜 := ℝ) W (x i - m) :=
    measurable_pi_lambda _ fun i => hres.comp (measurable_pi_apply i)
  rw [hcomp, ← Measure.map_map measurable_auditStat hresPi,
    Measure.pi_map_pi (fun _ => hres.aemeasurable)]
  simp_rw [multivariateGaussian_standardize m S W hW]
  exact auditStat_null_law k b

/-- **The alternative, full pipeline**: `k` iid fiber residuals from the inflated truth
`xᵢ ~ N(m, r²Σ)`, standardized by the REPORTED `W = Σ^{-1/2}`, give
`T ~ r²·χ²(kb)` (the pushforward of `χ²(kb)` under `t ↦ r²t`). -/
theorem audit_alt_pipeline (k b : ℕ) (m : EuclideanSpace ℝ (Fin b))
    (S W : Matrix (Fin b) (Fin b) ℝ) (hS : S.PosSemidef) (hW : W * CFC.sqrt S = 1)
    {r : ℝ} (hr : 0 ≤ r) :
    (Measure.pi fun _ : Fin k => multivariateGaussian m (r ^ 2 • S)).map
        (fun x => auditStat fun i => toEuclideanCLM (𝕜 := ℝ) W (x i - m))
      = (chiSq (k * b)).map fun t => r ^ 2 * t := by
  have hres : Measurable fun x : EuclideanSpace ℝ (Fin b) =>
      toEuclideanCLM (𝕜 := ℝ) W (x - m) :=
    ((toEuclideanCLM (𝕜 := ℝ) W).continuous.comp (continuous_id.sub continuous_const)).measurable
  haveI : IsProbabilityMeasure ((multivariateGaussian m (r ^ 2 • S)).map
      fun x => toEuclideanCLM (𝕜 := ℝ) W (x - m)) :=
    isProbabilityMeasure_map hres.aemeasurable
  have hcomp : (fun x : Fin k → EuclideanSpace ℝ (Fin b) =>
        auditStat fun i => toEuclideanCLM (𝕜 := ℝ) W (x i - m))
      = auditStat ∘ fun x i => toEuclideanCLM (𝕜 := ℝ) W (x i - m) := rfl
  have hresPi : Measurable fun (x : Fin k → EuclideanSpace ℝ (Fin b)) i =>
      toEuclideanCLM (𝕜 := ℝ) W (x i - m) :=
    measurable_pi_lambda _ fun i => hres.comp (measurable_pi_apply i)
  rw [hcomp, ← Measure.map_map measurable_auditStat hresPi,
    Measure.pi_map_pi (fun _ => hres.aemeasurable)]
  simp_rw [multivariateGaussian_scaled_standardize m S W hS hW hr]
  exact auditStat_smul_law k b r

/-! ### Power against an arbitrary threshold, monotonicity, and level over the composite null -/

/-- Scaling a real law by `c > 0` shifts an upper-tail probability:
`P(c·X > q) = P(X > q/c)`. -/
theorem map_const_mul_Ioi (μ : Measure ℝ) {c : ℝ} (hc : 0 < c) (q : ℝ) :
    (μ.map fun t => c * t) (Set.Ioi q) = μ (Set.Ioi (q / c)) := by
  rw [Measure.map_apply (measurable_const_mul c) measurableSet_Ioi]
  congr 1
  ext t
  simp only [Set.mem_preimage, Set.mem_Ioi]
  rw [mul_comm, ← div_lt_iff₀ hc]

/-- **Size**: under the null the rejection event `{T > q}` has probability exactly
`χ²(kb)(Ioi q)` — with `q = χ²_{kb,1−α}` (hypothesis `hα` below) this is `α`. -/
theorem audit_size (k b : ℕ) (q : ℝ) :
    ((Measure.pi fun _ : Fin k => stdGaussian (EuclideanSpace ℝ (Fin b))).map auditStat)
        (Set.Ioi q)
      = chiSq (k * b) (Set.Ioi q) := by
  rw [auditStat_null_law]

/-- **Power** ("the test rejecting at `q` has power `Pr(χ²_{kb} > q/r²)`"): against the
alternative `uᵢ ~ N(0, r²I_b)`, the rejection probability at any threshold `q` is exactly
`χ²(kb)(Ioi (q/r²))`.  With `q = χ²_{kb,1−α}` this is the proposition's
`1−β = Pr(χ²_{kb} > χ²_{kb,1−α}/r²)`. -/
theorem audit_power (k b : ℕ) {r : ℝ} (hr : 0 < r) (q : ℝ) :
    ((Measure.pi fun _ : Fin k =>
        multivariateGaussian (0 : EuclideanSpace ℝ (Fin b)) (r ^ 2 • 1)).map auditStat)
        (Set.Ioi q)
      = chiSq (k * b) (Set.Ioi (q / r ^ 2)) := by
  rw [auditStat_alt_law k b hr.le, map_const_mul_Ioi _ (by positivity) q]

/-- **Power, full pipeline**: the same rejection probability computed from the `k` iid inflated
fiber residuals standardized by the reported `W`. -/
theorem audit_power_pipeline (k b : ℕ) (m : EuclideanSpace ℝ (Fin b))
    (S W : Matrix (Fin b) (Fin b) ℝ) (hS : S.PosSemidef) (hW : W * CFC.sqrt S = 1)
    {r : ℝ} (hr : 0 < r) (q : ℝ) :
    ((Measure.pi fun _ : Fin k => multivariateGaussian m (r ^ 2 • S)).map
        (fun x => auditStat fun i => toEuclideanCLM (𝕜 := ℝ) W (x i - m)))
        (Set.Ioi q)
      = chiSq (k * b) (Set.Ioi (q / r ^ 2)) := by
  rw [audit_alt_pipeline k b m S W hS hW hr.le, map_const_mul_Ioi _ (by positivity) q]

/-- **The rejection probability is monotone nondecreasing in the inflation `r`** (at any
nonnegative threshold): the mechanism behind "level over the composite null". -/
theorem audit_rejection_mono (n : ℕ) {r r' q : ℝ} (hr : 0 < r) (hrr' : r ≤ r') (hq : 0 ≤ q) :
    chiSq n (Set.Ioi (q / r ^ 2)) ≤ chiSq n (Set.Ioi (q / r' ^ 2)) := by
  have h2 : r ^ 2 ≤ r' ^ 2 := by nlinarith
  exact measure_mono (Set.Ioi_subset_Ioi (div_le_div_of_nonneg_left hq (by positivity) h2))

/-- **Level over the composite null** ("rejection probability increases in the inflation, so the
test is level α over the composite null"): for every deflated-or-correct report `0 < r ≤ 1`,
the rejection probability is at most the size `χ²(kb)(Ioi q)` attained at `r = 1`. -/
theorem audit_level (k b : ℕ) {r : ℝ} (hr : 0 < r) (hr1 : r ≤ 1) {q : ℝ} (hq : 0 ≤ q) :
    ((Measure.pi fun _ : Fin k =>
        multivariateGaussian (0 : EuclideanSpace ℝ (Fin b)) (r ^ 2 • 1)).map auditStat)
        (Set.Ioi q)
      ≤ chiSq (k * b) (Set.Ioi q) := by
  rw [audit_power k b hr q]
  have h := audit_rejection_mono (k * b) hr hr1 hq
  simpa using h

/-- **Level α, named**: with the threshold's size named `α` (i.e. `q = χ²_{kb,1−α}`), every
report with `0 < r ≤ 1` is rejected with probability at most `α` — the test is level α over the
composite null. -/
theorem audit_level_alpha (k b : ℕ) {r q : ℝ} {α : ℝ≥0∞} (hr : 0 < r) (hr1 : r ≤ 1)
    (hq : 0 ≤ q) (hα : chiSq (k * b) (Set.Ioi q) = α) :
    ((Measure.pi fun _ : Fin k =>
        multivariateGaussian (0 : EuclideanSpace ℝ (Fin b)) (r ^ 2 • 1)).map auditStat)
        (Set.Ioi q)
      ≤ α :=
  hα ▸ audit_level k b hr hr1 hq

end AuditPrice

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms AuditPrice.stdGaussian_normSq_law
#print axioms AuditPrice.chiSq_add
#print axioms AuditPrice.pi_chiSq_sum_law
#print axioms AuditPrice.measurable_auditStat
#print axioms AuditPrice.auditStat_null_law
#print axioms AuditPrice.auditStat_smul_law
#print axioms AuditPrice.sqrt_sq_smul
#print axioms AuditPrice.multivariateGaussian_isotropic
#print axioms AuditPrice.auditStat_alt_law
#print axioms AuditPrice.multivariateGaussian_standardize
#print axioms AuditPrice.multivariateGaussian_standardize_inv
#print axioms AuditPrice.multivariateGaussian_scaled_standardize
#print axioms AuditPrice.audit_null_pipeline
#print axioms AuditPrice.audit_alt_pipeline
#print axioms AuditPrice.map_const_mul_Ioi
#print axioms AuditPrice.audit_size
#print axioms AuditPrice.audit_power
#print axioms AuditPrice.audit_power_pipeline
#print axioms AuditPrice.audit_rejection_mono
#print axioms AuditPrice.audit_level
#print axioms AuditPrice.audit_level_alpha
