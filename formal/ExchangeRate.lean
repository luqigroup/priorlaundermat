import Mathlib
import SpreadKL
import MeanKL
import AuditPrice

/-!
# The exchange rate between surveys and references (Remark `r:exchange`)

Machine-checked assembly of the paper's Remark *"The exchange rate between surveys and
references"* (`r:exchange` of `paper/v2/manuscript.tex`): the same rank-one blind-spread
misreport `Σ⋆ ↦ Σ⋆ + w·vvᵀ` can be pursued in surveys or in reference truths, and the two
prices behave oppositely.  Almost everything here is REUSED from already-checked modules —
`SpreadKL.lean` (the spread crossover `n < 2/t²` on the physical record laws), `MeanKL.lean`
(the mean crossover `n < 1/β`), `DataCovarianceSqrt.lean` (`blindGap`, the constructed data
covariance), and `AuditPrice.lean` (the audit's χ² laws, power and level) — and this module
adds only the dictionary, the noise-floor bound, and the packaging.

## The four steps

**(1) The `r`–`w`–`s⋆` dictionary.**  `spreadRatio w s⋆ = √(1 + w/s⋆²)` is the audit's spread
ratio induced by the witness `Σ⋆ + w·vvᵀ` along a blind direction with reported standard
deviation `s⋆`: `spreadRatio_sq_mul` says `r²·s⋆² = s⋆² + w` (the inflated variance), and
`spreadRatio_witness` / `spreadRatio_strictMonoOn` / `spreadRatio_bijOn` make `w ↦ r` a
strictly monotone bijection `[0,∞) → [1,∞)` with explicit inverse `w = s⋆²(r²−1)`.

**(2) The survey price diverges.**  `blindGap_le_noise_floor` proves
`(Av)ᵀS_y⁻¹(Av) ≤ ‖Av‖²/γ` from the variational noise floor `γ ≤ λ_min(Γ)` (hypothesis `hΓ`
below, PROVED satisfiable by `exists_noise_floor` — so nothing here is vacuous), which turns the
checked spread crossover `n·t² < 2`, `t = w·(Av)ᵀS_y⁻¹(Av)` (`SpreadKL.lean`) into the remark's
`n < 2·(γ/(w‖Av‖²))²` — a guaranteed-indistinguishable survey budget that `survey_price_diverges`
shows tends to `∞` as the illumination `‖Av‖² → 0⁺` (a `Filter.Tendsto` statement), and that is
infinite on a genuine kernel: at `Av = 0` the two record laws are EQUAL (via the checked
per-survey KL `= 0` and mathlib's converse Gibbs inequality), so every test at every `n` has
summed error `≥ 1`.  `survey_price_mean` is the mean-misreport analogue (`n < γ/(w²‖Av‖²)`,
from `MeanKL.lean`): both moments' survey prices diverge.

**(3) The reference price does not involve `A`.**  `surveyedAuditNullLaw` / `surveyedAuditAltLaw`
/ `surveyedAuditPower` are the audit's null law, alternative law and rejection probability
written WITH a forward operator in scope — and `audit_price_operator_free` proves (by `rfl`,
the definitional equality made explicit) that for two ARBITRARY operators, of arbitrary and even
different dimensions, all three coincide.  `surveyedAuditPower_factors` exhibits the power as the
operator-free `referencePrice k b r q = χ²_{kb}(Ioi (q/r²))` (reusing `AuditPrice.audit_power`),
and `surveyedAuditLevel` re-exports the level bound.  `reference_price_pipeline` runs the audit
at the physical witness itself: `k` references with true one-dimensional blind fiber variance
`s⋆² + w` audited against the reported `s⋆²` reject with probability
`χ²_k(Ioi (q·s⋆²/(s⋆²+w)))` — a function of `(k, s⋆, w, q)` alone.

**(4) CAPSTONE `exchange_rate`.**  One theorem packaging (2)+(3): for the physical record laws
`y = Ax + ε` there is a root `Lx'` of the bumped truth covariance `Σ⋆ + w·vvᵀ` such that
(i) below the crossover — `n·(w‖Av‖²/γ)² < 2`, verbatim form `n < 2·(γ/(w‖Av‖²))²` — no test on
`n` surveys separates the two truths (summed error `> 1/2`); (ii) on a genuine kernel
(`Av = 0`) the two record laws are literally equal and every test at every `n` has summed error
`≥ 1`; while (iii) the `k`-reference audit at the same misreport (ratio `spreadRatio w s⋆`) has
rejection probability the fixed χ² expression `χ²_k(Ioi (q·s⋆²/(s⋆²+w)))`, with no operator
anywhere in the formula.  The exchange rate is a divergence, not a constant.

## Named hypotheses (scenario inputs, none un-closable)

* `hΓ : ∀ ξ, γ‖ξ‖² ≤ ‖Lεᵀξ‖²` — the variational form of `γ ≤ λ_min(Γ)`, `Γ = LεLεᵀ` (the
  remark's `λ_min(Γ)` is the largest such `γ`).  It is a named constant of the scenario, and
  `exists_noise_floor` PROVES a positive `γ` always exists, so the hypothesis is satisfiable
  for every invertible noise root — nothing is assumed that cannot be discharged.

There are no other hypotheses beyond the scenario (`w ≥ 0`, `s⋆ > 0`, invertible roots).

## Deliberately not formalized

The remark's `≳` is sharpened here to the checked constant (`n < 2·(γ/(w‖Av‖²))²`, from
`SpreadKL`'s `n < 2/t²`); the `log b` search penalty for a rank-one misreport in an UNKNOWN
direction, and the asymptotic `k ≈ 1 + (z_{1−α}+z_{1−β})²/(2b log²r)` display, are out of scope
(the latter deliberately absent from `AuditPrice.lean` as well).

## Honesty

No `sorry`/`admit`/`native_decide`; `#print axioms` on each public result (end of file) lists
only `propext`, `Classical.choice`, `Quot.sound`.
-/

open MeasureTheory Measure ProbabilityTheory InformationTheory Matrix Filter
open scoped ENNReal NNReal RealInnerProductSpace MatrixOrder

namespace ExchangeRate

open NearNull

/-! ### (1) The `r`–`w`–`s⋆` dictionary -/

/-- **The audit's spread ratio induced by the witness** `Σ⋆ ↦ Σ⋆ + w·vvᵀ` along a blind
direction with reported standard deviation `s⋆`: `r = √(1 + w/s⋆²)` (the remark's dictionary). -/
noncomputable def spreadRatio (w sStar : ℝ) : ℝ :=
  Real.sqrt (1 + w / sStar ^ 2)

/-- `r² = 1 + w/s⋆²` — the dictionary squared. -/
theorem spreadRatio_sq {w : ℝ} (hw : 0 ≤ w) (s : ℝ) :
    spreadRatio w s ^ 2 = 1 + w / s ^ 2 :=
  Real.sq_sqrt (by have := div_nonneg hw (sq_nonneg s); linarith)

/-- An inflation never deflates: `1 ≤ r`. -/
theorem one_le_spreadRatio {w : ℝ} (hw : 0 ≤ w) (s : ℝ) : 1 ≤ spreadRatio w s :=
  Real.one_le_sqrt.mpr (le_add_of_nonneg_right (div_nonneg hw (sq_nonneg s)))

/-- The ratio is positive. -/
theorem spreadRatio_pos {w : ℝ} (hw : 0 ≤ w) (s : ℝ) : 0 < spreadRatio w s :=
  lt_of_lt_of_le one_pos (one_le_spreadRatio hw s)

/-- **The inflated variance:** `r²·s⋆² = s⋆² + w` — the witness `Σ⋆ + w·vvᵀ` read along `v` is
exactly the audit's `r²`-inflated fiber variance. -/
theorem spreadRatio_sq_mul {w s : ℝ} (hw : 0 ≤ w) (hs : s ≠ 0) :
    spreadRatio w s ^ 2 * s ^ 2 = s ^ 2 + w := by
  rw [spreadRatio_sq hw, add_mul, one_mul, div_mul_cancel₀ _ (pow_ne_zero 2 hs)]

/-- **The inverse dictionary:** every ratio `r ≥ 1` is bought by the witness size
`w = s⋆²(r²−1)`. -/
theorem spreadRatio_witness {r s : ℝ} (hr : 1 ≤ r) (hs : s ≠ 0) :
    spreadRatio (s ^ 2 * (r ^ 2 - 1)) s = r := by
  unfold spreadRatio
  have h1 : s ^ 2 * (r ^ 2 - 1) / s ^ 2 = r ^ 2 - 1 := by
    rw [mul_comm, mul_div_assoc, div_self (pow_ne_zero 2 hs), mul_one]
  rw [h1, show (1 : ℝ) + (r ^ 2 - 1) = r ^ 2 by ring]
  exact Real.sqrt_sq (by linarith)

/-- The dictionary is strictly monotone on `w ∈ [0,∞)` (for `s⋆ ≠ 0`). -/
theorem spreadRatio_strictMonoOn {s : ℝ} (hs : s ≠ 0) :
    StrictMonoOn (fun w => spreadRatio w s) (Set.Ici 0) := by
  intro w₁ hw₁ w₂ _ h
  have hs2 : (0 : ℝ) < s ^ 2 := lt_of_le_of_ne (sq_nonneg s) (Ne.symm (pow_ne_zero 2 hs))
  have hdiv : w₁ / s ^ 2 < w₂ / s ^ 2 := (div_lt_div_iff_of_pos_right hs2).mpr h
  have h1 : (0 : ℝ) ≤ 1 + w₁ / s ^ 2 := by
    have := div_nonneg (Set.mem_Ici.mp hw₁) (sq_nonneg s)
    linarith
  exact Real.sqrt_lt_sqrt h1 (by linarith)

/-- **The monotone bijection `w ↔ r`:** for `s⋆ ≠ 0` the dictionary is a bijection from the
witness sizes `[0,∞)` onto the spread ratios `[1,∞)`. -/
theorem spreadRatio_bijOn {s : ℝ} (hs : s ≠ 0) :
    Set.BijOn (fun w => spreadRatio w s) (Set.Ici 0) (Set.Ici 1) := by
  refine ⟨fun w hw => Set.mem_Ici.mpr (one_le_spreadRatio (Set.mem_Ici.mp hw) s),
    (spreadRatio_strictMonoOn hs).injOn, ?_⟩
  intro r hr
  have hr1 : (1 : ℝ) ≤ r := Set.mem_Ici.mp hr
  refine ⟨s ^ 2 * (r ^ 2 - 1), Set.mem_Ici.mpr ?_, spreadRatio_witness hr1 hs⟩
  have h1 : (0 : ℝ) ≤ r ^ 2 - 1 := by nlinarith [mul_self_nonneg (r - 1), hr1]
  exact mul_nonneg (sq_nonneg s) h1

/-! ### (2a) The noise floor and the blind-gap bound -/

/-- **The noise floor exists:** every invertible noise root `Lε` admits a positive `γ` with
`γ‖ξ‖² ≤ ‖Lεᵀξ‖²` for all `ξ` — the variational `γ ≤ λ_min(Γ)`, `Γ = LεLεᵀ`.  So the named
hypothesis `hΓ` of the theorems below is satisfiable for every scenario. -/
theorem exists_noise_floor {m : ℕ} (Lε : (E m) ≃L[ℝ] (E m)) :
    ∃ γ : ℝ, 0 < γ ∧ ∀ ξ : E m,
      γ * ‖ξ‖ ^ 2 ≤ ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) ξ‖ ^ 2 := by
  -- `ξ = (Lε⁻¹)ᵀ (Lεᵀ ξ)`, so `‖ξ‖ ≤ ‖(Lε⁻¹)ᵀ‖·‖Lεᵀξ‖` and `γ = ‖(Lε⁻¹)ᵀ‖⁻²` works.
  have hrec : ∀ ξ : E m,
      (ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m)))
        ((ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m))) ξ) = ξ := by
    intro ξ
    have hcomp : ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))
        ∘L ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m))
        = ContinuousLinearMap.id ℝ (E m) := by
      rw [← ContinuousLinearMap.adjoint_comp]
      have hid : (↑Lε : (E m) →L[ℝ] (E m)) ∘L (↑Lε.symm : (E m) →L[ℝ] (E m))
          = ContinuousLinearMap.id ℝ (E m) := by
        ext ζ; simp
      rw [hid, ContinuousLinearMap.adjoint_id]
    calc (ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m)))
          ((ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m))) ξ)
        = (ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))
            ∘L ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m))) ξ := rfl
      _ = ContinuousLinearMap.id ℝ (E m) ξ := by rw [hcomp]
      _ = ξ := rfl
  have hbound : ∀ ξ : E m,
      ‖ξ‖ ≤ ‖ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))‖
        * ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) ξ‖ := by
    intro ξ
    conv_lhs => rw [← hrec ξ]
    exact (ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))).le_opNorm _
  have hBpos : 0 < ‖ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))‖ := by
    rcases (norm_nonneg
        (ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m)))).lt_or_eq with h | h
    · exact h
    · exfalso
      have h1 := hbound (e0 m)
      rw [← h, zero_mul, norm_e0] at h1
      linarith
  refine ⟨(‖ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))‖ ^ 2)⁻¹,
    inv_pos.mpr (pow_pos hBpos 2), fun ξ => ?_⟩
  have h1 := hbound ξ
  have h2 : ‖ξ‖ ^ 2 ≤ (‖ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))‖
      * ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) ξ‖) ^ 2 :=
    pow_le_pow_left₀ (norm_nonneg ξ) h1 2
  rw [mul_pow] at h2
  have h3 := mul_le_mul_of_nonneg_left h2
    (inv_nonneg.mpr (sq_nonneg ‖ContinuousLinearMap.adjoint (↑Lε.symm : (E m) →L[ℝ] (E m))‖))
  rw [← mul_assoc, inv_mul_cancel₀ (pow_pos hBpos 2).ne', one_mul] at h3
  exact h3

/-- **The blind-gap bound from the noise floor:** `(Av)ᵀ S_y⁻¹ (Av) ≤ ‖Av‖²/γ` whenever
`γ‖ξ‖² ≤ ‖Lεᵀξ‖²` for all `ξ` — the paper's `t ≤ w‖Av‖²/λ_min(Γ)` step, at the constructed
data covariance of `DataCovarianceSqrt.lean` (`S_y ⪰ Γ ⪰ γ·I`, so `S_y⁻¹ ⪯ γ⁻¹·I` on `Av`,
proved variationally through the self-adjoint square root: no eigenvalue API). -/
theorem blindGap_le_noise_floor {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) {γ : ℝ} (hγ : 0 < γ)
    (hΓ : ∀ ξ : E m, γ * ‖ξ‖ ^ 2
      ≤ ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) ξ‖ ^ 2)
    (v : EuclideanSpace ℝ (Fin d)) :
    blindGap Aop Lx Lε v ≤ ‖Aop v‖ ^ 2 / γ := by
  rw [blindGap_eq_norm_sq]
  have hsa := dataCovSqrt_isSelfAdjoint (Aop ∘L Lx) Lε
  -- `‖Lyᵀ z‖ = ‖Ly z‖ = ‖Av‖` at `z = Ly⁻¹(Av)`, by self-adjointness
  have h3 : ‖ContinuousLinearMap.adjoint
        (↑(dataCovSqrt (Aop ∘L Lx) Lε) : (E m) →L[ℝ] (E m))
        ((dataCovSqrt (Aop ∘L Lx) Lε).symm (Aop v))‖ = ‖Aop v‖ := by
    rw [hsa.adjoint_eq]
    exact congrArg norm ((dataCovSqrt (Aop ∘L Lx) Lε).apply_symm_apply (Aop v))
  have h3sq : ‖ContinuousLinearMap.adjoint
        (↑(dataCovSqrt (Aop ∘L Lx) Lε) : (E m) →L[ℝ] (E m))
        ((dataCovSqrt (Aop ∘L Lx) Lε).symm (Aop v))‖ ^ 2 = ‖Aop v‖ ^ 2 := by
    rw [h3]
  -- `‖Av‖² = ‖(A Lx)ᵀ z‖² + ‖Lεᵀ z‖² ≥ γ‖z‖²`
  have h2 := dataCovSqrt_spec (Aop ∘L Lx) Lε ((dataCovSqrt (Aop ∘L Lx) Lε).symm (Aop v))
  rw [h3sq] at h2
  have h1 := hΓ ((dataCovSqrt (Aop ∘L Lx) Lε).symm (Aop v))
  have h4 : (0 : ℝ) ≤ ‖ContinuousLinearMap.adjoint (Aop ∘L Lx)
      ((dataCovSqrt (Aop ∘L Lx) Lε).symm (Aop v))‖ ^ 2 := sq_nonneg _
  rw [le_div_iff₀ hγ, mul_comm]
  linarith

/-! ### (2b) The survey price diverges -/

/-- **The survey crossover diverges as the direction goes blind:** for a fixed misreport `w > 0`
and noise floor `γ > 0`, the guaranteed-indistinguishable survey budget `2·(γ/(w·‖Av‖²))²` of
the checked spread crossover tends to `∞` as the illumination `‖Av‖² → 0⁺`.  The exchange rate
is a divergence, not a constant. -/
theorem survey_price_diverges {w γ : ℝ} (hw : 0 < w) (hγ : 0 < γ) :
    Tendsto (fun illum : ℝ => 2 * (γ / (w * illum)) ^ 2)
      (nhdsWithin (0 : ℝ) (Set.Ioi 0)) atTop := by
  have h2 : Tendsto (fun illum : ℝ => illum⁻¹)
      (nhdsWithin (0 : ℝ) (Set.Ioi 0)) atTop := tendsto_inv_nhdsGT_zero
  have h3 : Tendsto (fun illum : ℝ => (γ / w) * illum⁻¹)
      (nhdsWithin (0 : ℝ) (Set.Ioi 0)) atTop :=
    h2.const_mul_atTop (by positivity)
  have h4 : Tendsto (fun illum : ℝ => γ / (w * illum))
      (nhdsWithin (0 : ℝ) (Set.Ioi 0)) atTop :=
    h3.congr fun x => by ring
  have h5 := h4.atTop_mul_atTop₀ h4
  have h6 := h5.const_mul_atTop (show (0 : ℝ) < 2 by norm_num)
  exact h6.congr fun x => by ring

/-- **The mean-misreport survey price also diverges** (the supplementary moment, via
`MeanKL.lean`'s checked `n < 1/β` crossover): with the noise floor `γ`, whenever
`n·(w²‖Av‖²/γ) < 1` no test on `n` surveys separates a `w·v` blind-mean misreport — a budget
diverging as the inverse FIRST power of the illumination. -/
theorem survey_price_mean {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d)) (w : ℝ)
    (Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m)) {γ : ℝ} (hγ : 0 < γ)
    (hΓ : ∀ ξ : E m, γ * ‖ξ‖ ^ 2
      ≤ ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) ξ‖ ^ 2)
    (n : ℕ) (hn : (n : ℝ) * (w ^ 2 * ‖Aop v‖ ^ 2 / γ) < 1)
    {A : Set (Fin n → E m)} (hA : MeasurableSet A) :
    1 / 2 < (Measure.pi fun _ : Fin n =>
            modelLaw Aop (μx + w • v) Lx (↑Lε : (E m) →L[ℝ] (E m))).real A
        + (Measure.pi fun _ : Fin n =>
            modelLaw Aop μx Lx (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ := by
  refine model_mean_kl_crossover_closed Aop μx v w Lx Lε n ?_ hA
  have hble := blindGap_le_noise_floor Aop Lx Lε hγ hΓ v
  have h1 : w ^ 2 * blindGap Aop Lx Lε v ≤ w ^ 2 * ‖Aop v‖ ^ 2 / γ := by
    rw [mul_div_assoc]
    exact mul_le_mul_of_nonneg_left hble (sq_nonneg w)
  calc (n : ℝ) * (w ^ 2 * blindGap Aop Lx Lε v)
      ≤ (n : ℝ) * (w ^ 2 * ‖Aop v‖ ^ 2 / γ) :=
        mul_le_mul_of_nonneg_left h1 (Nat.cast_nonneg n)
    _ < 1 := hn

/-! ### (3) The reference price does not involve the operator -/

/-- The audit's null law, written WITH a forward operator in scope (as a deployment would carry
one).  The definition provably never reads it: see `audit_price_operator_free`. -/
noncomputable def surveyedAuditNullLaw {d m : ℕ}
    (_Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (k b : ℕ) : Measure ℝ :=
  (Measure.pi fun _ : Fin k => stdGaussian (EuclideanSpace ℝ (Fin b))).map
    AuditPrice.auditStat

/-- The audit's alternative law (blind spread inflated by `r` throughout the subspace), with the
operator in scope. -/
noncomputable def surveyedAuditAltLaw {d m : ℕ}
    (_Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (k b : ℕ) (r : ℝ) : Measure ℝ :=
  (Measure.pi fun _ : Fin k =>
      multivariateGaussian (0 : EuclideanSpace ℝ (Fin b)) (r ^ 2 • 1)).map
    AuditPrice.auditStat

/-- The audit's rejection probability at threshold `q` against the `r`-inflated alternative,
with the operator in scope. -/
noncomputable def surveyedAuditPower {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (k b : ℕ) (r q : ℝ) : ℝ≥0∞ :=
  surveyedAuditAltLaw Aop k b r (Set.Ioi q)

/-- **The reference price as a function of `(k, b, r, q)` alone** — no operator argument
exists to pass. -/
noncomputable def referencePrice (k b : ℕ) (r q : ℝ) : ℝ≥0∞ :=
  AuditPrice.chiSq (k * b) (Set.Ioi (q / r ^ 2))

/-- **The audit does not involve `A` — as a theorem, not an observation.**  For two ARBITRARY
forward operators, of arbitrary (even different) dimensions, the audit's null law, alternative
law and rejection probability coincide, definitionally (`rfl`): the reference price is the same
whether the direction is faintly illuminated or exactly blind, because the operator never
enters. -/
theorem audit_price_operator_free {d₁ m₁ d₂ m₂ : ℕ}
    (A₁ : EuclideanSpace ℝ (Fin d₁) →L[ℝ] E m₁)
    (A₂ : EuclideanSpace ℝ (Fin d₂) →L[ℝ] E m₂) (k b : ℕ) (r q : ℝ) :
    surveyedAuditNullLaw A₁ k b = surveyedAuditNullLaw A₂ k b
      ∧ surveyedAuditAltLaw A₁ k b r = surveyedAuditAltLaw A₂ k b r
      ∧ surveyedAuditPower A₁ k b r q = surveyedAuditPower A₂ k b r q :=
  ⟨rfl, rfl, rfl⟩

/-- The audit's null law is `χ²(kb)`, whatever the operator (re-export of
`AuditPrice.auditStat_null_law`). -/
theorem surveyedAuditNullLaw_eq_chiSq {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (k b : ℕ) :
    surveyedAuditNullLaw Aop k b = AuditPrice.chiSq (k * b) :=
  AuditPrice.auditStat_null_law k b

/-- **The power factors through the operator-free reference price:**
`surveyedAuditPower A k b r q = referencePrice k b r q = χ²_{kb}(Ioi (q/r²))` (re-export of
`AuditPrice.audit_power`). -/
theorem surveyedAuditPower_factors {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (k b : ℕ) {r : ℝ} (hr : 0 < r) (q : ℝ) :
    surveyedAuditPower Aop k b r q = referencePrice k b r q :=
  AuditPrice.audit_power k b hr q

/-- The audit's level over the composite null, whatever the operator (re-export of
`AuditPrice.audit_level`). -/
theorem surveyedAuditLevel {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (k b : ℕ) {r q : ℝ}
    (hr : 0 < r) (hr1 : r ≤ 1) (hq : 0 ≤ q) :
    surveyedAuditPower Aop k b r q ≤ AuditPrice.chiSq (k * b) (Set.Ioi q) :=
  AuditPrice.audit_level k b hr hr1 hq

/-- **The reference price at the physical witness** (one-dimensional blind fiber, the rank-one
case of the remark): `k` references whose true blind fiber variance is `s⋆² + w` (the witness
`Σ⋆ + w·vvᵀ` along `v`), audited against the reported `s⋆²` through the reported
standardization `W = s⋆⁻¹`, reject at threshold `q` with probability
`χ²_k(Ioi (q·s⋆²/(s⋆²+w)))` — an expression in `(k, s⋆, w, q)` with no operator. -/
theorem reference_price_pipeline (k : ℕ) (mB : EuclideanSpace ℝ (Fin 1)) {w s : ℝ}
    (hw : 0 ≤ w) (hs : 0 < s) (q : ℝ) :
    ((Measure.pi fun _ : Fin k =>
        multivariateGaussian mB ((s ^ 2 + w) • (1 : Matrix (Fin 1) (Fin 1) ℝ))).map
        (fun x => AuditPrice.auditStat fun i =>
          toEuclideanCLM (𝕜 := ℝ) (s⁻¹ • (1 : Matrix (Fin 1) (Fin 1) ℝ)) (x i - mB)))
        (Set.Ioi q)
      = AuditPrice.chiSq k (Set.Ioi (q * s ^ 2 / (s ^ 2 + w))) := by
  have hrpos : 0 < spreadRatio w s := spreadRatio_pos hw s
  have hS : (s ^ 2 • (1 : Matrix (Fin 1) (Fin 1) ℝ)).PosSemidef :=
    Matrix.PosSemidef.one.smul (sq_nonneg s)
  have hsqrt1 : CFC.sqrt (1 : Matrix (Fin 1) (Fin 1) ℝ) = 1 :=
    CFC.sqrt_unique (one_mul 1) Matrix.PosSemidef.one.nonneg
  have hsqrtS : CFC.sqrt (s ^ 2 • (1 : Matrix (Fin 1) (Fin 1) ℝ)) = s • 1 := by
    rw [AuditPrice.sqrt_sq_smul hs.le Matrix.PosSemidef.one, hsqrt1]
  have hW : (s⁻¹ • (1 : Matrix (Fin 1) (Fin 1) ℝ))
      * CFC.sqrt (s ^ 2 • (1 : Matrix (Fin 1) (Fin 1) ℝ)) = 1 := by
    rw [hsqrtS, Matrix.smul_mul, Matrix.mul_smul, smul_smul, one_mul,
      inv_mul_cancel₀ hs.ne', one_smul]
  have hcov : (s ^ 2 + w) • (1 : Matrix (Fin 1) (Fin 1) ℝ)
      = spreadRatio w s ^ 2 • (s ^ 2 • (1 : Matrix (Fin 1) (Fin 1) ℝ)) := by
    rw [smul_smul, spreadRatio_sq_mul hw hs.ne']
  have hs2 : (s : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 hs.ne'
  have h1 : (s ^ 2 + w) / s ^ 2 = 1 + w / s ^ 2 := by
    rw [add_div, div_self hs2]
  have harg : q / spreadRatio w s ^ 2 = q * s ^ 2 / (s ^ 2 + w) := by
    rw [spreadRatio_sq hw, ← h1, div_div_eq_mul_div]
  rw [hcov, AuditPrice.audit_power_pipeline k 1 mB (s ^ 2 • (1 : Matrix (Fin 1) (Fin 1) ℝ))
    (s⁻¹ • (1 : Matrix (Fin 1) (Fin 1) ℝ)) hS hW hrpos q, mul_one, harg]

/-! ### (4) CAPSTONE: the exchange rate is a divergence -/

/-- **The exchange rate between surveys and references (Remark `r:exchange`), assembled.**
For the physical record laws `y = Ax + ε` (Gaussian truth with invertible root `Lx`, invertible
noise root `Lε` with variational noise floor `γ ≤ λ_min(Γ)`), there exists a root `Lx'` of the
witness-bumped truth covariance `Σ⋆ + w·vvᵀ` such that:

(i) **surveys, below the diverging crossover, buy nothing**: whenever
`n·(w‖Av‖²/γ)² < 2` — in verbatim `n⋆` form, `n < 2·(γ/(w‖Av‖²))²`, which diverges as the
inverse square of the illumination — every test on `n` iid surveys has summed type-I + type-II
error strictly above `1/2` (reusing `SpreadKL.model_spread_kl_crossover_closed`);

(ii) **on a genuine kernel the price is infinite**: at `Av = 0` the two record laws are EQUAL
(the checked per-survey KL is `0`; converse Gibbs), so at EVERY survey count every test has
summed error at least `1`;

(iii) **references pay a fixed, operator-free price**: the `k`-reference audit at the same
witness — spread ratio `spreadRatio w s⋆ = √(1+w/s⋆²)` on the one-dimensional blind fiber with
reported deviation `s⋆` — rejects with probability exactly `χ²_k(Ioi (q·s⋆²/(s⋆²+w)))`, an
expression in `(k, s⋆, w, q)` alone, the same whether the direction is faintly illuminated or
exactly blind. -/
theorem exchange_rate {d m : ℕ}
    (Aop : EuclideanSpace ℝ (Fin d) →L[ℝ] E m) (μx v : EuclideanSpace ℝ (Fin d))
    {w s γ : ℝ} (hw : 0 ≤ w) (hs : 0 < s) (hγ : 0 < γ)
    (Lx : EuclideanSpace ℝ (Fin d) ≃L[ℝ] EuclideanSpace ℝ (Fin d))
    (Lε : (E m) ≃L[ℝ] (E m))
    (hΓ : ∀ ξ : E m, γ * ‖ξ‖ ^ 2
      ≤ ‖ContinuousLinearMap.adjoint (↑Lε : (E m) →L[ℝ] (E m)) ξ‖ ^ 2) :
    ∃ Lx' : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d),
      (∀ η : EuclideanSpace ℝ (Fin d),
        ‖ContinuousLinearMap.adjoint Lx' η‖ ^ 2
          = ‖ContinuousLinearMap.adjoint
              (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) η‖ ^ 2
            + w * (inner ℝ v η) ^ 2) ∧
      (∀ n : ℕ, (n : ℝ) * (w * ‖Aop v‖ ^ 2 / γ) ^ 2 < 2 →
        ∀ {A : Set (Fin n → E m)}, MeasurableSet A →
        1 / 2 < (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))).real A
            + (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx
                  (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
                  (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ) ∧
      (0 < w → Aop v ≠ 0 → ∀ n : ℕ, (n : ℝ) < 2 * (γ / (w * ‖Aop v‖ ^ 2)) ^ 2 →
        ∀ {A : Set (Fin n → E m)}, MeasurableSet A →
        1 / 2 < (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))).real A
            + (Measure.pi fun _ : Fin n =>
                modelLaw Aop μx
                  (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
                  (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ) ∧
      (Aop v = 0 →
        modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))
          = modelLaw Aop μx
              (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
              (↑Lε : (E m) →L[ℝ] (E m)) ∧
        ∀ (n : ℕ) {A : Set (Fin n → E m)}, MeasurableSet A →
          1 ≤ (Measure.pi fun _ : Fin n =>
                  modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))).real A
              + (Measure.pi fun _ : Fin n =>
                  modelLaw Aop μx
                    (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
                    (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ) ∧
      (∀ (k : ℕ) (q : ℝ),
        surveyedAuditPower Aop k 1 (spreadRatio w s) q
          = AuditPrice.chiSq k (Set.Ioi (q * s ^ 2 / (s ^ 2 + w)))) := by
  obtain ⟨Lx', hbump, hkl, _hklpi, hcross⟩ :=
    model_spread_kl_crossover_closed Aop μx v hw Lx Lε
  have hble := blindGap_le_noise_floor Aop
    (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε hγ hΓ v
  have hbg0 := blindGap_nonneg Aop
    (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε v
  -- the survey clause, proved once
  have hsurvey : ∀ n : ℕ, (n : ℝ) * (w * ‖Aop v‖ ^ 2 / γ) ^ 2 < 2 →
      ∀ {A : Set (Fin n → E m)}, MeasurableSet A →
      1 / 2 < (Measure.pi fun _ : Fin n =>
              modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m))).real A
          + (Measure.pi fun _ : Fin n =>
              modelLaw Aop μx
                (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
                (↑Lε : (E m) →L[ℝ] (E m))).real Aᶜ := by
    intro n hn A hA
    refine hcross n ?_ hA
    have hwb : w * blindGap Aop
        (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε v
        ≤ w * ‖Aop v‖ ^ 2 / γ := by
      rw [mul_div_assoc]
      exact mul_le_mul_of_nonneg_left hble hw
    have hwb0 : 0 ≤ w * blindGap Aop
        (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε v :=
      mul_nonneg hw hbg0
    calc (n : ℝ) * (w * blindGap Aop
          (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε v) ^ 2
        ≤ (n : ℝ) * (w * ‖Aop v‖ ^ 2 / γ) ^ 2 :=
          mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hwb0 hwb 2) (Nat.cast_nonneg n)
      _ < 2 := hn
  refine ⟨Lx', hbump, hsurvey, ?_, ?_, ?_⟩
  · -- the verbatim `n⋆` form: `n < 2·(γ/(w‖Av‖²))²`
    intro hw' hAv n hn A hA
    have hnv : 0 < ‖Aop v‖ := norm_pos_iff.mpr hAv
    have hx : 0 < w * ‖Aop v‖ ^ 2 / γ := div_pos (mul_pos hw' (pow_pos hnv 2)) hγ
    refine hsurvey n ?_ hA
    have hinv : γ / (w * ‖Aop v‖ ^ 2) = (w * ‖Aop v‖ ^ 2 / γ)⁻¹ := (inv_div _ _).symm
    rw [hinv] at hn
    calc (n : ℝ) * (w * ‖Aop v‖ ^ 2 / γ) ^ 2
        < 2 * ((w * ‖Aop v‖ ^ 2 / γ)⁻¹) ^ 2 * (w * ‖Aop v‖ ^ 2 / γ) ^ 2 :=
          mul_lt_mul_of_pos_right hn (pow_pos hx 2)
      _ = 2 := by
          rw [mul_assoc, ← mul_pow, inv_mul_cancel₀ hx.ne', one_pow, mul_one]
  · -- the kernel endpoint: the record laws coincide, at every `n`
    intro hAv
    have hbgz : blindGap Aop
        (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε v = 0 :=
      blindGap_eq_zero_of_blind Aop
        (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) Lε hAv
    have hprob : ∀ Lx0 : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d),
        IsProbabilityMeasure (modelLaw Aop μx Lx0 (↑Lε : (E m) →L[ℝ] (E m))) := by
      intro Lx0
      rw [modelLaw_repr_closed Aop μx Lx0 Lε]
      exact isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))
    haveI h₁ := hprob Lx'
    haveI h₂ := hprob (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    have hkl0 : klDiv (modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m)))
        (modelLaw Aop μx
          (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
          (↑Lε : (E m) →L[ℝ] (E m))) = 0 := by
      rw [hkl, hbgz, mul_zero, add_zero, Real.log_one, sub_zero, zero_div,
        ENNReal.ofReal_zero]
    have heq := klDiv_eq_zero_iff.mp hkl0
    refine ⟨heq, fun n A hA => ?_⟩
    have hpieq : (Measure.pi fun _ : Fin n =>
          modelLaw Aop μx Lx' (↑Lε : (E m) →L[ℝ] (E m)))
        = Measure.pi fun _ : Fin n =>
            modelLaw Aop μx
              (↑Lx : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
              (↑Lε : (E m) →L[ℝ] (E m)) := by rw [heq]
    exact lecam_exact hpieq hA
  · -- the reference price at the same witness, operator-free
    intro k q
    have hs2 : (s : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 hs.ne'
    have h1 : (s ^ 2 + w) / s ^ 2 = 1 + w / s ^ 2 := by
      rw [add_div, div_self hs2]
    have harg : q / spreadRatio w s ^ 2 = q * s ^ 2 / (s ^ 2 + w) := by
      rw [spreadRatio_sq hw, ← h1, div_div_eq_mul_div]
    calc surveyedAuditPower Aop k 1 (spreadRatio w s) q
        = AuditPrice.chiSq (k * 1) (Set.Ioi (q / spreadRatio w s ^ 2)) :=
          AuditPrice.audit_power k 1 (spreadRatio_pos hw s) q
      _ = AuditPrice.chiSq k (Set.Ioi (q * s ^ 2 / (s ^ 2 + w))) := by
          rw [mul_one, harg]

end ExchangeRate

-- #print axioms audit (only propext / Classical.choice / Quot.sound expected)
#print axioms ExchangeRate.spreadRatio_sq
#print axioms ExchangeRate.one_le_spreadRatio
#print axioms ExchangeRate.spreadRatio_pos
#print axioms ExchangeRate.spreadRatio_sq_mul
#print axioms ExchangeRate.spreadRatio_witness
#print axioms ExchangeRate.spreadRatio_strictMonoOn
#print axioms ExchangeRate.spreadRatio_bijOn
#print axioms ExchangeRate.exists_noise_floor
#print axioms ExchangeRate.blindGap_le_noise_floor
#print axioms ExchangeRate.survey_price_diverges
#print axioms ExchangeRate.survey_price_mean
#print axioms ExchangeRate.audit_price_operator_free
#print axioms ExchangeRate.surveyedAuditNullLaw_eq_chiSq
#print axioms ExchangeRate.surveyedAuditPower_factors
#print axioms ExchangeRate.surveyedAuditLevel
#print axioms ExchangeRate.reference_price_pipeline
#print axioms ExchangeRate.exchange_rate
