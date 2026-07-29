import Mathlib.Probability.Kernel.Posterior
import Mathlib.InformationTheory.KullbackLeibler.Basic

/-!
# Theorem 1 of the paper (`p:law`): the EM-step identity

This file machine-checks **Theorem 1** ("A legacy archive is one guess-then-correct step from the
regularizer", label `p:law`) of the paper *Priors learned from legacy
reconstructions inherit undetectable overconfidence*.

## Dictionary (paper ↔ mathlib)

The paper writes parameters `x` and data `y`; mathlib's `Probability.Kernel.Posterior` writes
parameters `Ω` and data `𝓧`.  We follow mathlib's variable names but keep the paper's roles:

| paper                              | here                                            |
| ---------------------------------- | ----------------------------------------------- |
| parameter `x`                      | `Ω`                                             |
| data `y`                           | `𝓧`                                             |
| likelihood `p(y ∣ x)`              | Markov kernel `κ : Kernel Ω 𝓧`                  |
| regularizer / prior `reg(x)`       | probability measure `μ : Measure Ω`             |
| truth `truth(x)`                   | probability measure `ν : Measure Ω`             |
| reg's data marginal `reg(y)`       | `κ ∘ₘ μ`                                         |
| TRUE data marginal `truth(y)`      | `κ ∘ₘ ν`                                         |
| posterior `reg(x ∣ y)`             | posterior kernel `κ†μ` (mathlib's Bayes inverse)|
| `p(y∣x)/reg(y)`                    | `κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y`       |

The curated-prior map of eq. (curated),
`T[reg](x) := E_{y ∼ truth(y)}[ reg(x ∣ y) ]`, is the posterior kernel `κ†μ` averaged over the
TRUE data marginal `κ ∘ₘ ν`, i.e. `(κ†μ) ∘ₘ (κ ∘ₘ ν)` — see `curatedPrior`.

## Part A (the identity, `curatedPrior_eq_withDensity` / `pLaw_partA`)

`T[reg] = reg · E_{y∼truth}[ p(y∣x)/reg(y) ]`, i.e. the population average of the Bayesian
posterior over real measurements equals the multiplicative NPMLE/Vardi–Lucy–Richardson reweight of
the regularizer.  Measure-level: `(κ†μ) ∘ₘ (κ ∘ₘ ν) = μ.withDensity (emReweight)`.

Two absolute-continuity hypotheses are stated explicitly, exactly the regularity the paper's formula
silently assumes when it divides by `reg(y)` (i.e. `reg` has full support on the data):
* `h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ`  — each likelihood dominated by reg's data marginal (this is the
  hypothesis of mathlib's Bayes theorem `posterior_eq_withDensity`);
* `h_ac  : κ ∘ₘ ν ≪ κ ∘ₘ μ`        — truth's data marginal dominated by reg's data marginal, so that
  `E_{y∼truth}[p(y∣x)/reg(y)]` is well-defined.

## Standing hypotheses

Standard-Borel parameter space, a Markov likelihood kernel, probability prior and truth — the
faithful population setting of the paper.  Nothing is trivialized: the identity is proved at the
level of measures via mathlib's posterior (Bayes-inverse) kernel and Tonelli.

## Main results

* `curatedPrior_eq_withDensity` / `pLaw_partA` — Part A (the EM-step identity), fully proved.
* `pLaw_partB` — Part B (the marginal-likelihood ascent, eq. `ascent`), fully proved:
  `klDiv (κ ∘ₘ ν) (κ ∘ₘ curatedPrior) ≤ klDiv (κ ∘ₘ ν) (κ ∘ₘ μ)`, the EM/NPMLE monotonicity.

Both parts are machine-checked; `#print axioms` lists only `propext`, `Classical.choice`,
`Quot.sound` (no `sorryAx`).
-/

open MeasureTheory ProbabilityTheory InformationTheory Real
open scoped ENNReal

namespace PriorLaundering

variable {Ω 𝓧 : Type*} {mΩ : MeasurableSpace Ω} {m𝓧 : MeasurableSpace 𝓧}
  [StandardBorelSpace Ω] [Nonempty Ω]
  [MeasurableSpace.CountableOrCountablyGenerated Ω 𝓧]
  (κ : Kernel Ω 𝓧) [IsMarkovKernel κ]
  (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]

/-- The NPMLE / EM multiplicative reweight
`g_reg(x) = E_{y ∼ truth(y)}[ p(y∣x) / reg(y) ]`.
Here `p(y∣x)/reg(y)` is the Radon–Nikodym derivative of the likelihood `κ x` with respect to the
regularizer's data marginal `κ ∘ₘ μ`, and the average is over the TRUE data marginal `κ ∘ₘ ν`. -/
noncomputable def emReweight : Ω → ℝ≥0∞ :=
  fun x => ∫⁻ y, κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y ∂(κ ∘ₘ ν)

/-- The **curated prior** / data-averaged-posterior map `T[reg] = E_{y ∼ truth(y)}[ reg(· ∣ y) ]`
of eq. (curated): the `reg`-posterior kernel `κ†μ` averaged over the true data marginal `κ ∘ₘ ν`. -/
noncomputable def curatedPrior : Measure Ω := (κ†μ) ∘ₘ (κ ∘ₘ ν)

/-- **Theorem 1, Part A — the EM-step identity**, stated with all objects inline.

`E_{y∼truth}[ reg(·∣y) ]  =  reg · E_{y∼truth}[ p(y∣x)/reg(y) ]`.

The population average of the Bayesian posterior over the real measurements equals the
multiplicative NPMLE / Vardi–Lucy–Richardson reweight of the regularizer. -/
theorem curatedPrior_eq_withDensity
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ) :
    (κ†μ) ∘ₘ (κ ∘ₘ ν)
      = μ.withDensity (fun x => ∫⁻ y, κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y ∂(κ ∘ₘ ν)) := by
  -- Bayes' theorem (mathlib `posterior_eq_withDensity`): the posterior is a `withDensity` of the
  -- likelihood's RN-derivative, `κ ∘ₘ μ`-a.e.  Transfer to `κ ∘ₘ ν`-a.e. via `h_ac`.
  have hpost : ∀ᵐ y ∂(κ ∘ₘ ν),
      (κ†μ) y = μ.withDensity (fun x => κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y) :=
    (posterior_eq_withDensity h_dom).filter_mono h_ac.ae_le
  -- Joint measurability of the RN-derivative (for Tonelli).
  have hswap : Measurable
      (Function.uncurry fun (y : 𝓧) (x : Ω) => κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y) :=
    (Kernel.measurable_rnDeriv κ (Kernel.const Ω (κ ∘ₘ μ))).comp measurable_swap
  ext s hs
  -- Unfold the outer composition `(κ†μ) ∘ₘ (κ ∘ₘ ν) = Measure.bind (κ ∘ₘ ν) (κ†μ)`.
  rw [Measure.bind_apply hs (κ†μ).aemeasurable]
  -- Rewrite the posterior mass by Bayes, then `withDensity_apply`.
  have hstep : (fun y => (κ†μ) y s)
      =ᵐ[κ ∘ₘ ν] fun y => ∫⁻ x in s, κ.rnDeriv (Kernel.const Ω (κ ∘ₘ μ)) x y ∂μ := by
    filter_upwards [hpost] with y hy
    rw [hy, withDensity_apply _ hs]
  rw [lintegral_congr_ae hstep]
  -- Tonelli: swap the outer `y`-average (over `κ ∘ₘ ν`) with the inner `x`-integral (over `μ|s`).
  rw [lintegral_lintegral_swap hswap.aemeasurable, withDensity_apply _ hs]

/-- **Theorem 1, Part A**, packaged with the paper-matching definitions:
`curatedPrior = reg.withDensity (emReweight)`, i.e. the data-averaged posterior equals the
NPMLE reweight of the regularizer. -/
theorem pLaw_partA
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ) :
    curatedPrior κ μ ν = μ.withDensity (emReweight κ μ ν) :=
  curatedPrior_eq_withDensity κ μ ν h_dom h_ac

/-!
## Part B — the marginal-likelihood ascent (eq. `ascent`), fully proved

Part B of `p:law` says the EM step does not decrease the data-marginal log-likelihood
`ℓ(π) = E_{y∼truth}[ log (κ ∘ₘ π)(y) ]`, equivalently (eq. `ascent`, and the base-independent form we
prove)

    klDiv (truth(y)) (T[reg](y))  ≤  klDiv (truth(y)) (reg(y)),

i.e. `klDiv (κ ∘ₘ ν) (κ ∘ₘ curatedPrior) ≤ klDiv (κ ∘ₘ ν) (κ ∘ₘ μ)`
(`MarginalLikelihoodAscent`, proved in `pLaw_partB`).

The proof follows the classical EM/NPMLE monotonicity (Dempster–Laird–Rubin 1977; Csiszár–Tusnády
1984; Vardi 1985), realized measure-theoretically:

    ℓ(q) − ℓ(reg) = ∫ log( d(κ∘ₘq)/d(κ∘ₘreg) ) dtruth        [`hchain`, RN chain rule]
                  = ∫ log( E_{x∼reg(·|y)} g(x) ) dtruth       [`comp_curatedPrior_eq_withDensity`]
                  ≥ ∫ E_{x∼reg(·|y)}[ log g(x) ] dtruth        [`integral_log_le_log_integral`, Jensen]
                  = ∫ log g dq  =  ∫ llr(q‖reg) dq  ≥  0       [Tonelli + `..._klDiv_reg_integral_nonneg`]

where `g = emReweight = d q / d reg` (Part A).  The Jensen step is proved for `Real.log` via the
change-of-measure trick `log x ≤ x − 1` (`Real.log_le_sub_one_of_pos`), which sidesteps the
non-closed-domain issue of `ConcaveOn.le_map_integral`.

The two absolute-continuity conditions and the three llr-integrability conditions in `pLaw_partB`'s
signature (`hTmq`, `hIm`, `hImq`, `hIq`) are exactly the finiteness of the relevant KL divergences,
and `hg_pos` the positivity of the reweight (a full-support condition) — the regularity the paper
assumes implicitly.  Everything else is derived; the conclusion is exactly the paper's ascent, not a
weakening.

Also recorded: two faithful *components* that follow directly from Part A —
`curatedPrior_absolutelyContinuous` (the step reweights, adds no support) and
`curatedPrior_klDiv_reg_integral_nonneg` (the sharp EM lower bound `KL(q ‖ reg) ≥ 0`, which
`pLaw_partB` uses).
-/

/-- The curated prior is a probability measure (Markov posterior kernel composed with the
probability data marginal `κ ∘ₘ ν`). -/
instance isProbabilityMeasure_curatedPrior : IsProbabilityMeasure (curatedPrior κ μ ν) := by
  unfold curatedPrior; infer_instance

/-- **Verified component of Part B.** The curated prior only *reweights* the regularizer; it never
creates support that the regularizer lacks: `T[reg] ≪ reg`.  Immediate from Part A's `withDensity`
form.  This is the measure-level seed of the freeze — the step cannot put mass on configurations
`reg` rules out (in particular anywhere along the blind fibers). -/
theorem curatedPrior_absolutelyContinuous
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ) :
    curatedPrior κ μ ν ≪ μ := by
  rw [pLaw_partA κ μ ν h_dom h_ac]
  exact withDensity_absolutelyContinuous _ _

/-- **Verified component of Part B (Gibbs — the sharp EM lower bound is nonnegative).**
`KL(T[reg] ‖ reg) ≥ 0` in integral form, `0 ≤ ∫ log (d q / d reg) d q`.  The classical EM ascent
gives `ℓ(q) − ℓ(reg) ≥ KL(q ‖ reg)`; this lemma verifies the nonnegativity of that lower bound (with
equality iff the step changes nothing).  `Integrable (llr q reg) q` is an explicit hypothesis — the
standard finiteness condition making the divergence finite. -/
theorem curatedPrior_klDiv_reg_integral_nonneg
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ)
    (h_int : Integrable (llr (curatedPrior κ μ ν) μ) (curatedPrior κ μ ν)) :
    0 ≤ ∫ x, llr (curatedPrior κ μ ν) μ x ∂(curatedPrior κ μ ν) := by
  have hac := curatedPrior_absolutelyContinuous κ μ ν h_dom h_ac
  have h := integral_llr_add_sub_measure_univ_nonneg hac h_int
  simpa using h

/-- **Jensen's inequality for `Real.log`** against a probability measure, `∫ log h ≤ log ∫ h`.
Proved via the change-of-measure bound `log x ≤ x − 1` (`Real.log_le_sub_one_of_pos`), which avoids
the non-closed-domain (`Ioi 0`) obstruction to `ConcaveOn.le_map_integral`. -/
lemma integral_log_le_log_integral {α : Type*} {mα : MeasurableSpace α}
    (P : Measure α) [IsProbabilityMeasure P] {h : α → ℝ}
    (hh : Integrable h P) (hlog : Integrable (fun x => Real.log (h x)) P)
    (hpos : ∀ᵐ x ∂P, 0 < h x) (hc : 0 < ∫ x, h x ∂P) :
    ∫ x, Real.log (h x) ∂P ≤ Real.log (∫ x, h x ∂P) := by
  set c := ∫ x, h x ∂P with hcdef
  have hbound : Integrable (fun x => h x / c - 1 + Real.log c) P :=
    ((hh.div_const c).sub (integrable_const 1)).add (integrable_const _)
  have key : (fun x => Real.log (h x)) ≤ᵐ[P] fun x => h x / c - 1 + Real.log c := by
    filter_upwards [hpos] with x hx
    have h1 : Real.log (h x / c) ≤ h x / c - 1 := Real.log_le_sub_one_of_pos (div_pos hx hc)
    rw [Real.log_div hx.ne' hc.ne'] at h1
    linarith
  calc ∫ x, Real.log (h x) ∂P
      ≤ ∫ x, (h x / c - 1 + Real.log c) ∂P := integral_mono_ae hlog hbound key
    _ = ∫ x, (h x / c + (Real.log c - 1)) ∂P := by congr 1; funext x; ring
    _ = Real.log c := by
        rw [integral_add (hh.div_const c) (integrable_const _), integral_div, integral_const,
          ← hcdef]
        simp only [probReal_univ, smul_eq_mul, one_mul]
        rw [div_self hc.ne']
        ring

/-- The data marginal of the curated prior is the regularizer's data marginal reweighted by the
posterior-average of the reweight `g`; i.e. `d(κ ∘ₘ q)/d(κ ∘ₘ reg)(y) = E_{x∼reg(·|y)} g(x)`.
Uses the posterior property `(κ ∘ₘ μ) ⊗ₘ κ†μ = (μ ⊗ₘ κ).map swap` and Tonelli. -/
lemma comp_curatedPrior_eq_withDensity
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ) :
    κ ∘ₘ (curatedPrior κ μ ν)
      = (κ ∘ₘ μ).withDensity (fun y => ∫⁻ x, emReweight κ μ ν x ∂((κ†μ) y)) := by
  set g := emReweight κ μ ν with hgdef
  have hg_meas : Measurable g := by
    rw [hgdef]; unfold emReweight
    exact (Kernel.measurable_rnDeriv κ (Kernel.const Ω (κ ∘ₘ μ))).lintegral_kernel_prod_right'
      (κ := Kernel.const Ω (κ ∘ₘ ν))
  have hgB : Measurable (fun p : Ω × 𝓧 => g p.1) := hg_meas.comp measurable_fst
  have hgB2 : Measurable (fun p : 𝓧 × Ω => g p.2) := hg_meas.comp measurable_snd
  rw [pLaw_partA κ μ ν h_dom h_ac]
  ext B hB
  rw [Measure.bind_apply hB κ.aemeasurable,
      lintegral_withDensity_eq_lintegral_mul _ hg_meas (κ.measurable_coe hB),
      withDensity_apply _ hB]
  symm
  calc ∫⁻ y in B, (∫⁻ x, g x ∂(κ†μ) y) ∂(κ ∘ₘ μ)
      = ∫⁻ y in B, (∫⁻ x in Set.univ, g x ∂(κ†μ) y) ∂(κ ∘ₘ μ) := by
        simp_rw [Measure.restrict_univ]
    _ = ∫⁻ p in B ×ˢ Set.univ, g p.2 ∂((κ ∘ₘ μ) ⊗ₘ (κ†μ)) :=
        (Measure.setLIntegral_compProd hgB2 hB MeasurableSet.univ).symm
    _ = ∫⁻ p in B ×ˢ Set.univ, g p.2 ∂((μ ⊗ₘ κ).map Prod.swap) := by
        rw [compProd_posterior_eq_map_swap]
    _ = ∫⁻ p in Prod.swap ⁻¹' (B ×ˢ Set.univ), g (Prod.swap p).2 ∂(μ ⊗ₘ κ) :=
        setLIntegral_map (hB.prod MeasurableSet.univ) hgB2 measurable_swap
    _ = ∫⁻ p in Set.univ ×ˢ B, g p.1 ∂(μ ⊗ₘ κ) := by
        rw [Set.preimage_swap_prod]; simp only [Prod.snd_swap]
    _ = ∫⁻ x in Set.univ, (∫⁻ y in B, g x ∂κ x) ∂μ :=
        Measure.setLIntegral_compProd hgB MeasurableSet.univ hB
    _ = ∫⁻ x, (g * fun x => κ x B) x ∂μ := by
        rw [Measure.restrict_univ]; simp_rw [setLIntegral_const, Pi.mul_apply]

/-- The analytic core of the ascent: `∫_truth log( E_{x∼reg(·|y)} g(x) ) dy ≥ 0`
(Jensen inside, Tonelli to recognize the curated prior as the data-averaged posterior, then Gibbs).
The `Integrable` hypotheses are the finiteness conditions of the relevant divergences. -/
lemma partB_logF_integral_nonneg
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ)
    (hg_pos : ∀ᵐ x ∂μ, 0 < emReweight κ μ ν x)
    (hIq : Integrable (llr (curatedPrior κ μ ν) μ) (curatedPrior κ μ ν))
    (hIlogF : Integrable
      (fun y => Real.log ((∫⁻ x, emReweight κ μ ν x ∂((κ†μ) y)).toReal)) (κ ∘ₘ ν)) :
    0 ≤ ∫ y, Real.log ((∫⁻ x, emReweight κ μ ν x ∂((κ†μ) y)).toReal) ∂(κ ∘ₘ ν) := by
  set g := emReweight κ μ ν with hgdef
  set T := κ ∘ₘ ν with hTdef
  set F := fun y => ∫⁻ x, g x ∂((κ†μ) y) with hFdef
  set f := fun x => Real.log (g x).toReal with hfdef
  have hg_meas : Measurable g := by
    rw [hgdef]; unfold emReweight
    exact (Kernel.measurable_rnDeriv κ (Kernel.const Ω (κ ∘ₘ μ))).lintegral_kernel_prod_right'
      (κ := Kernel.const Ω (κ ∘ₘ ν))
  have hf_meas : Measurable f := (Real.measurable_log.comp (hg_meas.ennreal_toReal))
  have hgB2 : Measurable (fun p : 𝓧 × Ω => g p.2) := hg_meas.comp measurable_snd
  -- curated prior in bind form and as a withDensity of g
  have hqb : curatedPrior κ μ ν = (κ†μ) ∘ₘ T := rfl
  have hq_eq : curatedPrior κ μ ν = μ.withDensity g := pLaw_partA κ μ ν h_dom h_ac
  -- llr(q‖reg) = log g  a.e.-q, so `f` is `q`-integrable and `∫ f dq ≥ 0` (Gibbs)
  have hllr_eq : llr (curatedPrior κ μ ν) μ =ᵐ[curatedPrior κ μ ν] f := by
    have h1 : (curatedPrior κ μ ν).rnDeriv μ =ᵐ[μ] g := by
      rw [hq_eq]; exact Measure.rnDeriv_withDensity μ hg_meas
    have hac : curatedPrior κ μ ν ≪ μ := by
      rw [hq_eq]; exact withDensity_absolutelyContinuous _ _
    filter_upwards [hac.ae_le h1] with x hx
    rw [llr_def, hfdef]; simp only []; rw [hx]
  have hIq' : Integrable f (curatedPrior κ μ ν) := (integrable_congr hllr_eq).mp hIq
  have hf_gibbs : 0 ≤ ∫ x, f x ∂(curatedPrior κ μ ν) := by
    rw [← integral_congr_ae hllr_eq]
    exact curatedPrior_klDiv_reg_integral_nonneg κ μ ν h_dom h_ac hIq
  -- posterior a.c. a.e.-T
  have hpost_ac : ∀ᵐ y ∂T, (κ†μ) y ≪ μ :=
    h_ac.ae_le (absolutelyContinuous_posterior h_dom)
  -- F finite a.e.-T and positive a.e.-T
  have hF_meas : Measurable F := by
    rw [hFdef]
    exact hgB2.lintegral_kernel_prod_right' (κ := κ†μ)
  have hFint_one : ∫⁻ y, F y ∂(κ ∘ₘ μ) = 1 := by
    have hcw := comp_curatedPrior_eq_withDensity κ μ ν h_dom h_ac
    have h2 : (κ ∘ₘ curatedPrior κ μ ν) Set.univ = 1 := by simp
    rw [hcw, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ] at h2
    exact h2
  have hF_lt : ∀ᵐ y ∂T, F y < ∞ :=
    h_ac.ae_le (ae_lt_top hF_meas (by rw [hFint_one]; exact ENNReal.one_ne_top))
  have hF_pos : ∀ᵐ y ∂T, 0 < F y := by
    filter_upwards [hpost_ac] with y hy
    rw [hFdef]
    simp only []
    rw [pos_iff_ne_zero]
    intro hzero
    rw [lintegral_eq_zero_iff hg_meas] at hzero
    have hcontra : ∀ᵐ x ∂((κ†μ) y), (0 : ℝ≥0∞) < 0 := by
      filter_upwards [hy.ae_le hg_pos, hzero] with x hpos hz
      rwa [hz] at hpos
    exact absurd hcontra.exists (by simp)
  -- per-y integrabilities from the composition-integrability of `f` over `q = κ†μ ∘ₘ T`
  have hcomp := (Measure.integrable_comp_iff hf_meas.aestronglyMeasurable).mp (hqb ▸ hIq')
  have hlogg_int_post : ∀ᵐ y ∂T, Integrable f ((κ†μ) y) := hcomp.1
  have hg_int_post : ∀ᵐ y ∂T, Integrable (fun x => (g x).toReal) ((κ†μ) y) := by
    filter_upwards [hF_lt] with y hy
    exact integrable_toReal_of_lintegral_ne_top hg_meas.aemeasurable hy.ne
  have hgpos_post : ∀ᵐ y ∂T, ∀ᵐ x ∂((κ†μ) y), 0 < (g x).toReal := by
    filter_upwards [hpost_ac, hF_lt] with y hy hylt
    have hlt : ∀ᵐ x ∂((κ†μ) y), g x < ∞ := by
      have : ∫⁻ x, g x ∂((κ†μ) y) ≠ ∞ := hylt.ne
      exact ae_lt_top hg_meas this
    filter_upwards [hy.ae_le hg_pos, hlt] with x hpos hltx
    exact ENNReal.toReal_pos hpos.ne' hltx.ne
  -- per-y Jensen: ∫ log g dκ†μy ≤ log (F y).toReal
  have hjensen : ∀ᵐ y ∂T, (∫ x, f x ∂((κ†μ) y)) ≤ Real.log (F y).toReal := by
    filter_upwards [hlogg_int_post, hg_int_post, hgpos_post, hF_pos, hF_lt] with
      y hlog_int hg_int hgpos hFpos hFlt
    have hceq : ∫ x, (g x).toReal ∂((κ†μ) y) = (F y).toReal :=
      integral_toReal hg_meas.aemeasurable (ae_lt_top hg_meas hFlt.ne)
    have hc_pos : 0 < ∫ x, (g x).toReal ∂((κ†μ) y) := by
      rw [hceq]; exact ENNReal.toReal_pos hFpos.ne' hFlt.ne
    have := integral_log_le_log_integral ((κ†μ) y) hg_int hlog_int hgpos hc_pos
    rwa [hceq] at this
  -- Tonelli: ∫ f dq = ∫ y, ∫ x, f x dκ†μy dT
  have hsnd : (T ⊗ₘ (κ†μ)).snd = (κ†μ) ∘ₘ T := Measure.snd_compProd T (κ†μ)
  have hf2_int : Integrable (fun p : 𝓧 × Ω => f p.2) (T ⊗ₘ (κ†μ)) :=
    (Measure.integrable_compProd_snd_iff hf_meas.aestronglyMeasurable).mpr (hqb ▸ hIq')
  have htonelli : ∫ x, f x ∂(curatedPrior κ μ ν) = ∫ y, (∫ x, f x ∂((κ†μ) y)) ∂T := by
    rw [hqb, ← hsnd, Measure.snd, integral_map measurable_snd.aemeasurable
      hf_meas.aestronglyMeasurable, Measure.integral_compProd hf2_int]
  have hinner_int : Integrable (fun y => ∫ x, f x ∂((κ†μ) y)) T :=
    hf2_int.integral_compProd
  -- assemble
  calc (0 : ℝ) ≤ ∫ x, f x ∂(curatedPrior κ μ ν) := hf_gibbs
    _ = ∫ y, (∫ x, f x ∂((κ†μ) y)) ∂T := htonelli
    _ ≤ ∫ y, Real.log (F y).toReal ∂T :=
        integral_mono_ae hinner_int hIlogF hjensen

/-- **The exact statement of Part B** (marginal-likelihood ascent / eq. `ascent`): one EM step does
not increase the KL divergence of the true data marginal `truth(y) = κ ∘ₘ ν` from the model's data
marginal.  Proved in `pLaw_partB`. -/
def MarginalLikelihoodAscent : Prop :=
  klDiv (κ ∘ₘ ν) (κ ∘ₘ curatedPrior κ μ ν) ≤ klDiv (κ ∘ₘ ν) (κ ∘ₘ μ)

/-- **Theorem 1, Part B — the marginal-likelihood ascent (EM/NPMLE monotonicity, eq. `ascent`).**
One EM step does not increase the KL divergence of the truth's data marginal from the model's data
marginal: `klDiv (κ ∘ₘ ν) (κ ∘ₘ T[reg]) ≤ klDiv (κ ∘ₘ ν) (κ ∘ₘ reg)`.

Hypotheses (all honest regularity the paper assumes implicitly): `hTmq` and `h_ac` are the two
absolute-continuity conditions making the divergences well-posed; `hIm`, `hImq`, `hIq` are the
finiteness (llr-integrability) of the three KL divergences involved; `hg_pos` is positivity of the
reweight.  The conclusion is exactly the paper's ascent — nothing is weakened. -/
theorem pLaw_partB
    (h_dom : ∀ᵐ x ∂μ, κ x ≪ κ ∘ₘ μ) (h_ac : κ ∘ₘ ν ≪ κ ∘ₘ μ)
    (hTmq : κ ∘ₘ ν ≪ κ ∘ₘ curatedPrior κ μ ν)
    (hg_pos : ∀ᵐ x ∂μ, 0 < emReweight κ μ ν x)
    (hIm : Integrable (llr (κ ∘ₘ ν) (κ ∘ₘ μ)) (κ ∘ₘ ν))
    (hImq : Integrable (llr (κ ∘ₘ ν) (κ ∘ₘ curatedPrior κ μ ν)) (κ ∘ₘ ν))
    (hIq : Integrable (llr (curatedPrior κ μ ν) μ) (curatedPrior κ μ ν)) :
    MarginalLikelihoodAscent κ μ ν := by
  set g := emReweight κ μ ν with hgdef
  set T := κ ∘ₘ ν with hTdef
  set m := κ ∘ₘ μ with hmdef
  set mq := κ ∘ₘ curatedPrior κ μ ν with hmqdef
  set F := fun y => ∫⁻ x, g x ∂((κ†μ) y) with hFdef
  have hg_meas : Measurable g := by
    rw [hgdef]; unfold emReweight
    exact (Kernel.measurable_rnDeriv κ (Kernel.const Ω (κ ∘ₘ μ))).lintegral_kernel_prod_right'
      (κ := Kernel.const Ω (κ ∘ₘ ν))
  have hgB2 : Measurable (fun p : 𝓧 × Ω => g p.2) := hg_meas.comp measurable_snd
  have hF_meas : Measurable F := by rw [hFdef]; exact hgB2.lintegral_kernel_prod_right' (κ := κ†μ)
  -- `mq = m.withDensity F`, so `d mq / d m =ᵐ[m] F`
  have hmq_eq : mq = m.withDensity F := comp_curatedPrior_eq_withDensity κ μ ν h_dom h_ac
  have hrn_mq : mq.rnDeriv m =ᵐ[m] F := by
    rw [hmq_eq]; exact Measure.rnDeriv_withDensity m hF_meas
  -- F positive and finite a.e.-T (as in `partB_logF_integral_nonneg`)
  have hpost_ac : ∀ᵐ y ∂T, (κ†μ) y ≪ μ :=
    h_ac.ae_le (absolutelyContinuous_posterior h_dom)
  have hFint_one : ∫⁻ y, F y ∂m = 1 := by
    have hcw := comp_curatedPrior_eq_withDensity κ μ ν h_dom h_ac
    have h2 : (κ ∘ₘ curatedPrior κ μ ν) Set.univ = 1 := by simp
    rw [hcw, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ] at h2
    exact h2
  have hF_lt : ∀ᵐ y ∂T, F y < ∞ :=
    h_ac.ae_le (ae_lt_top hF_meas (by rw [hFint_one]; exact ENNReal.one_ne_top))
  have hF_pos : ∀ᵐ y ∂T, 0 < F y := by
    filter_upwards [hpost_ac] with y hy
    rw [hFdef]; simp only []; rw [pos_iff_ne_zero]
    intro hzero
    rw [lintegral_eq_zero_iff hg_meas] at hzero
    have hcontra : ∀ᵐ x ∂((κ†μ) y), (0 : ℝ≥0∞) < 0 := by
      filter_upwards [hy.ae_le hg_pos, hzero] with x hpos hz; rwa [hz] at hpos
    exact absurd hcontra.exists (by simp)
  -- the log-likelihood-ratio chain: `llr T m =ᵐ[T] llr T mq + log F`
  have hchain : llr T m =ᵐ[T] fun y => llr T mq y + Real.log (F y).toReal := by
    have h1 : T.rnDeriv mq * mq.rnDeriv m =ᵐ[m] T.rnDeriv m := Measure.rnDeriv_mul_rnDeriv hTmq
    have hpos_mq : ∀ᵐ y ∂T, 0 < T.rnDeriv mq y := Measure.rnDeriv_pos hTmq
    have hlt_mq : ∀ᵐ y ∂T, T.rnDeriv mq y < ∞ := hTmq.ae_le (Measure.rnDeriv_lt_top T mq)
    filter_upwards [h_ac.ae_le h1, h_ac.ae_le hrn_mq, hpos_mq, hlt_mq, hF_pos, hF_lt]
      with y he hrn hp hl hFp hFl
    show Real.log (T.rnDeriv m y).toReal
        = Real.log (T.rnDeriv mq y).toReal + Real.log (F y).toReal
    rw [← he, Pi.mul_apply, ENNReal.toReal_mul,
      Real.log_mul (ENNReal.toReal_pos hp.ne' hl.ne).ne'
        (by rw [hrn]; exact (ENNReal.toReal_pos hFp.ne' hFl.ne).ne'), hrn]
  -- integrability of `log F` under T
  have hIlogF : Integrable (fun y => Real.log (F y).toReal) T := by
    have heq : (fun y => Real.log (F y).toReal) =ᵐ[T] fun y => llr T m y - llr T mq y := by
      filter_upwards [hchain] with y hy; rw [hy]; ring
    exact (integrable_congr heq).mpr (hIm.sub hImq)
  -- integral split
  have hsplit : ∫ y, llr T m y ∂T = ∫ y, llr T mq y ∂T + ∫ y, Real.log (F y).toReal ∂T := by
    rw [← integral_add hImq hIlogF]; exact integral_congr_ae hchain
  -- the ascent term is nonnegative (Jensen + Tonelli + Gibbs)
  have hlogF_nonneg : 0 ≤ ∫ y, Real.log (F y).toReal ∂T :=
    partB_logF_integral_nonneg κ μ ν h_dom h_ac hg_pos hIq hIlogF
  have hle : ∫ y, llr T mq y ∂T ≤ ∫ y, llr T m y ∂T := by rw [hsplit]; linarith
  -- lift the real inequality to `klDiv`
  show klDiv T mq ≤ klDiv T m
  have hne_m : klDiv T m ≠ ⊤ := klDiv_ne_top h_ac hIm
  have hne_mq : klDiv T mq ≠ ⊤ := klDiv_ne_top hTmq hImq
  rw [← ENNReal.toReal_le_toReal hne_mq hne_m,
    toReal_klDiv_of_measure_eq hTmq (by simp), toReal_klDiv_of_measure_eq h_ac (by simp)]
  exact hle

end PriorLaundering

/-! ### Axiom audit

The doc comment above claims these rest only on Lean's three standard axioms. These commands
are what make that checkable rather than asserted: each must print `propext, Classical.choice,
Quot.sound` and never `sorryAx`. -/

#print axioms PriorLaundering.pLaw_partB
#print axioms PriorLaundering.comp_curatedPrior_eq_withDensity
