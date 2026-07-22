/-
# `p:nearnull` (i), definitional: the preamble's relations proved, not assumed

`NearNullComposed.precision_bound` states the near-null freeze bound in one theorem, but takes the
preamble's definitional relations (`hSq`, `hG`, `hGT`, `hGG`, `hpost`, `hQ`), the Woodbury
invertibility side conditions (`hpu`, `hsu`, `hqu`), and the variational middle-factor bound as
hypotheses.  This module closes them from the definitions themselves.

**The objects** (matching the paper's preamble and the proof of `p:nearnull` at ~line 654 of
`manuscript.tex`), for an operator `A`, noise covariance `Γ ≻ 0`, regularizer covariance
`Σ_ρ ≻ 0`, and truth covariance `Σ_⋆ ⪰ 0`:

    Σ_post = (Σ_ρ⁻¹ + Aᵀ Γ⁻¹ A)⁻¹          (posterior covariance)
    G      = Σ_post Aᵀ Γ⁻¹                  (posterior gain)
    S_ρ    = A Σ_ρ Aᵀ + Γ                   (the regularizer's data covariance, `Sdata`)
    S_y    = A Σ_⋆ Aᵀ + Γ                   (the true data covariance)
    Q      = S_y⁻¹ + (Γ⁻¹ − S_ρ⁻¹)
    Σ_q    = Σ_post + G S_y Gᵀ              (the curated covariance — the law of total covariance
                                             is *taken as the definition* of `Σ_q` here, so `hSq`
                                             is discharged by `rfl`, not re-proved)

**What is proved from the definitions** (part A1): `hG` and `hGT` (one line each from `G`'s
definition plus symmetry of `Σ_post`, `Γ`), `hpost` (`⁻¹⁻¹` on a positive definite matrix), and
`hGG` — the appendix's two-Woodbury computation `A Σ_post Aᵀ = Γ − Γ S_ρ⁻¹ Γ` (`a_spost_at`,
via mathlib's Woodbury identity `Matrix.add_mul_mul_inv_eq_sub` applied to `Σ_post` and the
algebraic identity `P − P S⁻¹ P = Γ − Γ S⁻¹ Γ` for `P = S − Γ`).  All invertibility side
conditions come from positive definiteness; in particular `Q ≻ 0` is obtained *without* any
Loewner inverse-antitonicity, from the identity `Γ⁻¹ − S_ρ⁻¹ = (Γ⁻¹A) Σ_post (Γ⁻¹A)ᵀ ⪰ 0`
(`gaminv_sub_sdatainv_eq`), which the paper's "`S_ρ ⪰ Γ` makes `Γ⁻¹ − S_ρ⁻¹ ⪰ 0`" step asserted
spectrally.

**The variational bounds** (part A2): from the spectral band in variational form —
`hGam_lb : ∀ w, γ (w·w) ≤ w·(Γw)` with `γ > 0` (i.e. `λ_min(Γ) ≥ γ`) and
`hSy_ub : ∀ w, w·(S_y w) ≤ s (w·w)` (i.e. `λ_max(S_y) ≤ s`) — the inverse bounds
`0 ≤ w·(Γ⁻¹w) ≤ γ⁻¹ (w·w)` (`inv_form_le`), the Loewner step `Q ⪰ S_y⁻¹` hence
`Q⁻¹ ⪯ S_y` (`inv_antitone_form`, proved variationally via the quadratic-form Cauchy–Schwarz
`form_cauchy_schwarz`, itself from `discrim_le_zero` — no spectral theorem), and the middle-factor
bound `Γ⁻¹ Q⁻¹ Γ⁻¹ ⪯ (s/γ²) I` (`middle_factor_form_le`).

**The capstone** (part A3, `precision_bound_spectral`): hypotheses ONLY `Γ ≻ 0`, `Σ_ρ ≻ 0`,
`Σ_⋆ ⪰ 0` and the band `(γ, s)`:

    |⟪v, (Σ_q⁻¹ − Σ_ρ⁻¹) v⟫| ≤ max(γ⁻¹, s/γ²) ⟪A v, A v⟫   for every `v`,

with the exact-kernel endpoint `A v = 0 → the form moves by exactly 0`
(`precision_zero_of_kernel_definitional`).  This is `p:nearnull` (i) with the paper's constant
`c ≤ λ_min(Γ)⁻¹ max{1, λ_max(S_y)/λ_min(Γ)}` instantiated through the band: `max(γ⁻¹, s/γ²) =
γ⁻¹ max{1, s/γ}`.

**Honesty.**  No `sorry`/`admit`.  The only hypotheses anywhere are the paper's standing
assumptions (`Γ ≻ 0`, `Σ_ρ ≻ 0`, `Σ_⋆ ⪰ 0`) and, for the quantitative constant, the two
variational band inequalities defining `γ` and `s` — nothing else remains.  `#print axioms` on
every public result at the end.
-/
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Algebra.QuadraticDiscriminant
import NearNullComposed

namespace NearNullDefinitional

open Matrix

/-! ## Real-carrier helpers: `star` is trivial, Hermitian is symmetric, sandwiches are PSD -/

/-- `star` is trivial on real vectors. -/
private lemma star_real_vec {ι : Type*} (v : ι → ℝ) : star v = v := by
  funext i; simp

/-- The quadratic form of a real positive semidefinite matrix is nonnegative, in the
`⬝ᵥ`-carrier the composition lives in. -/
theorem psd_form_nonneg {ι : Type*} [Fintype ι] {M : Matrix ι ι ℝ} (hM : M.PosSemidef)
    (w : ι → ℝ) : 0 ≤ w ⬝ᵥ (M *ᵥ w) := by
  have h := hM.dotProduct_mulVec_nonneg w
  rwa [star_real_vec] at h

/-- A real positive definite matrix is symmetric. -/
theorem pd_transpose_eq {ι : Type*} [Fintype ι] {M : Matrix ι ι ℝ} (hM : M.PosDef) :
    Mᵀ = M := by
  have h := hM.isHermitian.eq
  rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at h

/-- The inverse of a real positive definite matrix is symmetric. -/
theorem pd_inv_transpose_eq {ι : Type*} [Fintype ι] [DecidableEq ι] {M : Matrix ι ι ℝ}
    (hM : M.PosDef) : (M⁻¹)ᵀ = M⁻¹ := by
  rw [Matrix.transpose_nonsing_inv, pd_transpose_eq hM]

/-- The sandwich `B M Bᵀ` of a real PSD matrix is PSD (transpose form of
`Matrix.PosSemidef.mul_mul_conjTranspose_same`). -/
theorem psd_mul_mul_transpose {ι κ : Type*} [Fintype ι] [Fintype κ] {M : Matrix ι ι ℝ}
    (hM : M.PosSemidef) (B : Matrix κ ι ℝ) : (B * M * Bᵀ).PosSemidef := by
  have h := hM.mul_mul_conjTranspose_same B
  rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at h

/-- The sandwich `Bᵀ M B` of a real PSD matrix is PSD (transpose form of
`Matrix.PosSemidef.conjTranspose_mul_mul_same`). -/
theorem psd_transpose_mul_mul {ι κ : Type*} [Fintype ι] [Fintype κ] {M : Matrix ι ι ℝ}
    (hM : M.PosSemidef) (B : Matrix ι κ ℝ) : (Bᵀ * M * B).PosSemidef := by
  have h := hM.conjTranspose_mul_mul_same B
  rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at h

/-! ## The preamble's objects, as definitions -/

variable {n m : Type*} [Fintype n] [DecidableEq n] [Fintype m] [DecidableEq m]

/-- The posterior covariance `Σ_post = (Σ_ρ⁻¹ + Aᵀ Γ⁻¹ A)⁻¹`. -/
noncomputable def Spost (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ) :
    Matrix n n ℝ := (Sig_rho⁻¹ + Aᵀ * Gam⁻¹ * A)⁻¹

/-- The posterior gain `G = Σ_post Aᵀ Γ⁻¹`. -/
noncomputable def gain (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ) :
    Matrix n m ℝ := Spost A Gam Sig_rho * Aᵀ * Gam⁻¹

/-- The regularizer's data covariance `S_ρ = A Σ_ρ Aᵀ + Γ` (the paper's `S_reg`). -/
def Sdata (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ) :
    Matrix m m ℝ := A * Sig_rho * Aᵀ + Gam

/-- The true data covariance `S_y = A Σ_⋆ Aᵀ + Γ`. -/
def Sy (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_star : Matrix n n ℝ) :
    Matrix m m ℝ := A * Sig_star * Aᵀ + Gam

/-- The middle-factor matrix `Q = S_y⁻¹ + (Γ⁻¹ − S_ρ⁻¹)` of `eq:nearnull-freeze`. -/
noncomputable def Qmat (A : Matrix m n ℝ) (Gam : Matrix m m ℝ)
    (Sig_rho Sig_star : Matrix n n ℝ) : Matrix m m ℝ :=
  (Sy A Gam Sig_star)⁻¹ + (Gam⁻¹ - (Sdata A Gam Sig_rho)⁻¹)

/-- The curated covariance `Σ_q = Σ_post + G S_y Gᵀ` — the law of total covariance taken as the
definition, so `hSq` needs no hypothesis. -/
noncomputable def Sq (A : Matrix m n ℝ) (Gam : Matrix m m ℝ)
    (Sig_rho Sig_star : Matrix n n ℝ) : Matrix n n ℝ :=
  Spost A Gam Sig_rho + gain A Gam Sig_rho * Sy A Gam Sig_star * (gain A Gam Sig_rho)ᵀ

/-! ## A1 — the definitional relations, proved -/

/-- The posterior precision `Σ_ρ⁻¹ + Aᵀ Γ⁻¹ A` is positive definite (PD plus a PSD sandwich). -/
theorem precisionPost_posDef (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    (Sig_rho⁻¹ + Aᵀ * Gam⁻¹ * A).PosDef :=
  hrho.inv.add_posSemidef (psd_transpose_mul_mul hGam.inv.posSemidef A)

/-- `Σ_post ≻ 0`: the inverse of the PD posterior precision. -/
theorem spost_posDef (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    (Spost A Gam Sig_rho).PosDef :=
  (precisionPost_posDef A Gam Sig_rho hGam hrho).inv

/-- **`hpost`, proved.**  `Σ_post⁻¹ = Σ_ρ⁻¹ + Aᵀ Γ⁻¹ A` — `⁻¹⁻¹` on the PD precision. -/
theorem spost_inv_eq (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    (Spost A Gam Sig_rho)⁻¹ = Sig_rho⁻¹ + Aᵀ * Gam⁻¹ * A := by
  simp only [Spost]
  exact Matrix.nonsing_inv_nonsing_inv _
    ((Matrix.isUnit_iff_isUnit_det _).mp (precisionPost_posDef A Gam Sig_rho hGam hrho).isUnit)

/-- **`hG`, proved.**  `Σ_post⁻¹ G = Aᵀ Γ⁻¹` — cancel `Σ_post` in `G`'s definition. -/
theorem spost_inv_gain (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    (Spost A Gam Sig_rho)⁻¹ * gain A Gam Sig_rho = Aᵀ * Gam⁻¹ := by
  have hdet : IsUnit (Spost A Gam Sig_rho).det :=
    (Matrix.isUnit_iff_isUnit_det _).mp (spost_posDef A Gam Sig_rho hGam hrho).isUnit
  simp only [gain]
  rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hdet, Matrix.one_mul]

/-- **`hGT`, proved.**  `Gᵀ Σ_post⁻¹ = Γ⁻¹ A` — `hG` transposed under symmetry of `Γ`, `Σ_post`. -/
theorem gainT_spost_inv (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    (gain A Gam Sig_rho)ᵀ * (Spost A Gam Sig_rho)⁻¹ = Gam⁻¹ * A := by
  have h := congrArg Matrix.transpose (spost_inv_gain A Gam Sig_rho hGam hrho)
  rwa [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
    Matrix.transpose_nonsing_inv, pd_transpose_eq (spost_posDef A Gam Sig_rho hGam hrho),
    Matrix.transpose_nonsing_inv, pd_transpose_eq hGam] at h

omit [DecidableEq n] [DecidableEq m] in
/-- `S_ρ ≻ 0`: PSD sandwich plus PD noise. -/
theorem sdata_posDef (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    (Sdata A Gam Sig_rho).PosDef := by
  simp only [Sdata]
  exact Matrix.PosDef.posSemidef_add (psd_mul_mul_transpose hrho.posSemidef A) hGam

omit [DecidableEq n] [DecidableEq m] in
/-- `S_y ≻ 0`: PSD sandwich plus PD noise (only `Σ_⋆ ⪰ 0` is needed of the truth). -/
theorem sy_posDef (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_star : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hstar : Sig_star.PosSemidef) :
    (Sy A Gam Sig_star).PosDef := by
  simp only [Sy]
  exact Matrix.PosDef.posSemidef_add (psd_mul_mul_transpose hstar A) hGam

/-- **The second Woodbury of the appendix.**  `Σ_post = Σ_ρ − Σ_ρ Aᵀ S_ρ⁻¹ A Σ_ρ`. -/
theorem spost_woodbury (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    Spost A Gam Sig_rho
      = Sig_rho - Sig_rho * Aᵀ * (Sdata A Gam Sig_rho)⁻¹ * A * Sig_rho := by
  have hrho_ii : (Sig_rho⁻¹)⁻¹ = Sig_rho :=
    Matrix.nonsing_inv_nonsing_inv _ ((Matrix.isUnit_iff_isUnit_det _).mp hrho.isUnit)
  have hGam_ii : (Gam⁻¹)⁻¹ = Gam :=
    Matrix.nonsing_inv_nonsing_inv _ ((Matrix.isUnit_iff_isUnit_det _).mp hGam.isUnit)
  have hsd : IsUnit ((Gam⁻¹)⁻¹ + A * (Sig_rho⁻¹)⁻¹ * Aᵀ) := by
    rw [hGam_ii, hrho_ii, add_comm Gam (A * Sig_rho * Aᵀ)]
    exact (sdata_posDef A Gam Sig_rho hGam hrho).isUnit
  have hwood := Matrix.add_mul_mul_inv_eq_sub Sig_rho⁻¹ Aᵀ Gam⁻¹ A
    hrho.inv.isUnit hGam.inv.isUnit hsd
  simp only [Spost, Sdata]
  rw [hwood, hrho_ii, hGam_ii, add_comm Gam (A * Sig_rho * Aᵀ)]

/-- The algebraic core of the appendix's "`S = D + Γ` twice" step:
`P − P S⁻¹ P = Γ − Γ S⁻¹ Γ` for `P = S − Γ` and invertible `S`. -/
private lemma sub_inv_sandwich {S Gm : Matrix m m ℝ} (h : IsUnit S.det) :
    (S - Gm) - (S - Gm) * S⁻¹ * (S - Gm) = Gm - Gm * S⁻¹ * Gm := by
  have h1 : S * S⁻¹ = 1 := Matrix.mul_nonsing_inv _ h
  have h2 : Gm * S⁻¹ * S = Gm := Matrix.nonsing_inv_mul_cancel_right _ _ h
  simp only [Matrix.sub_mul, Matrix.mul_sub, h1, h2, Matrix.one_mul]
  abel

/-- **The appendix's two-Woodbury computation, proved.**
`A Σ_post Aᵀ = Γ − Γ S_ρ⁻¹ Γ` (the paper's `D − D S_ρ⁻¹ D = Γ − Γ S_ρ⁻¹ Γ` for `D = A Σ_ρ Aᵀ`). -/
theorem a_spost_at (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    A * Spost A Gam Sig_rho * Aᵀ = Gam - Gam * (Sdata A Gam Sig_rho)⁻¹ * Gam := by
  have hsd_det : IsUnit (Sdata A Gam Sig_rho).det :=
    (Matrix.isUnit_iff_isUnit_det _).mp (sdata_posDef A Gam Sig_rho hGam hrho).isUnit
  have hP : A * Sig_rho * Aᵀ = Sdata A Gam Sig_rho - Gam := by
    simp only [Sdata]
    rw [add_sub_cancel_right]
  rw [spost_woodbury A Gam Sig_rho hGam hrho]
  have hexp : A * (Sig_rho - Sig_rho * Aᵀ * (Sdata A Gam Sig_rho)⁻¹ * A * Sig_rho) * Aᵀ
      = (A * Sig_rho * Aᵀ)
        - (A * Sig_rho * Aᵀ) * (Sdata A Gam Sig_rho)⁻¹ * (A * Sig_rho * Aᵀ) := by
    simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_assoc]
  rw [hexp, hP]
  exact sub_inv_sandwich hsd_det

/-- **The PSD identity behind `Q ≻ 0`.**  `Γ⁻¹ − S_ρ⁻¹ = (Γ⁻¹A) Σ_post (Γ⁻¹A)ᵀ` — so the paper's
"`S_ρ ⪰ Γ` makes `Γ⁻¹ − S_ρ⁻¹ ⪰ 0`" needs no Loewner inverse-antitonicity. -/
theorem gaminv_sub_sdatainv_eq (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    Gam⁻¹ - (Sdata A Gam Sig_rho)⁻¹
      = Gam⁻¹ * A * Spost A Gam Sig_rho * (Gam⁻¹ * A)ᵀ := by
  have hGam_det : IsUnit Gam.det := (Matrix.isUnit_iff_isUnit_det _).mp hGam.isUnit
  have hT : (Gam⁻¹ * A)ᵀ = Aᵀ * Gam⁻¹ := by
    rw [Matrix.transpose_mul, pd_inv_transpose_eq hGam]
  rw [hT]
  have hassoc : Gam⁻¹ * A * Spost A Gam Sig_rho * (Aᵀ * Gam⁻¹)
      = Gam⁻¹ * (A * Spost A Gam Sig_rho * Aᵀ) * Gam⁻¹ := by
    simp only [Matrix.mul_assoc]
  rw [hassoc, a_spost_at A Gam Sig_rho hGam hrho]
  have hfin : Gam⁻¹ * (Gam - Gam * (Sdata A Gam Sig_rho)⁻¹ * Gam) * Gam⁻¹
      = Gam⁻¹ - (Sdata A Gam Sig_rho)⁻¹ := by
    calc Gam⁻¹ * (Gam - Gam * (Sdata A Gam Sig_rho)⁻¹ * Gam) * Gam⁻¹
        = (Gam⁻¹ * Gam - Gam⁻¹ * (Gam * ((Sdata A Gam Sig_rho)⁻¹ * Gam))) * Gam⁻¹ := by
          rw [Matrix.mul_sub, Matrix.mul_assoc Gam _ Gam]
      _ = (1 - (Sdata A Gam Sig_rho)⁻¹ * Gam) * Gam⁻¹ := by
          rw [Matrix.nonsing_inv_mul _ hGam_det,
            Matrix.nonsing_inv_mul_cancel_left _ _ hGam_det]
      _ = Gam⁻¹ - (Sdata A Gam Sig_rho)⁻¹ * (Gam * Gam⁻¹) := by
          rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc]
      _ = Gam⁻¹ - (Sdata A Gam Sig_rho)⁻¹ := by
          rw [Matrix.mul_nonsing_inv _ hGam_det, Matrix.mul_one]
  rw [hfin]

/-- `Γ⁻¹ − S_ρ⁻¹ ⪰ 0`, from the sandwich identity (not from spectral antitonicity). -/
theorem gaminv_sub_sdatainv_psd (A : Matrix m n ℝ) (Gam : Matrix m m ℝ)
    (Sig_rho : Matrix n n ℝ) (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    (Gam⁻¹ - (Sdata A Gam Sig_rho)⁻¹).PosSemidef := by
  rw [gaminv_sub_sdatainv_eq A Gam Sig_rho hGam hrho]
  exact psd_mul_mul_transpose (spost_posDef A Gam Sig_rho hGam hrho).posSemidef (Gam⁻¹ * A)

/-- **`hGG`, proved.**  `Gᵀ Σ_post⁻¹ G = Γ⁻¹ − S_ρ⁻¹` — the appendix's chain
`Gᵀ Σ_post⁻¹ G = Γ⁻¹ A Σ_post Aᵀ Γ⁻¹ = Γ⁻¹ − S_ρ⁻¹`. -/
theorem gainT_spost_inv_gain (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) :
    (gain A Gam Sig_rho)ᵀ * (Spost A Gam Sig_rho)⁻¹ * gain A Gam Sig_rho
      = Gam⁻¹ - (Sdata A Gam Sig_rho)⁻¹ := by
  rw [gainT_spost_inv A Gam Sig_rho hGam hrho, gaminv_sub_sdatainv_eq A Gam Sig_rho hGam hrho]
  have hT : (Gam⁻¹ * A)ᵀ = Aᵀ * Gam⁻¹ := by
    rw [Matrix.transpose_mul, pd_inv_transpose_eq hGam]
  rw [hT]
  simp only [gain, Matrix.mul_assoc]

/-- `Q ≻ 0`: `S_y⁻¹ ≻ 0` plus the PSD difference `Γ⁻¹ − S_ρ⁻¹`. -/
theorem q_posDef (A : Matrix m n ℝ) (Gam : Matrix m m ℝ) (Sig_rho Sig_star : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) (hstar : Sig_star.PosSemidef) :
    (Qmat A Gam Sig_rho Sig_star).PosDef := by
  simp only [Qmat]
  exact (sy_posDef A Gam Sig_star hGam hstar).inv.add_posSemidef
    (gaminv_sub_sdatainv_psd A Gam Sig_rho hGam hrho)

/-! ## A1 capstone — `NearNullComposed.precision_bound` with every definitional hypothesis closed -/

/-- **`p:nearnull` (i), definitional.**  Hypotheses: `Γ ≻ 0`, `Σ_ρ ≻ 0`, `Σ_⋆ ⪰ 0`, and the
variational bound `c` on the middle factor `Γ⁻¹ − Γ⁻¹ Q⁻¹ Γ⁻¹` (discharged from the spectral band
by `middle_factor_form_le` below, giving `precision_bound_spectral`).  Every definitional relation
and invertibility side condition of `NearNullComposed.precision_bound` is proved, not assumed. -/
theorem precision_bound_definitional (A : Matrix m n ℝ) (Gam : Matrix m m ℝ)
    (Sig_rho Sig_star : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) (hstar : Sig_star.PosSemidef) {c : ℝ}
    (hB : ∀ w : m → ℝ,
      |w ⬝ᵥ ((Gam⁻¹ - Gam⁻¹ * (Qmat A Gam Sig_rho Sig_star)⁻¹ * Gam⁻¹) *ᵥ w)|
        ≤ c * (w ⬝ᵥ w))
    (v : n → ℝ) :
    |v ⬝ᵥ (((Sq A Gam Sig_rho Sig_star)⁻¹ - Sig_rho⁻¹) *ᵥ v)|
      ≤ c * ((A *ᵥ v) ⬝ᵥ (A *ᵥ v)) := by
  refine NearNullComposed.precision_bound A (Spost A Gam Sig_rho) Sig_rho
    (Sq A Gam Sig_rho Sig_star) (gain A Gam Sig_rho) Gam (Sy A Gam Sig_star)
    (Sdata A Gam Sig_rho) (Qmat A Gam Sig_rho Sig_star)
    rfl
    (spost_inv_gain A Gam Sig_rho hGam hrho)
    (gainT_spost_inv A Gam Sig_rho hGam hrho)
    (gainT_spost_inv_gain A Gam Sig_rho hGam hrho)
    (spost_inv_eq A Gam Sig_rho hGam hrho)
    rfl
    (spost_posDef A Gam Sig_rho hGam hrho).isUnit
    (sy_posDef A Gam Sig_star hGam hstar).isUnit
    ?_ hB v
  rw [gainT_spost_inv_gain A Gam Sig_rho hGam hrho]
  exact (q_posDef A Gam Sig_rho Sig_star hGam hrho hstar).isUnit

/-- **The exact freeze at the kernel, definitional.**  On `A v = 0` the one step moves the prior's
precision along `v` by exactly `0` — no hypothesis beyond `Γ ≻ 0`, `Σ_ρ ≻ 0`, `Σ_⋆ ⪰ 0`. -/
theorem precision_zero_of_kernel_definitional (A : Matrix m n ℝ) (Gam : Matrix m m ℝ)
    (Sig_rho Sig_star : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) (hstar : Sig_star.PosSemidef)
    {v : n → ℝ} (hv : A *ᵥ v = 0) :
    v ⬝ᵥ (((Sq A Gam Sig_rho Sig_star)⁻¹ - Sig_rho⁻¹) *ᵥ v) = 0 := by
  refine NearNullComposed.precision_zero_of_kernel A (Spost A Gam Sig_rho) Sig_rho
    (Sq A Gam Sig_rho Sig_star) (gain A Gam Sig_rho) Gam (Sy A Gam Sig_star)
    (Sdata A Gam Sig_rho) (Qmat A Gam Sig_rho Sig_star)
    rfl
    (spost_inv_gain A Gam Sig_rho hGam hrho)
    (gainT_spost_inv A Gam Sig_rho hGam hrho)
    (gainT_spost_inv_gain A Gam Sig_rho hGam hrho)
    (spost_inv_eq A Gam Sig_rho hGam hrho)
    rfl
    (spost_posDef A Gam Sig_rho hGam hrho).isUnit
    (sy_posDef A Gam Sig_star hGam hstar).isUnit
    ?_ hv
  rw [gainT_spost_inv_gain A Gam Sig_rho hGam hrho]
  exact (q_posDef A Gam Sig_rho Sig_star hGam hrho hstar).isUnit

/-! ## A2 — the variational bounds from the spectral band

The quadratic-form Cauchy–Schwarz (`discrim_le_zero`, no spectral theorem), the inverse band
`0 ≤ w·(Γ⁻¹w) ≤ γ⁻¹(w·w)`, the variational inverse-antitonicity `X ⪯ Y → Y⁻¹ ⪯ X⁻¹`, and the
middle-factor bound `Γ⁻¹ Q⁻¹ Γ⁻¹ ⪯ (s/γ²) I`. -/

omit [DecidableEq n] [DecidableEq m] in
/-- **Quadratic-form Cauchy–Schwarz.**  For symmetric `X` with nonnegative form,
`⟪u, Xv⟫² ≤ ⟪u, Xu⟫ ⟪v, Xv⟫` — by `discrim_le_zero` on `t ↦ ⟪u + tv, X(u + tv)⟫ ≥ 0`. -/
theorem form_cauchy_schwarz {X : Matrix m m ℝ} (hsym : Xᵀ = X)
    (h0 : ∀ w : m → ℝ, 0 ≤ w ⬝ᵥ (X *ᵥ w)) (u v : m → ℝ) :
    (u ⬝ᵥ (X *ᵥ v)) ^ 2 ≤ (u ⬝ᵥ (X *ᵥ u)) * (v ⬝ᵥ (X *ᵥ v)) := by
  have hvm : v ᵥ* X = X *ᵥ v := by
    conv_lhs => rw [← hsym]
    rw [Matrix.vecMul_transpose]
  have hswap : v ⬝ᵥ (X *ᵥ u) = u ⬝ᵥ (X *ᵥ v) := by
    rw [dotProduct_mulVec, hvm, dotProduct_comm]
  have hexp : ∀ t : ℝ, (u + t • v) ⬝ᵥ (X *ᵥ (u + t • v))
      = (v ⬝ᵥ (X *ᵥ v)) * (t * t) + 2 * (u ⬝ᵥ (X *ᵥ v)) * t + u ⬝ᵥ (X *ᵥ u) := by
    intro t
    simp only [Matrix.mulVec_add, Matrix.mulVec_smul, dotProduct_add, add_dotProduct,
      dotProduct_smul, smul_dotProduct, smul_eq_mul, hswap]
    ring
  have hquad : ∀ t : ℝ,
      0 ≤ (v ⬝ᵥ (X *ᵥ v)) * (t * t) + 2 * (u ⬝ᵥ (X *ᵥ v)) * t + u ⬝ᵥ (X *ᵥ u) := by
    intro t
    rw [← hexp t]
    exact h0 _
  have hd := discrim_le_zero hquad
  rw [discrim] at hd
  nlinarith [hd]

omit [DecidableEq n] in
/-- Plain Cauchy–Schwarz for the dot product, as the `X = 1` case. -/
theorem dot_cauchy_schwarz (u v : m → ℝ) : (u ⬝ᵥ v) ^ 2 ≤ (u ⬝ᵥ u) * (v ⬝ᵥ v) := by
  have h := form_cauchy_schwarz (X := (1 : Matrix m m ℝ)) Matrix.transpose_one
    (fun w => by rw [Matrix.one_mulVec]; exact NearNullComposed.dotProduct_self_nonneg w) u v
  simpa [Matrix.one_mulVec] using h

omit [DecidableEq n] in
/-- `0 ≤ w·(M⁻¹w)` for `M ≻ 0`. -/
theorem inv_form_nonneg {M : Matrix m m ℝ} (hM : M.PosDef) (w : m → ℝ) :
    0 ≤ w ⬝ᵥ (M⁻¹ *ᵥ w) :=
  psd_form_nonneg hM.inv.posSemidef w

omit [DecidableEq n] in
/-- **The inverse band.**  From `γ (w·w) ≤ w·(Γw)` with `γ > 0`:
`w·(Γ⁻¹w) ≤ γ⁻¹ (w·w)` — inverse antitonicity at the identity, by Cauchy–Schwarz. -/
theorem inv_form_le {Gm : Matrix m m ℝ} (hGm : Gm.PosDef) {gam : ℝ} (hgam : 0 < gam)
    (hlb : ∀ w : m → ℝ, gam * (w ⬝ᵥ w) ≤ w ⬝ᵥ (Gm *ᵥ w)) (w : m → ℝ) :
    w ⬝ᵥ (Gm⁻¹ *ᵥ w) ≤ gam⁻¹ * (w ⬝ᵥ w) := by
  have hdet : IsUnit Gm.det := (Matrix.isUnit_iff_isUnit_det _).mp hGm.isUnit
  set u := Gm⁻¹ *ᵥ w with hu
  have hGu : Gm *ᵥ u = w := by
    rw [hu, Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hdet, Matrix.one_mulVec]
  have ht0 : 0 ≤ w ⬝ᵥ u := by
    have h := inv_form_nonneg hGm w
    rwa [← hu] at h
  have hband : gam * (u ⬝ᵥ u) ≤ w ⬝ᵥ u := by
    have h := hlb u
    rwa [hGu, dotProduct_comm u w] at h
  have hCS : (w ⬝ᵥ u) ^ 2 ≤ (w ⬝ᵥ w) * (u ⬝ᵥ u) := dot_cauchy_schwarz w u
  have hww0 : 0 ≤ w ⬝ᵥ w := NearNullComposed.dotProduct_self_nonneg w
  have huu0 : 0 ≤ u ⬝ᵥ u := NearNullComposed.dotProduct_self_nonneg u
  have hkey : gam * (w ⬝ᵥ u) ≤ w ⬝ᵥ w := by
    rcases ht0.lt_or_eq with hpos | hzero
    · nlinarith [hCS, hband, hpos, hww0, huu0,
        mul_le_mul_of_nonneg_left hCS hgam.le, mul_le_mul_of_nonneg_right hband hww0]
    · rw [← hzero, mul_zero]
      exact hww0
  calc w ⬝ᵥ u = gam⁻¹ * (gam * (w ⬝ᵥ u)) := (inv_mul_cancel_left₀ (ne_of_gt hgam) _).symm
    _ ≤ gam⁻¹ * (w ⬝ᵥ w) := mul_le_mul_of_nonneg_left hkey (inv_pos.mpr hgam).le

omit [DecidableEq n] in
/-- The squared-norm form of the inverse band: `‖Γ⁻¹w‖² ≤ γ⁻¹ (γ⁻¹ ‖w‖²)`. -/
theorem inv_mulVec_sq_le {Gm : Matrix m m ℝ} (hGm : Gm.PosDef) {gam : ℝ} (hgam : 0 < gam)
    (hlb : ∀ w : m → ℝ, gam * (w ⬝ᵥ w) ≤ w ⬝ᵥ (Gm *ᵥ w)) (w : m → ℝ) :
    (Gm⁻¹ *ᵥ w) ⬝ᵥ (Gm⁻¹ *ᵥ w) ≤ gam⁻¹ * (gam⁻¹ * (w ⬝ᵥ w)) := by
  have hdet : IsUnit Gm.det := (Matrix.isUnit_iff_isUnit_det _).mp hGm.isUnit
  set z := Gm⁻¹ *ᵥ w with hz
  have hGz : Gm *ᵥ z = w := by
    rw [hz, Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hdet, Matrix.one_mulVec]
  have h1 : gam * (z ⬝ᵥ z) ≤ w ⬝ᵥ z := by
    have h := hlb z
    rwa [hGz, dotProduct_comm z w] at h
  have h2 : w ⬝ᵥ z ≤ gam⁻¹ * (w ⬝ᵥ w) := by
    have h := inv_form_le hGm hgam hlb w
    rwa [← hz] at h
  have h3 : gam * (z ⬝ᵥ z) ≤ gam⁻¹ * (w ⬝ᵥ w) := le_trans h1 h2
  calc z ⬝ᵥ z = gam⁻¹ * (gam * (z ⬝ᵥ z)) := (inv_mul_cancel_left₀ (ne_of_gt hgam) _).symm
    _ ≤ gam⁻¹ * (gam⁻¹ * (w ⬝ᵥ w)) := mul_le_mul_of_nonneg_left h3 (inv_pos.mpr hgam).le

omit [DecidableEq n] in
/-- **Variational inverse-antitonicity.**  `X ⪯ Y` (both `≻ 0`) gives `Y⁻¹ ⪯ X⁻¹`, form by form —
the quadratic-form Cauchy–Schwarz route, no Loewner-order API. -/
theorem inv_antitone_form {X Y : Matrix m m ℝ} (hX : X.PosDef) (hY : Y.PosDef)
    (hle : ∀ w : m → ℝ, w ⬝ᵥ (X *ᵥ w) ≤ w ⬝ᵥ (Y *ᵥ w)) (w : m → ℝ) :
    w ⬝ᵥ (Y⁻¹ *ᵥ w) ≤ w ⬝ᵥ (X⁻¹ *ᵥ w) := by
  have hXdet : IsUnit X.det := (Matrix.isUnit_iff_isUnit_det _).mp hX.isUnit
  have hYdet : IsUnit Y.det := (Matrix.isUnit_iff_isUnit_det _).mp hY.isUnit
  set u := Y⁻¹ *ᵥ w with hu
  set z := X⁻¹ *ᵥ w with hz
  have hYu : Y *ᵥ u = w := by
    rw [hu, Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hYdet, Matrix.one_mulVec]
  have hXz : X *ᵥ z = w := by
    rw [hz, Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hXdet, Matrix.one_mulVec]
  have ht0 : 0 ≤ w ⬝ᵥ u := by
    have h := inv_form_nonneg hY w
    rwa [← hu] at h
  have hs0 : 0 ≤ w ⬝ᵥ z := by
    have h := inv_form_nonneg hX w
    rwa [← hz] at h
  have hCS : (u ⬝ᵥ (X *ᵥ z)) ^ 2 ≤ (u ⬝ᵥ (X *ᵥ u)) * (z ⬝ᵥ (X *ᵥ z)) :=
    form_cauchy_schwarz (pd_transpose_eq hX) (fun w' => psd_form_nonneg hX.posSemidef w') u z
  rw [hXz, dotProduct_comm u w, dotProduct_comm z w] at hCS
  have hXu : u ⬝ᵥ (X *ᵥ u) ≤ w ⬝ᵥ u := by
    have h := hle u
    rwa [hYu, dotProduct_comm u w] at h
  have hXu0 : 0 ≤ u ⬝ᵥ (X *ᵥ u) := psd_form_nonneg hX.posSemidef u
  rcases ht0.lt_or_eq with hpos | hzero
  · nlinarith [hCS, hXu, hs0, hpos, hXu0, mul_le_mul_of_nonneg_right hXu hs0]
  · rw [← hzero]
    exact hs0

/-- **`Q ⪰ S_y⁻¹`**, form by form: `Q = S_y⁻¹ + (Γ⁻¹ − S_ρ⁻¹)` and the difference is PSD. -/
theorem q_form_ge_sy_inv (A : Matrix m n ℝ) (Gam : Matrix m m ℝ)
    (Sig_rho Sig_star : Matrix n n ℝ) (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef)
    (w : m → ℝ) :
    w ⬝ᵥ ((Sy A Gam Sig_star)⁻¹ *ᵥ w) ≤ w ⬝ᵥ (Qmat A Gam Sig_rho Sig_star *ᵥ w) := by
  simp only [Qmat]
  rw [Matrix.add_mulVec, dotProduct_add]
  have h := psd_form_nonneg (gaminv_sub_sdatainv_psd A Gam Sig_rho hGam hrho) w
  linarith

/-- **The middle-factor bound from the band.**  With `λ_min(Γ) ≥ γ > 0` and `λ_max(S_y) ≤ s`
(variationally), the middle factor of `eq:nearnull-freeze` satisfies
`|w·((Γ⁻¹ − Γ⁻¹Q⁻¹Γ⁻¹)w)| ≤ max(γ⁻¹, s/γ²) (w·w)` — the paper's
`c ≤ λ_min(Γ)⁻¹ max{1, λ_max(S_y)/λ_min(Γ)}` via `Q ⪰ S_y⁻¹`, `Q⁻¹ ⪯ S_y`. -/
theorem middle_factor_form_le (A : Matrix m n ℝ) (Gam : Matrix m m ℝ)
    (Sig_rho Sig_star : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) (hstar : Sig_star.PosSemidef)
    {gam s : ℝ} (hgam : 0 < gam)
    (hGam_lb : ∀ w : m → ℝ, gam * (w ⬝ᵥ w) ≤ w ⬝ᵥ (Gam *ᵥ w))
    (hSy_ub : ∀ w : m → ℝ, w ⬝ᵥ ((Sy A Gam Sig_star) *ᵥ w) ≤ s * (w ⬝ᵥ w))
    (w : m → ℝ) :
    |w ⬝ᵥ ((Gam⁻¹ - Gam⁻¹ * (Qmat A Gam Sig_rho Sig_star)⁻¹ * Gam⁻¹) *ᵥ w)|
      ≤ max gam⁻¹ (s / gam ^ 2) * (w ⬝ᵥ w) := by
  have hQpd := q_posDef A Gam Sig_rho Sig_star hGam hrho hstar
  have hSypd := sy_posDef A Gam Sig_star hGam hstar
  have hSydet : IsUnit (Sy A Gam Sig_star).det :=
    (Matrix.isUnit_iff_isUnit_det _).mp hSypd.isUnit
  have hne : gam ≠ 0 := ne_of_gt hgam
  -- the factored form of the `Y`-quadratic: `w·(Γ⁻¹Q⁻¹Γ⁻¹ w) = (Γ⁻¹w)·(Q⁻¹ (Γ⁻¹w))`
  have hfact : ∀ w' : m → ℝ,
      w' ⬝ᵥ ((Gam⁻¹ * (Qmat A Gam Sig_rho Sig_star)⁻¹ * Gam⁻¹) *ᵥ w')
        = (Gam⁻¹ *ᵥ w') ⬝ᵥ ((Qmat A Gam Sig_rho Sig_star)⁻¹ *ᵥ (Gam⁻¹ *ᵥ w')) := by
    intro w'
    have h := NearNullComposed.quadratic_form_factored Gam⁻¹
      (Qmat A Gam Sig_rho Sig_star)⁻¹ w'
    rwa [pd_inv_transpose_eq hGam] at h
  have hY0 : ∀ w' : m → ℝ,
      0 ≤ w' ⬝ᵥ ((Gam⁻¹ * (Qmat A Gam Sig_rho Sig_star)⁻¹ * Gam⁻¹) *ᵥ w') := by
    intro w'
    rw [hfact w']
    exact psd_form_nonneg hQpd.inv.posSemidef _
  have hYb : ∀ w' : m → ℝ,
      w' ⬝ᵥ ((Gam⁻¹ * (Qmat A Gam Sig_rho Sig_star)⁻¹ * Gam⁻¹) *ᵥ w')
        ≤ (s / gam ^ 2) * (w' ⬝ᵥ w') := by
    intro w'
    rw [hfact w']
    rcases (NearNullComposed.dotProduct_self_nonneg w').lt_or_eq with hwpos | hwzero
    · -- `w' ≠ 0`, so `S_y ≻ 0` forces `s > 0` and the chain closes
      have hw'ne : w' ≠ 0 := by
        intro h0
        rw [h0] at hwpos
        simp at hwpos
      have hSyform : 0 < w' ⬝ᵥ ((Sy A Gam Sig_star) *ᵥ w') := by
        have h := hSypd.dotProduct_mulVec_pos hw'ne
        rwa [star_real_vec] at h
      have hs : 0 < s := by nlinarith [hSy_ub w', hSyform, hwpos]
      set z := Gam⁻¹ *ᵥ w' with hzdef
      have hanti := inv_antitone_form hSypd.inv hQpd
        (fun w'' => q_form_ge_sy_inv A Gam Sig_rho Sig_star hGam hrho w'') z
      rw [Matrix.nonsing_inv_nonsing_inv _ hSydet] at hanti
      have hSyz := hSy_ub z
      have hzz : z ⬝ᵥ z ≤ gam⁻¹ * (gam⁻¹ * (w' ⬝ᵥ w')) := by
        have h := inv_mulVec_sq_le hGam hgam hGam_lb w'
        rwa [← hzdef] at h
      have hchain : z ⬝ᵥ ((Qmat A Gam Sig_rho Sig_star)⁻¹ *ᵥ z) ≤ s * (z ⬝ᵥ z) :=
        le_trans hanti hSyz
      have hfin : s * (z ⬝ᵥ z) ≤ s / gam ^ 2 * (w' ⬝ᵥ w') := by
        have h := mul_le_mul_of_nonneg_left hzz hs.le
        have heq : s * (gam⁻¹ * (gam⁻¹ * (w' ⬝ᵥ w'))) = s / gam ^ 2 * (w' ⬝ᵥ w') := by
          field_simp
        rwa [heq] at h
      linarith
    · -- `w' = 0`: both sides vanish
      have hw0 : w' = 0 := dotProduct_self_eq_zero.mp hwzero.symm
      subst hw0
      simp
  exact NearNullComposed.psd_sub_form_le
    (fun w' => inv_form_nonneg hGam w')
    (fun w' => inv_form_le hGam hgam hGam_lb w')
    hY0 hYb w

/-! ## A3 — the capstone: `p:nearnull` (i) from `Γ ≻ 0`, `Σ_ρ ≻ 0`, `Σ_⋆ ⪰ 0` and the band -/

/-- **`p:nearnull` (i), hypothesis-free up to the spectral band.**  For `Γ ≻ 0`, `Σ_ρ ≻ 0`,
`Σ_⋆ ⪰ 0`, and the band `γ ≤ λ_min(Γ)`, `λ_max(S_y) ≤ s` (stated variationally), the one step
moves the prior's precision along any `v` by at most `max(γ⁻¹, s/γ²) ‖A v‖²`:

    |⟪v, ((Σ_post + G S_y Gᵀ)⁻¹ − Σ_ρ⁻¹) v⟫| ≤ max(γ⁻¹, s/γ²) ⟪A v, A v⟫,

with `Σ_post`, `G`, `S_y` *defined* from `(A, Γ, Σ_ρ, Σ_⋆)` — nothing definitional assumed. -/
theorem precision_bound_spectral (A : Matrix m n ℝ) (Gam : Matrix m m ℝ)
    (Sig_rho Sig_star : Matrix n n ℝ)
    (hGam : Gam.PosDef) (hrho : Sig_rho.PosDef) (hstar : Sig_star.PosSemidef)
    {gam s : ℝ} (hgam : 0 < gam)
    (hGam_lb : ∀ w : m → ℝ, gam * (w ⬝ᵥ w) ≤ w ⬝ᵥ (Gam *ᵥ w))
    (hSy_ub : ∀ w : m → ℝ, w ⬝ᵥ ((Sy A Gam Sig_star) *ᵥ w) ≤ s * (w ⬝ᵥ w))
    (v : n → ℝ) :
    |v ⬝ᵥ (((Sq A Gam Sig_rho Sig_star)⁻¹ - Sig_rho⁻¹) *ᵥ v)|
      ≤ max gam⁻¹ (s / gam ^ 2) * ((A *ᵥ v) ⬝ᵥ (A *ᵥ v)) :=
  precision_bound_definitional A Gam Sig_rho Sig_star hGam hrho hstar
    (middle_factor_form_le A Gam Sig_rho Sig_star hGam hrho hstar hgam hGam_lb hSy_ub) v

end NearNullDefinitional

-- #print axioms audit (p:nearnull (i), definitional)
#print axioms NearNullDefinitional.psd_form_nonneg
#print axioms NearNullDefinitional.pd_transpose_eq
#print axioms NearNullDefinitional.pd_inv_transpose_eq
#print axioms NearNullDefinitional.psd_mul_mul_transpose
#print axioms NearNullDefinitional.psd_transpose_mul_mul
#print axioms NearNullDefinitional.precisionPost_posDef
#print axioms NearNullDefinitional.spost_posDef
#print axioms NearNullDefinitional.spost_inv_eq
#print axioms NearNullDefinitional.spost_inv_gain
#print axioms NearNullDefinitional.gainT_spost_inv
#print axioms NearNullDefinitional.sdata_posDef
#print axioms NearNullDefinitional.sy_posDef
#print axioms NearNullDefinitional.spost_woodbury
#print axioms NearNullDefinitional.a_spost_at
#print axioms NearNullDefinitional.gaminv_sub_sdatainv_eq
#print axioms NearNullDefinitional.gaminv_sub_sdatainv_psd
#print axioms NearNullDefinitional.gainT_spost_inv_gain
#print axioms NearNullDefinitional.q_posDef
#print axioms NearNullDefinitional.precision_bound_definitional
#print axioms NearNullDefinitional.precision_zero_of_kernel_definitional
#print axioms NearNullDefinitional.form_cauchy_schwarz
#print axioms NearNullDefinitional.dot_cauchy_schwarz
#print axioms NearNullDefinitional.inv_form_nonneg
#print axioms NearNullDefinitional.inv_form_le
#print axioms NearNullDefinitional.inv_mulVec_sq_le
#print axioms NearNullDefinitional.inv_antitone_form
#print axioms NearNullDefinitional.q_form_ge_sy_inv
#print axioms NearNullDefinitional.middle_factor_form_le
#print axioms NearNullDefinitional.precision_bound_spectral
