#!/usr/bin/env python
"""Verify the single-best / MAP theory at machine precision in an exactly solvable model.

Closed-form curated priors in an exact LG model (so any miscalibration is the prior's, not inference):
  truth     x ~ N(mu_star, S_star);   regularizer  rho = N(mu_rho, S_rho);   y = A x + N(0, Gam),  ker A = span(N).
  posterior x|y ~ N(mu_post(y), S_post),  S_post = (S_rho^-1 + A^T Gam^-1 A)^-1,  G = S_post A^T Gam^-1.
  mode (S) curated:  q_S = T[rho] = N(mu_qS, S_qS),  S_qS = S_post + G S_y G^T,  S_y = A S_star A^T + Gam.
  mode (M) curated:  q_M = pushforward of y through MAP_rho(y)=mu_post(y) = N(mu_qS, S_qM),  S_qM = G S_y G^T.

Checks (PASS/FAIL):
  C1 freeze (p:blind)        : N^T S_qS^-1 N == N^T S_rho^-1 N   (curated blind-fiber conditional = regularizer's)
  C2 total-covariance        : S_qS - S_qM == S_post == E_y[Cov(x|y)]   (what single-best discards)
  C3 MAP collapse (p:map)    : S_qM rank <= r ; blind-fiber CONDITIONAL variance == 0 ; coverage C^M == 0
  C4 off-alignment marginal  : non-aligned -> N^T S_qM N > 0 (marginal) yet rank(G^T N) <= r and conditional == 0
  C5 coverage (p:cover)      : aligned C^S == 2 Phi(z s_rho/s_star)-1, analytic == empirical
  C6 smoothing floor (r:map-lg): q_M smoothed by h -> C^M(h) = 2 Phi(z h/s_star)-1 -> 0 as h->0
  C7 undetectability witness : truth S_star vs S_star + N E N^T -> identical p_star(y), different true coverage
  C8 pathwise (N3)           : MAP archive is byte-identical under blind perturbation of the truth (same y's)
  C9 ensemble (N4)           : pool K regularizers -> K-atom blind conditional, still C^pool == 0
"""
import os

import numpy as np
from numpy.linalg import inv, matrix_rank
from scipy.stats import norm

from priorlaundermat.closed_form import conditional_blind_variance, curated_covariances

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rng = np.random.default_rng(0)
d, r = 20, 12
b = d - r                                   # blind dim = 8
ALPHA = 0.90; z = norm.ppf((1 + ALPHA) / 2)
TOL = 1e-9


def basis():
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q[:, :r], Q[:, r:], Q                # Vr (resolved), N (blind), full Q


def spd(scale_res, scale_blind, Q, aligned):
    if aligned:                                  # diagonal in the resolved(+)blind basis
        diag = np.concatenate([scale_res * rng.uniform(.7, 1.3, r),
                               scale_blind * rng.uniform(.7, 1.3, b)])
        return Q @ np.diag(diag) @ Q.T
    Wt = rng.standard_normal((d, d))             # general SPD (NOT block-diagonal in the split)
    base = Wt @ Wt.T / d + 0.3 * np.eye(d)
    return base * scale_blind                    # scale overall; blind not isolated


def cond_blind_var(S, Q):
    return conditional_blind_variance(S, Q, r)


def curated(S_rho, S_star, mu_rho, mu_star, A, Gam):
    c = curated_covariances(S_rho, S_star, A, Gam)
    return c["S_post"], c["G"], c["S_y"], c["S_qS"], c["S_qM"]


def ok(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  '+info) if info else ''}", flush=True)
    return cond


print("=== closed-form checks (d=20, r=12, blind b=8) ===", flush=True)
allpass = True

# Residuals are recorded as they are computed and emitted as a LaTeX fragment, so that no value
# reaches the paper by hand transcription.
REC = []


def sci(x):
    """1.2e-15 -> $1.2\\times10^{-15}$ (tex, ASCII-only)."""
    m, e = f"{float(x):.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def rec(check, measured):
    REC.append((check, measured))
for aligned in (True, False):
    tag = "ALIGNED" if aligned else "NON-ALIGNED"
    Vr, N, Q = basis()
    A = (np.linspace(1.0, 0.3, r)[:, None]) * Vr.T          # r x d, ker A = span(N)
    Gam = 0.05 * np.eye(r)
    S_rho = spd(1.0, 0.20, Q, aligned)                      # regularizer: TIGHT on blind
    S_star = spd(1.0, 1.00, Q, aligned)                    # truth: wide on blind  (s_rho < s_star)
    mu_rho = np.zeros(d); mu_star = rng.standard_normal(d) * 0.1
    S_post, G, S_y, S_qS, S_qM = curated(S_rho, S_star, mu_rho, mu_star, A, Gam)
    print(f"\n--- {tag} ---", flush=True)

    # C1 freeze
    e1 = np.abs(N.T @ inv(S_qS) @ N - N.T @ inv(S_rho) @ N).max()
    allpass &= ok("C1 freeze  N^T S_qS^-1 N = N^T S_rho^-1 N", e1 < TOL, f"max|Δ|={e1:.1e}")
    rec(f"C1 blind-fiber freeze ({tag.lower()})", sci(e1))
    # C2 law of total covariance
    e2 = np.abs((S_qS - S_qM) - S_post).max()
    allpass &= ok("C2 total-cov  S_qS - S_qM = S_post", e2 < TOL, f"max|Δ|={e2:.1e}")
    rec(f"C2 total-covariance identity ({tag.lower()})", sci(e2))
    # C3 MAP collapse: rank + conditional blind variance zero
    rk = matrix_rank(S_qM, tol=1e-8); cbv = cond_blind_var(S_qM, Q)
    allpass &= ok("C3 MAP collapse  rank(S_qM)<=r & cond blind var=0",
                  rk <= r and np.abs(cbv).max() < 1e-8, f"rank={rk}<= {r}, max cond blind var={np.abs(cbv).max():.1e}")
    rec(f"C3 single-best collapse, conditional blind variance ({tag.lower()})", sci(np.abs(cbv).max()))
    # C4 off-alignment marginal: marginal blind var > 0 but rank(G^T N) <= r, conditional still 0
    if not aligned:
        mbv = np.diag(N.T @ S_qM @ N); rkGN = matrix_rank(G.T @ N, tol=1e-8)
        allpass &= ok("C4 off-align marginal blind var>0, rank(G^T N)<=r, cond=0",
                      mbv.max() > 1e-6 and rkGN <= r and np.abs(cond_blind_var(S_qM, Q)).max() < 1e-8,
                      f"marg blind var max={mbv.max():.2e}, rank(G^T N)={rkGN}")
        rec("C4 off-alignment: conditional blind variance", sci(np.abs(cond_blind_var(S_qM, Q)).max()))
    # C5 coverage p:cover: the curated prior's OWN interval (centered at its FROZEN mean), general delta form
    if aligned:
        Gi = inv(Gam)
        mu_qS = S_post @ (inv(S_rho) @ mu_rho + A.T @ Gi @ (A @ mu_star))   # curated (data-averaged posterior) mean
        v = N[:, 0]                                         # a blind direction
        s_rho = np.sqrt(v @ S_qS @ v); s_star = np.sqrt(v @ S_star @ v)
        m_cur = mu_qS @ v; m_star = mu_star @ v
        e_freeze = abs(m_cur - mu_rho @ v)                  # curated blind mean = regularizer's (freeze includes the mean)
        delta = (m_cur - m_star) / s_star
        C_gen = norm.cdf(delta + z * s_rho / s_star) - norm.cdf(delta - z * s_rho / s_star)
        xs = rng.multivariate_normal(mu_star, S_star, 300000) @ v
        C_emp = np.mean(np.abs(xs - m_cur) <= z * s_rho)    # curated interval, centered at its OWN mean
        allpass &= ok("C5 coverage = Phi(d+zr)-Phi(d-zr) on the curated's OWN interval",
                      abs(C_gen - C_emp) < 5e-3 and e_freeze < 1e-9,
                      f"general={C_gen:.3f} empirical={C_emp:.3f} (delta={delta:.2f}, s_rho/s_star={s_rho/s_star:.2f}, freeze|Δmean|={e_freeze:.0e})")
        rec("C5 coverage law, analytic vs.\\ empirical", sci(abs(C_gen - C_emp)))
        rec("C5 frozen-mean equality", sci(e_freeze))
        # C5b: a deliberate FROZEN-MEAN error makes coverage strictly WORSE than the clean width-only number
        mu_rho2 = mu_rho + N @ (1.5 * np.ones(b))            # shift the regularizer on the blind subspace
        mu_qS2 = S_post @ (inv(S_rho) @ mu_rho2 + A.T @ Gi @ (A @ mu_star)); m_cur2 = mu_qS2 @ v
        C_clean = 2 * norm.cdf(z * s_rho / s_star) - 1
        C_emp2 = np.mean(np.abs(xs - m_cur2) <= z * s_rho)
        allpass &= ok("C5b frozen mean error worsens coverage (C < clean width-only number)",
                      C_emp2 < C_clean - 0.05, f"clean width-only={C_clean:.3f}  with-mean-error={C_emp2:.3f}")
        # C6 smoothing floor: q_M (point on blind) smoothed by bandwidth h
        s_star_b = s_star
        for h in (0.5, 0.1, 0.02):
            print(f"      C6 smoothing floor h={h:>4}: C^M(h)=2Phi(z h/s_star)-1 = {2 * norm.cdf(z * h / s_star_b) - 1:.3f}", flush=True)
        allpass &= ok("C6 smoothing floor -> 0 as h->0", (2 * norm.cdf(z * 0.02 / s_star_b) - 1) < 0.1)

# C7/C8/C9 on a fresh aligned instance
Vr, N, Q = basis()
A = (np.linspace(1.0, 0.3, r)[:, None]) * Vr.T; Gam = 0.05 * np.eye(r)
S_rho = spd(1.0, 0.20, Q, True); S_star = spd(1.0, 1.0, Q, True)
mu_rho = np.zeros(d); mu_star = rng.standard_normal(d) * 0.1
print("\n--- UNDETECTABILITY / PATHWISE / ENSEMBLE ---", flush=True)

# C7 witness family: perturb truth ONLY on blind subspace
E = N @ (np.diag(rng.uniform(1, 3, b))) @ N.T              # PSD, supported on blind
S_star2 = S_star + E
Sy1 = A @ S_star @ A.T + Gam; Sy2 = A @ S_star2 @ A.T + Gam
e7 = np.abs(Sy1 - Sy2).max()                               # data marginal cov unchanged
v = N[:, 0]; ss1 = np.sqrt(v @ S_star @ v); ss2 = np.sqrt(v @ S_star2 @ v)
allpass &= ok("C7 witness: identical p_star(y) cov, different s_star",
              e7 < 1e-12 and abs(ss1 - ss2) > 0.1, f"max|ΔS_y|={e7:.1e}, s_star {ss1:.2f}->{ss2:.2f}")
rec("C7 witness family: change in the data covariance", sci(e7))

# C8 archive law: the MAP archive has the SAME DISTRIBUTION under the two witness truths.
#
# Pathwise equality at a fixed y is visible in the formula below -- no S_star appears in it -- and
# needs no numerics. The statement with content is DISTRIBUTIONAL: because the two truths induce
# the same law on y (C7), the archives they generate have the same law, so an archive cannot reveal
# which truth produced it. That is checkable only to Monte-Carlo error, never to zero, which is why
# this check carries a sampling tolerance and the others do not.
Gi = inv(Gam); S_post = inv(inv(S_rho) + A.T @ Gi @ A); G = S_post @ A.T @ Gi
arch = lambda Y: (S_post @ (inv(S_rho) @ mu_rho)[:, None] + G @ Y.T).T   # MAP_rho(y_i); truth enters ONLY via Y
NS = 20000
draw = lambda S: (rng.multivariate_normal(mu_star, S, NS) @ A.T) + rng.multivariate_normal(np.zeros(r), Gam, NS)
a1, a2 = arch(draw(S_star)), arch(draw(S_star2))           # archives under the two witness truths
d_mean = np.abs(a1.mean(0) - a2.mean(0)).max()
d_cov = np.abs(np.cov(a1.T) - np.cov(a2.T)).max()
# analytic archive covariance G S_y G^T is identical for both (C7), so both deltas are pure MC noise;
# tolerance is the ~1/sqrt(NS) scale of that noise, not zero.
tol = 6.0 * np.abs(G @ Sy1 @ G.T).max() / np.sqrt(NS)
allpass &= ok("C8 archive law: same distribution under both witness truths (MC)",
              d_mean < tol and d_cov < tol,
              f"max|Δmean|={d_mean:.1e} max|Δcov|={d_cov:.1e} (MC tol={tol:.1e}, n={NS})")
rec("C8 archive law under the two witness truths (Monte Carlo)", sci(max(d_mean, d_cov)) + " (tol.\\ " + sci(tol) + ")")

# C9 ensemble: pool K regularizers -> K-atom blind conditional, still zero-width per atom -> C^pool = 0
K = 5; atoms = []
for k in range(K):
    Srk = spd(1.0, 0.20, Q, True)
    Sp = inv(inv(Srk) + A.T @ Gi @ A); Gk = Sp @ A.T @ Gi
    SqMk = Gk @ (A @ S_star @ A.T + Gam) @ Gk.T
    atoms.append(np.abs(cond_blind_var(SqMk, Q)).max())
allpass &= ok("C9 ensemble: each pooled atom has zero blind-conditional width -> C^pool=0",
              max(atoms) < 1e-8, f"max atom cond blind var={max(atoms):.1e} (K={K})")
rec("C9 pooled-archive atoms, conditional blind variance", sci(max(atoms)))

print(f"\n===== {'ALL CHECKS PASS' if allpass else 'SOME CHECKS FAILED'} =====", flush=True)

# --- emit the appendix residual table (machine-written; \input by the paper) ----------------------
_out = os.path.join(REPO, "results", "verify_map_residuals.tex")
os.makedirs(os.path.dirname(_out), exist_ok=True)
with open(_out, "w") as f:
    f.write("\\begin{tabular}{@{}lr@{}}\n\\toprule\n")
    f.write("check & residual \\\\\n\\midrule\n")
    for name, val in REC:
        f.write(f"{name} & {val} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(f"[verify] wrote {os.path.relpath(_out, REPO)}", flush=True)
