"""Briques 2 à 5 : vraisemblance, a priori de Laplace, moyenne bayésienne et élagage.

Références au papier (Taylor et McGuire 2023) :
- vraisemblance Gamma : éq. (21), p. 12 ;
- a priori de Laplace sur les coefficients : éq. (17) et (22), p. 10 et 12 ;
- même a priori pour tous les modèles de la famille : éq. (25) et (27), p. 14-15 ;
- a posteriori sur les modèles : éq. (11), p. 9 ; moyenne bayésienne : éq. (12) et (26) ;
- IMSE = coefficient de variation de la provision sous l'a posteriori : éq. (10), section 6.2.6 ;
- portes d'inclusion : section 6.2.5, Table 2, p. 17.
"""
import numpy as np
from scipy.special import gammaln

# Table 2 : (fenêtre, borne basse, borne haute) en proportion de la prévision du modèle principal.
GATES_ACC = [(2, 0.75, 1.33), (5, 0.80, 1.25), (10, 0.83, 1.20)]   # derniers trimestres de survenance
GATES_PAY = [(2, 0.91, 1.10), (5, 0.87, 1.15), (10, 0.83, 1.20)]   # prochains trimestres de paiement
EPS = 0.0005                                                      # seuil des a priori simple et complexe


def gamma_loglik(y, mu, phi):
    """Éq. (21) : log-vraisemblance Gamma de dispersion phi (gamma = 1/phi, c = 1/(phi mu)).

    y : (npast,), mu : (..., npast). Renvoie (...,)."""
    g = 1.0 / phi
    c = 1.0 / (phi * mu)
    return np.sum(g * np.log(c) - gammaln(g) + (g - 1) * np.log(y) - c * y, axis=-1)


def log_prior(a0, l1, s, m0, s0=2.0):
    """Éq. (22) à une constante près : Laplace(m0, s0) sur l'intercept, Laplace(0, s) sur les autres.

    La constante -(p-1) log(2s) est commune à tous les modèles et disparaît à la normalisation.
    s est l'échelle de Laplace, inverse du paramètre lambda_0[G] du papier."""
    return -np.abs(a0 - m0) / s0 - l1 / s


def posterior(logw, admissible):
    """Éq. (11) : probabilités a posteriori normalisées sur les modèles admis (les autres à 0)."""
    lw = np.where(admissible, logw, -np.inf)
    lw = lw - np.nanmax(lw[admissible]) if admissible.any() else lw
    w = np.where(admissible, np.exp(lw), 0.0)
    w = np.nan_to_num(w)
    return w / w.sum() if w.sum() > 0 else w


def bma(w, reserves):
    """Éq. (12) et (10) : moyenne et variance a posteriori de la provision, CoV = IMSE."""
    ok = w > 0
    m = np.sum(w[ok] * reserves[ok])
    v = np.sum(w[ok] * (reserves[ok] - m) ** 2)
    return m, v, np.sqrt(v) / m


def aggregates(preds, n):
    """Agrégats futurs servant aux portes : par derniers trimestres de survenance et prochains de paiement.

    preds : (..., n*n). Renvoie dict de (...,)."""
    k, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    t = (k + j - 1).ravel()
    k = k.ravel()
    fut = t > n
    out = {}
    for m, _, _ in GATES_ACC:
        out[("acc", m)] = preds[..., fut & (k > n - m)].sum(-1)
    for m, _, _ in GATES_PAY:
        out[("pay", m)] = preds[..., (t > n) & (t <= n + m)].sum(-1)
    return out


def admissible(agg, agg_ref, widen=1.0):
    """Section 6.2.5 : le modèle passe toutes les portes (bornes élargies : basse / widen, haute x widen)."""
    ok = None
    for kind, gates in (("acc", GATES_ACC), ("pay", GATES_PAY)):
        for m, lo, hi in gates:
            r = agg[(kind, m)] / agg_ref[(kind, m)]
            cond = (r >= lo / widen) & (r <= hi * widen)
            ok = cond if ok is None else ok & cond
    return ok & np.isfinite(agg[("acc", GATES_ACC[0][0])])


def calibrate_scale(loglik, a0, l1, m0, adm, target, grid=None):
    """Échelle de Laplace s telle que le mode a posteriori tombe sur le modèle `target` (section 6.2.3).

    Les auteurs procèdent par essais successifs ; ici, balayage d'une grille logarithmique et choix du
    milieu (géométrique) de l'intervalle des s qui donnent ce mode. Renvoie (s, intervalle, atteint)."""
    grid = np.logspace(-4, 1, 2001) if grid is None else grid
    modes = np.array([np.nanargmax(np.where(adm, loglik + log_prior(a0, l1, s, m0), -np.inf)) for s in grid])
    hit = grid[modes == target]
    if len(hit):
        return float(np.sqrt(hit.min() * hit.max())), (float(hit.min()), float(hit.max())), True
    best = grid[np.argmin(np.abs(modes - target))]
    return float(best), (np.nan, np.nan), False


def calibrate_extreme(loglik, a0, l1, m0, adm, idx, kind):
    """Éq. (23) et (24) : a priori « simple » (masse eps sur les modèles au moins aussi complexes que le 1se)
    ou « complexe » (masse eps sur les modèles au plus aussi complexes que le minCV). Bissection sur log s."""
    q = np.arange(len(loglik))
    sel = (q >= idx) if kind == "simple" else (q <= idx)

    def mass(s):
        return posterior(loglik + log_prior(a0, l1, s, m0), adm)[sel].sum() - EPS

    lo, hi = np.log(1e-6), np.log(1e3)
    f_lo, f_hi = mass(np.exp(lo)), mass(np.exp(hi))
    if np.sign(f_lo) == np.sign(f_hi):
        return np.nan
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f = mass(np.exp(mid))
        if np.sign(f) == np.sign(f_lo):
            lo, f_lo = mid, f
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))
