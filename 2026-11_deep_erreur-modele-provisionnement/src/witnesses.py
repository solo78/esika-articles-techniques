"""Témoins de comparaison.

1. Chain ladder de Mack (1993) : pratique standard du marché. Erreur de paramètre + processus,
   modèle unique, pas d'erreur de modèle. Ce que le papier ne fait pas, mais que nous ajoutons.
2. GLM Gamma de structure connue, bootstrappé (papier, section 8.5, Table 8) : erreur de paramètre
   « pure » d'un modèle dont la forme est juste. Les covariables reprennent les fonctions du
   générateur, y compris l'arrêt de l'inflation superposée après t = 40.
"""
import numpy as np
import statsmodels.api as sm
from .data import grid, linear_spline
from . import bma as B


def mack(y, past):
    """Provision chain ladder et écart type de Mack (paramètre + processus) sur le triangle cumulé."""
    n = y.shape[0]
    inc = np.where(past, y, np.nan)
    C = np.nancumsum(np.where(past, y, 0.0), axis=1)
    C = np.where(past, C, np.nan)
    f = np.ones(n - 1)
    sig2 = np.zeros(n - 1)
    for j in range(n - 1):
        rows = np.arange(0, n - j - 1)
        f[j] = C[rows, j + 1].sum() / C[rows, j].sum()
        if len(rows) > 1:
            sig2[j] = np.sum(C[rows, j] * (C[rows, j + 1] / C[rows, j] - f[j]) ** 2) / (len(rows) - 1)
    for j in range(n - 1):  # extrapolation de Mack pour les dernières colonnes
        if j == n - 2 or sig2[j] == 0:
            sig2[j] = min(sig2[j - 1] ** 2 / sig2[j - 2], sig2[j - 2], sig2[j - 1]) if j >= 2 else sig2[j]
    full = C.copy()
    for i in range(1, n):
        for j in range(n - i, n):
            full[i, j] = full[i, j - 1] * f[j - 1]
    latest = np.array([C[i, n - 1 - i] for i in range(n)])
    R_i = full[:, -1] - latest
    R = R_i.sum()
    # MSEP total de Mack (1993), formule du total sur les survenances
    colsum = np.array([np.nansum(C[: n - j - 1, j]) for j in range(n - 1)])
    mse_i = np.zeros(n)
    for i in range(1, n):
        s = 0.0
        for j in range(n - i - 1, n - 1):
            s += sig2[j] / f[j] ** 2 * (1 / full[i, j] + 1 / colsum[j])
        mse_i[i] = full[i, -1] ** 2 * s
    tot = mse_i.sum()
    for i in range(1, n):
        for k in range(i + 1, n):
            s = sum(2 * sig2[j] / f[j] ** 2 / colsum[j] for j in range(n - i - 1, n - 1))
            tot += full[i, -1] * full[k, -1] * s
    return R, np.sqrt(tot) / R


def known_structure(n, dataset):
    """Covariables du générateur (section 8.5 : « structure connue »)."""
    k, j, t = grid(n)
    cols = [linear_spline(k, 1, 15), linear_spline(k, 15, 20), linear_spline(k, 30, 40), np.log(j), j]
    if dataset == 2:
        q1 = np.where(t <= 12, 0, np.where(t <= 24, (t - 12) * (t - 11) / 2, 78.0))
        q2 = np.where(t <= 32, 0, np.where(t <= 40, (t - 32) * (t - 31) / 2, 36.0))
        cols += [linear_spline(t, 1, 12), q1, q2]
    return sm.add_constant(np.column_stack([c.ravel().astype(float) for c in cols]))


def glm_bootstrap(y, past, n, dataset, B_, seed, gates=True, power=1, widen=1.0):
    """GLM Gamma (lien log) de structure connue, bootstrap semi-paramétrique identique au LASSO
    (mêmes résidus standardisés par mu^(power/2), même fonction de variance pour le processus).

    Les prévisions bootstrap passent les mêmes portes (Table 2) relativement au GLM du triangle observé."""
    X = known_structure(n, dataset)
    p = past.ravel()
    yp = y.ravel()[p]
    fam = sm.families.Gamma(sm.families.links.Log())
    g = sm.GLM(yp, X[p], family=fam).fit()
    mu_all = g.predict(X)
    mu = mu_all[p]
    phi = g.scale
    R0 = mu_all[~p].sum()
    rho = (yp - mu) / mu ** (power / 2)
    rho -= rho.mean()
    rng = np.random.default_rng(seed)
    ref = B.aggregates(mu_all, n)
    out = []
    for _ in range(B_):
        yb = np.maximum(mu + rho[rng.integers(0, len(rho), len(rho))] * mu ** (power / 2), 1e-3 * mu)
        gb = sm.GLM(yb, X[p], family=fam).fit()
        pb = gb.predict(X)
        if gates and not B.admissible(B.aggregates(pb, n), ref, widen):
            continue
        out.append(pb[~p].sum())
    out = np.array(out)
    if power == 2:
        w_pr = np.sqrt(phi * np.sum(mu_all[~p] ** 2)) / R0
    else:
        phi1 = np.sum(rho ** 2) / (len(rho) - X.shape[1])
        w_pr = np.sqrt(phi1 * np.sum(mu_all[~p])) / R0
    return dict(R0=R0, mean=out.mean(), w_pa=out.std(ddof=1) / out.mean(), n_kept=len(out), B=B_,
                phi=phi, w_pr=w_pr)
