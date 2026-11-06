"""Métriques actuarielles de fréquence, calculées indépendamment des agents.

- Déviance de Poisson moyenne (x100), la métrique de référence en fréquence.
- Gini « ordonné » (courbe de Lorenz : exposition cumulée vs sinistres cumulés,
  polices triées par fréquence prédite croissante).
- Ratio A/E (sinistres observés / attendus), global et par décile de prédiction.
- Bootstrap apparié de la différence de déviance entre deux modèles.
"""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy


def poisson_deviance(y, mu) -> float:
    """Déviance de Poisson moyenne par police, x100 (mu = nombre attendu de sinistres)."""
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-12, None)
    return float(200.0 * np.mean(xlogy(y, y / mu) - (y - mu)))


def gini(y, mu, expo) -> float:
    """Gini ordonné sur la fréquence prédite (mu / exposition)."""
    y, mu, expo = (np.asarray(a, float) for a in (y, mu, expo))
    order = np.argsort(mu / expo, kind="mergesort")
    cum_expo = np.concatenate([[0], np.cumsum(expo[order])]) / expo.sum()
    cum_y = np.concatenate([[0], np.cumsum(y[order])]) / y.sum()
    auc = np.trapezoid(cum_y, cum_expo)
    return float(1 - 2 * auc)


def ae_ratio(y, mu) -> float:
    return float(np.sum(y) / np.sum(mu))


def ae_by_decile(y, mu, expo, n=10):
    y, mu, expo = (np.asarray(a, float) for a in (y, mu, expo))
    freq = mu / expo
    order = np.argsort(freq, kind="mergesort")
    groups = np.array_split(order, n)
    return [float(y[g].sum() / mu[g].sum()) for g in groups]


def max_decile_gap(y, mu, expo) -> float:
    """Écart maximal |A/E - 1| sur les déciles : mesure de calibration locale."""
    return float(max(abs(r - 1) for r in ae_by_decile(y, mu, expo)))


def all_metrics(y, mu, expo) -> dict:
    return {
        "deviance_x100": poisson_deviance(y, mu),
        "gini": gini(y, mu, expo),
        "ae_global": ae_ratio(y, mu),
        "ae_max_decile_gap": max_decile_gap(y, mu, expo),
    }


def paired_bootstrap_deviance(y, mu_a, mu_b, n_boot=300, seed=0):
    """Différence de déviance (A - B) sur 300 rééchantillonnages appariés du jeu de test.

    Retourne la moyenne, l'IC à 95 % et la proportion de tirages où A < B.
    """
    rng = np.random.default_rng(seed)
    y, mu_a, mu_b = (np.asarray(a, float) for a in (y, mu_a, mu_b))
    mu_a = np.clip(mu_a, 1e-12, None)
    mu_b = np.clip(mu_b, 1e-12, None)
    la = xlogy(y, y / mu_a) - (y - mu_a)
    lb = xlogy(y, y / mu_b) - (y - mu_b)
    d = 200 * (la - lb)
    n = len(y)
    stats = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "diff_mean": float(d.mean()),
        "ci95": [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))],
        "share_a_better": float((stats < 0).mean()),
    }
