"""Génération des triangles fictifs.

Reprend le générateur publié par McGuire, Taylor et Miller (2021) et réutilisé par
Taylor et McGuire (2023), section 8.1 : triangle trimestriel 40 x 40, effets de
survenance (alpha), de développement (beta) et de paiement (gamma, inflation
superposée), paiements lognormaux de variance proportionnelle à la moyenne.
Source du code R d'origine : blog MLR de l'Institute and Faculty of Actuaries
(« Foundations : LASSO », section « generating the synthetic data set »).

Simplification : la graine est celle de numpy, pas celle de R ; le tirage diffère
donc de celui du papier, mais les espérances (et la provision vraie) sont identiques.
"""
import numpy as np


def linear_spline(x, a, b):
    """LinearSpline(x, a, b) du code R : rampe nulle avant a, pente 1 entre a et b, plate après b."""
    return np.minimum(np.maximum(np.asarray(x, float) - a, 0.0), b - a)


def gamma_si(t):
    """Effet de paiement (inflation superposée) gammafunc(t) du générateur.

    Constant sur les 12 premiers trimestres (pente 0,75 %), accélère de 13 à 24,
    constant de 25 à 32, accélère de nouveau de 33 à 40, puis FIGÉ après t = 40.
    Ce dernier point compte : le futur vrai ne contient plus d'inflation superposée.
    """
    t = np.asarray(t, float)
    b1 = 0.0075 * linear_spline(12, 1, 12)
    b2 = b1 + 0.001 * (24 - 12) * (24 - 11) / 2
    b3 = b2 + 0.002 * (40 - 32) * (40 - 31) / 2
    return np.where(t <= 12, 0.0075 * linear_spline(t, 1, 12),
           np.where(t <= 24, b1 + 0.001 * (t - 12) * (t - 11) / 2,
           np.where(t <= 32, b2,
           np.where(t <= 40, b2 + 0.002 * (t - 32) * (t - 31) / 2, b3))))


def grid(n):
    """Indices de survenance k (i du papier), développement j et paiement t = k + j - 1."""
    k, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    return k, j, k + j - 1


def means(dataset, n=40):
    """Espérances mu_ij des jeux 1 et 2 (papier, section 8.1)."""
    k, j, t = grid(n)
    alpha = np.log(1e5) + 0.1 * linear_spline(k, 1, 15) + 0.2 * linear_spline(k, 15, 20) \
        - 0.05 * linear_spline(k, 30, 40)
    beta = (16 / 3 - 1) * np.log(j) - (1 / 3) * j
    if dataset == 1:
        g = 0.0
    elif dataset == 2:
        g = gamma_si(t)
    else:
        raise ValueError("seuls les jeux 1 et 2 sont utilisés ici")
    return np.exp(alpha + beta + g)


def simulate(mu, rng, cv_ref=0.3, ref=(0, 15)):
    """Paiements lognormaux de variance CC * mu, CC calé pour un CV de 30 % en (k=1, j=16)."""
    varbase = (cv_ref * mu[ref]) ** 2
    cc = varbase / mu[ref]
    var = cc * mu
    tau2 = np.log(var / mu ** 2 + 1)
    return np.exp(rng.normal(np.log(mu) - tau2 / 2, np.sqrt(tau2)))


def triangle(dataset, seed, n=40):
    """Renvoie (y complet n x n, mu, masque du passé t <= n). Le futur simulé sert au contrôle de réalisme."""
    mu = means(dataset, n)
    rng = np.random.default_rng(seed)
    y = simulate(mu, rng)
    _, _, t = grid(n)
    return y, mu, t <= n


def fil_rouge(seed=2026):
    """Petit triangle annuel 10 x 10 du fil rouge (paramètres choisis pour la lisibilité, unités fictives).

    Survenance : +5 % par an à partir de l'an 4 ; développement : pic en année 3 ;
    inflation superposée : +3 % par année de paiement jusqu'à l'année 10 ; CV de 10 %.
    """
    n = 10
    k, j, t = grid(n)
    mu = np.exp(np.log(1000) + 0.05 * linear_spline(k, 3, 10) + 2.0 * np.log(j) - 0.65 * j
                + 0.03 * linear_spline(t, 1, 10))
    rng = np.random.default_rng(seed)
    y = simulate(mu, rng, cv_ref=0.10, ref=(0, 2))
    return y, mu, t <= n
