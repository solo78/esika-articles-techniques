"""Provisionnement : triangles trimestriels, Chain-Ladder, Bornhuetter-Ferguson et mélange par crédibilité."""
from __future__ import annotations

import numpy as np

from .simulation import N_MONTHS

NQ = N_MONTHS // 3  # 12 trimestres de survenance


def triangle(acc: np.ndarray, pay: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Triangle cumulé des paiements (survenance x développement, en trimestres)."""
    ok = ~np.isnan(acc)
    acc, pay, amount = acc[ok].astype(int), pay[ok], amount[ok]
    ok = (acc >= 1) & (acc <= N_MONTHS) & (pay <= N_MONTHS) & (pay >= acc)
    aq = (acc[ok] - 1) // 3
    dq = (pay[ok] - 1) // 3 - aq
    inc = np.zeros((NQ, NQ))
    np.add.at(inc, (aq, dq), amount[ok])
    cum = np.cumsum(inc, axis=1)
    mask = np.add.outer(np.arange(NQ), np.arange(NQ)) <= NQ - 1   # partie observée
    return np.where(mask, cum, np.nan)


def cl_factors(tri: np.ndarray) -> np.ndarray:
    """Facteurs de développement Chain-Ladder pondérés par les volumes (NaN si non estimable)."""
    f = np.full(NQ - 1, np.nan)
    for j in range(NQ - 1):
        rows = np.arange(NQ - 1 - j)              # lignes où j et j+1 sont observés
        num, den = np.nansum(tri[rows, j + 1]), np.nansum(tri[rows, j])
        if den > 0 and num > 0:
            f[j] = num / den
    return f


def latest(tri: np.ndarray) -> np.ndarray:
    """Paiements cumulés sur la dernière diagonale."""
    return np.array([tri[i, NQ - 1 - i] for i in range(NQ)])


def cdf_to_ult(f: np.ndarray) -> np.ndarray:
    """Coefficient cumulé pour aller de chaque âge de développement à l'ultime (sans queue)."""
    cdf = np.ones(NQ)
    for j in range(NQ - 2, -1, -1):
        cdf[j] = cdf[j + 1] * f[j]
    return cdf


def chain_ladder(tri: np.ndarray, f_fallback: np.ndarray) -> np.ndarray:
    f = cl_factors(tri)
    f = np.where(np.isnan(f), f_fallback, f)
    cdf = cdf_to_ult(f)
    ages = NQ - 1 - np.arange(NQ)
    return latest(tri) * cdf[ages]


def bornhuetter_ferguson(tri: np.ndarray, f_pattern: np.ndarray, premium: np.ndarray,
                         elr: float) -> np.ndarray:
    """Ultime BF = déjà payé + part restant à payer selon le profil x sinistralité a priori x prime."""
    cdf = cdf_to_ult(f_pattern)
    ages = NQ - 1 - np.arange(NQ)
    unpaid_share = 1 - 1 / cdf[ages]
    return latest(tri) + unpaid_share * elr * premium


def benktander_trust(tri: np.ndarray, f_fallback: np.ndarray, f_pattern: np.ndarray,
                     premium: np.ndarray, elr: float, trust: float = 1.0) -> np.ndarray:
    """Méthode de Benktander avec un facteur de confiance dans les données.

    Benktander classique : ultime = Z x CL + (1 - Z) x BF avec Z = part déjà payée (1 / CDF).
    Ici Z = confiance x part payée : plus la donnée du partenaire est jugée fiable,
    plus on suit son propre développement ; sinon on revient vers l'a priori (BF).
    trust = 1 redonne exactement la méthode de Benktander.
    """
    f = cl_factors(tri)
    f = np.where(np.isnan(f), f_fallback, f)
    ages = NQ - 1 - np.arange(NQ)
    cl = latest(tri) * cdf_to_ult(f)[ages]
    cdf_bf = cdf_to_ult(f_pattern)[ages]
    bf = latest(tri) + (1 - 1 / cdf_bf) * elr * premium
    z = trust / cdf_bf
    return z * cl + (1 - z) * bf


def pooled_factors(triangles: list[np.ndarray]) -> np.ndarray:
    return cl_factors(np.nansum(np.stack(triangles), axis=0) * _mask())


def _mask() -> np.ndarray:
    m = np.add.outer(np.arange(NQ), np.arange(NQ)) <= NQ - 1
    return np.where(m, 1.0, np.nan)
