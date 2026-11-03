"""Couche 3 : de la note atomique à la réputation.

Modèle bayésien Beta avec oubli (survey, section V-B : « time-decay weighting » et
« Bayesian aggregation ») et déclenchement événementiel pour les incidents critiques
(« event-triggered state change »).

Intuition : la réputation est une proportion inconnue « part de bonnes transmissions ».
On la décrit par une loi Beta(alpha, beta). Chaque bordereau ajoute
    poids x note        à alpha  (preuves favorables)
    poids x (1 - note)  à beta   (preuves défavorables).
Avant chaque mise à jour, alpha et beta sont multipliés par un facteur d'oubli lambda < 1 :
les vieux bordereaux comptent moins, ce qui permet à la fois de détecter une dégradation
récente et de laisser un partenaire se racheter.

On ne décide pas sur la moyenne mais sur le quantile à 5 % : un partenaire nouveau ou
mal connu a une loi très étalée, donc un quantile bas, même si ses premières notes sont bonnes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

GREEN, AMBER = 0.85, 0.70     # seuils sur le quantile à 5 %
Z_LOW, Z_HIGH = 0.70, 0.90    # passage linéaire de Z = 0 à Z = 1


@dataclass
class BetaReputation:
    alpha: float = 1.0          # loi a priori uniforme : on ne sait rien d'un nouveau partenaire
    beta: float = 1.0
    lam: float = 0.85           # facteur d'oubli mensuel (mémoire effective ~ 1 / (1 - lam) ~ 7 mois)
    weight: float = 8.0         # poids d'un bordereau
    crit_weight: float = 24.0   # poids d'un incident critique (3 bordereaux défavorables)

    def update(self, score: float, critical: bool = False) -> None:
        self.alpha *= self.lam
        self.beta *= self.lam
        if critical:
            self.beta += self.crit_weight
        else:
            self.alpha += self.weight * score
            self.beta += self.weight * (1.0 - score)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def q05(self) -> float:
        return float(beta_dist.ppf(0.05, self.alpha, self.beta))

    @property
    def q95(self) -> float:
        return float(beta_dist.ppf(0.95, self.alpha, self.beta))


def status(q05: float) -> str:
    if q05 >= GREEN:
        return "vert"
    if q05 >= AMBER:
        return "orange"
    return "rouge"


def credibility(q05: float) -> float:
    """Poids Z accordé aux données propres du partenaire dans le provisionnement."""
    return float(np.clip((q05 - Z_LOW) / (Z_HIGH - Z_LOW), 0.0, 1.0))


def timeline(scores: pd.DataFrame, rep: BetaReputation | None = None,
             available: np.ndarray | None = None, fresh_weight: float = 0.15) -> pd.DataFrame:
    """Rejoue l'historique mois par mois.

    Si `available` est fourni (vue à la date de clôture), un bordereau non reçu est noté
    1 - poids de la fraîcheur : on sait seulement qu'il manque, les autres dimensions
    ne peuvent pas être évaluées.
    """
    rep = rep or BetaReputation()
    out = []
    for month, r in scores.iterrows():
        if not r["active"] or np.isnan(r["score"]):
            out.append({"month": month, "mean": np.nan, "q05": np.nan, "q95": np.nan,
                        "status": None, "Z": np.nan, "critical": False})
            continue
        if available is not None and not available[month]:
            rep.update(1.0 - fresh_weight)
            crit = False
        else:
            crit = bool(r["critical"])
            rep.update(float(r["score"]), critical=crit)
        q = rep.q05
        out.append({"month": month, "mean": rep.mean, "q05": q, "q95": rep.q95,
                    "status": status(q), "Z": credibility(q), "critical": crit})
    return pd.DataFrame(out).set_index("month")
