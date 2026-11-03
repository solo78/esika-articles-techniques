"""Simulation de partenaires qui transmettent des bordereaux mensuels de paiements de sinistres.

Chaque partenaire (courtier, bancassureur, gestionnaire délégué, coassureur...) gère un
portefeuille de sinistres. Le simulateur produit :
  * la vérité : tous les paiements, y compris futurs, ce qui donne la charge ultime exacte ;
  * les bordereaux réellement reçus, dégradés selon le profil de qualité du partenaire ;
  * le total comptable de chaque mois (flux de trésorerie réel), qui sert au rapprochement.

Toutes les données sont synthétiques. Les montants sont exprimés en milliers d'unités monétaires.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_MONTHS = 36        # fenêtre observée : mois 1 à 36 (3 ans, 12 trimestres)
PRE_HISTORY = 48     # sinistres antérieurs simulés pour obtenir un portefeuille à maturité
HORIZON = 120        # dernier mois de paiement simulé (sert à connaître la charge ultime)
SEV_MEAN = 1.0       # coût moyen d'un sinistre
SEV_SIGMA = 1.0      # dispersion (log-normale)
CLOSING_DAYS = 15    # jours après la fin du mois 36 pour arrêter les comptes
DEADLINE_DAYS = 10   # délai contractuel de transmission d'un bordereau


@dataclass(frozen=True)
class Partner:
    code: str
    label: str
    profile: str
    claims_per_month: float
    loss_ratio: float
    start_month: int = 1 - PRE_HISTORY  # premier mois de survenance


PARTNERS: list[Partner] = [
    Partner("P1", "Courtier A", "Fiable", 60, 0.62),
    Partner("P2", "Courtier B", "Fiable, petits défauts", 45, 0.68),
    Partner("P3", "Bancassureur", "Dégradation progressive dès le mois 19", 70, 0.58),
    Partner("P4", "Courtier C", "Doublons chroniques", 40, 0.70),
    Partner("P5", "Gestionnaire santé", "Retards fréquents", 55, 0.75),
    Partner("P6", "Nouvel apporteur", "Entrée au mois 25", 50, 0.65, start_month=25),
    Partner("P7", "Coassureur", "Erreur d'unité au mois 27", 35, 0.60),
    Partner("P8", "Gestionnaire auto", "Mauvais codage de la survenance dès le mois 24", 50, 0.72),
]


def defect_schedule(p: Partner) -> dict[str, np.ndarray]:
    """Taux de défauts par mois de transmission (indices 0 à N_MONTHS, l'indice 0 est inutilisé)."""
    m = np.arange(N_MONTHS + 1)
    z = np.zeros(N_MONTHS + 1)
    s = {k: z.copy() for k in ["drop", "dup", "miss_acc", "miss_id", "neg", "shift", "late_prob"]}
    s["unit_error"] = np.zeros(N_MONTHS + 1, dtype=bool)
    s["late_min"], s["late_max"] = 11, 15

    if p.code == "P1":
        s["miss_id"][:] = 0.002
    elif p.code == "P2":
        s["dup"][:] = 0.005
        s["miss_acc"][:] = 0.005
        s["miss_id"][:] = 0.01
        s["neg"][:] = 0.003
        s["late_prob"][:] = 0.05
        s["late_min"], s["late_max"] = 12, 18
    elif p.code == "P3":
        s["miss_id"][:] = 0.005
        ramp = np.clip((m - 18) / 12, 0, 1)  # 0 jusqu'au mois 18, 1 à partir du mois 30
        s["drop"][:] = 0.15 * ramp
        s["dup"][:] = 0.06 * ramp
        s["miss_acc"][:] = 0.08 * ramp
    elif p.code == "P4":
        s["dup"][:] = 0.06
        s["miss_id"][:] = 0.02
    elif p.code == "P5":
        s["drop"][:] = 0.01
        s["late_prob"][:] = 0.35
        s["late_min"], s["late_max"] = 16, 50
    elif p.code == "P6":
        s["miss_id"][:] = 0.005
        s["dup"][:] = 0.003
    elif p.code == "P7":
        s["miss_id"][:] = 0.003
        s["unit_error"][27] = True  # bordereau transmis en milliers au lieu d'unités
    elif p.code == "P8":
        s["miss_id"][:] = 0.005
        s["shift"][m >= 24] = 0.25  # date de survenance remplacée par la date de paiement
    return s


def simulate_payments(p: Partner, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Paiements exacts de tous les sinistres du partenaire."""
    acc_months = np.arange(p.start_month, N_MONTHS + 1)
    n_claims = rng.poisson(p.claims_per_month, size=acc_months.size)
    acc = np.repeat(acc_months, n_claims)
    nc = acc.size
    mu = np.log(SEV_MEAN) - SEV_SIGMA**2 / 2
    severity = rng.lognormal(mu, SEV_SIGMA, nc)

    k = 1 + np.minimum(rng.poisson(1.2, nc), 5)          # nombre de paiements par sinistre
    total = int(k.sum())
    claim_idx = np.repeat(np.arange(nc), k)
    first = np.r_[True, claim_idx[1:] != claim_idx[:-1]]
    # premier paiement : délai moyen d'un mois ; paiements suivants : délai moyen de 3 mois
    inc = np.where(first, rng.geometric(0.5, total) - 1, rng.geometric(1 / 3, total))
    cs = np.cumsum(inc)
    starts = np.flatnonzero(first)
    delay = cs - np.repeat((cs - inc)[starts], k)

    g = rng.gamma(1.0, 1.0, total)                        # répartition aléatoire du coût
    share = g / np.bincount(claim_idx, g)[claim_idx]
    amount = severity[claim_idx] * share
    pay = acc[claim_idx] + delay

    keep = pay <= HORIZON
    return {
        "claim": claim_idx[keep].astype(np.int64),
        "acc": acc[claim_idx][keep].astype(float),
        "pay": pay[keep].astype(np.int64),
        "amount": amount[keep],
    }


def premiums_by_quarter(p: Partner) -> np.ndarray:
    """Primes acquises par trimestre de survenance (1 à 12)."""
    monthly = p.claims_per_month * SEV_MEAN / p.loss_ratio
    prem = np.zeros(N_MONTHS // 3)
    for q in range(N_MONTHS // 3):
        months = np.arange(3 * q + 1, 3 * q + 4)
        prem[q] = monthly * np.sum(months >= p.start_month)
    return prem


def transmit(p: Partner, truth: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    """Construit les bordereaux reçus (lignes dégradées), les totaux comptables et les délais."""
    sched = defect_schedule(p)
    window = (truth["pay"] >= 1) & (truth["pay"] <= N_MONTHS)
    rows = {k: v[window] for k, v in truth.items()}
    rows["uid"] = np.flatnonzero(window)                  # identifiant technique de la ligne d'origine
    m = rows["pay"]
    n = m.size

    cash = np.bincount(m, rows["amount"], minlength=N_MONTHS + 1)   # vérité comptable

    u = rng.random((2, n))
    keep = u[0] >= sched["drop"][m]
    dup = keep & (u[1] < sched["dup"][m])
    idx = np.r_[np.flatnonzero(keep), np.flatnonzero(dup)]
    rec = {k: v[idx].copy() for k, v in rows.items()}
    mm = rec["pay"]
    nr = mm.size

    v = rng.random((4, nr))
    rec["acc"][v[0] < sched["miss_acc"][mm]] = np.nan
    rec["claim"][v[1] < sched["miss_id"][mm]] = -1
    rec["amount"][v[2] < sched["neg"][mm]] *= -1
    shift = v[3] < sched["shift"][mm]
    rec["acc"][shift & ~np.isnan(rec["acc"])] = mm[shift & ~np.isnan(rec["acc"])]
    rec["amount"][sched["unit_error"][mm]] /= 1000.0

    late = rng.random(N_MONTHS + 1) < sched["late_prob"]
    delay = np.where(
        late,
        rng.integers(sched["late_min"], sched["late_max"] + 1, N_MONTHS + 1),
        rng.integers(3, 9, N_MONTHS + 1),
    )
    active = np.arange(N_MONTHS + 1) >= max(1, p.start_month)
    delay[~active] = 0
    # bordereau du mois c disponible à la clôture si le délai de transmission est tenu
    days_until_closing = (N_MONTHS - np.arange(N_MONTHS + 1)) * 30 + CLOSING_DAYS
    available = delay <= days_until_closing

    return {"rows": rec, "cash": cash, "delay": delay, "available": available, "active": active}
