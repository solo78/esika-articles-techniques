"""Expérience : provisionner à partir des mêmes bordereaux en ajoutant les couches une à une.

Oracle       : données parfaites, Chain-Ladder par partenaire (erreur propre à la méthode).
Brut         : bordereaux tels que reçus, Chain-Ladder.
Contrôle     : contrôle ponctuel sans mémoire à réception (doublons exacts supprimés, lignes
               invalides ou sans survenance écartées, bordereau rejeté si incident critique).
Redressement : Contrôle + exploitation du rapprochement comptable (couche 2) : chaque bordereau
               incomplet est remis à l'échelle du flux comptable du mois ; un bordereau non reçu
               à la clôture est estimé à partir du mois précédent, mis à l'échelle du flux comptable.
Benktander   : Redressement + méthode de Benktander classique (crédibilité = part déjà payée).
               C'est le témoin : la bonne pratique actuarielle, sans notion de confiance.
Réputation   : Redressement + Benktander dont la crédibilité est multipliée par la confiance Z du
               partenaire à la clôture (couche 3) ; l'a priori utilise le profil de développement
               des seuls partenaires jugés fiables (Z >= 0,5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import quality, reputation, reserving, simulation
from .simulation import N_MONTHS, PARTNERS

STRATEGIES = ["Oracle", "Brut", "Contrôle", "Redressement", "Benktander", "Réputation"]
# Sensibilité : même comparaison quand aucun flux comptable détaillé n'est disponible
SENSITIVITY = ["Benktander sans rapprochement", "Réputation sans rapprochement"]
ELR_PRIOR_SD = 0.10   # erreur de la sinistralité a priori utilisée par Bornhuetter-Ferguson
REDRESS_MIN_GAP = 0.02


def clean(rows: dict, keep_month: np.ndarray) -> dict:
    """Contrôle ponctuel : rejets de bordereaux, doublons exacts, lignes invalides ou incomplètes."""
    m = rows["pay"]
    _, first_idx = np.unique(rows["uid"], return_index=True)
    first = np.zeros(m.size, dtype=bool)
    first[first_idx] = True
    acc = rows["acc"]
    valid = (rows["amount"] > 0) & ~np.isnan(acc) & (np.nan_to_num(acc, nan=1e9) <= m)
    sel = keep_month[m] & first & valid
    return {k: v[sel] for k, v in rows.items()}


def redress(rows: dict, cash: np.ndarray, available: np.ndarray, active: np.ndarray) -> dict:
    """Remise à l'échelle du flux comptable, mois par mois, et imputation des mois non reçus."""
    rows = {k: v.copy() for k, v in rows.items()}
    m = rows["pay"]
    total = np.bincount(m, rows["amount"], minlength=N_MONTHS + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where(cash > 0, np.abs(total - cash) / cash, 0.0)
        factor = np.where((total > 0) & (gap > REDRESS_MIN_GAP), cash / total, 1.0)
    rows["amount"] = rows["amount"] * factor[m]
    extra = []
    for c in range(1, N_MONTHS + 1):
        if active[c] and available[c] and total[c] == 0 and cash[c] > 0:
            available = available.copy(); available[c] = False   # bordereau rejeté : on l'impute
        if active[c] and not available[c] and cash[c] > 0:
            prev = rows["pay"] == c - 1
            if prev.sum() == 0:
                continue
            tot_prev = rows["amount"][prev].sum()
            extra.append({
                "claim": rows["claim"][prev], "uid": rows["uid"][prev],
                "acc": rows["acc"][prev] + 1, "pay": rows["pay"][prev] + 1,
                "amount": rows["amount"][prev] * cash[c] / tot_prev,
            })
    for e in extra:
        for k in rows:
            rows[k] = np.r_[rows[k], e[k]]
    return rows


def run_one(seed: int, keep_details: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    data = {}
    for p in PARTNERS:
        truth = simulation.simulate_payments(p, rng)
        tx = simulation.transmit(p, truth, rng)
        scores = quality.evaluate(tx["rows"], tx["cash"], tx["delay"], tx["active"])
        monitor = reputation.timeline(scores)                              # suivi en continu
        at_close = reputation.timeline(scores, available=tx["available"])  # vue à la clôture
        Z = float(at_close["Z"].dropna().iloc[-1])

        in_scope = truth["acc"] >= 1
        ult_true = truth["amount"][in_scope].sum()
        ult_true_q = np.bincount(((truth["acc"][in_scope] - 1) // 3).astype(int),
                                 truth["amount"][in_scope], minlength=reserving.NQ)
        paid_true = truth["amount"][in_scope & (truth["pay"] <= N_MONTHS)].sum()

        rows, avail = tx["rows"], tx["available"]
        raw = {k: v[avail[rows["pay"]]] for k, v in rows.items()}
        crit = scores["critical"].reindex(range(N_MONTHS + 1), fill_value=False).to_numpy().astype(bool)
        ctrl = clean(rows, avail & ~crit)
        # pour le redressement, le bordereau critique n'est pas jeté : il est remis à l'échelle
        red = redress(clean(rows, avail), tx["cash"], avail, tx["active"])
        tri = {
            "Oracle": reserving.triangle(truth["acc"], truth["pay"], truth["amount"]),
            "Brut": reserving.triangle(raw["acc"], raw["pay"], raw["amount"]),
            "Contrôle": reserving.triangle(ctrl["acc"], ctrl["pay"], ctrl["amount"]),
            "Redressement": reserving.triangle(red["acc"], red["pay"], red["amount"]),
        }
        data[p.code] = {
            "partner": p, "tri": tri, "Z": Z, "ult_true": ult_true, "paid_true": paid_true,
            "ult_true_q": ult_true_q,
            "premium": simulation.premiums_by_quarter(p),
            "elr_prior": p.loss_ratio * float(np.exp(rng.normal(0, ELR_PRIOR_SD))),
            "scores": scores, "monitor": monitor, "at_close": at_close,
        }

    codes = [p.code for p in PARTNERS]
    f_pool = {s: reserving.pooled_factors([data[c]["tri"][s] for c in codes])
              for s in ["Oracle", "Brut", "Contrôle", "Redressement"]}
    trusted = [c for c in codes if data[c]["Z"] >= 0.5] or codes
    f_trusted = reserving.pooled_factors([data[c]["tri"]["Redressement"] for c in trusted])
    f_trusted_ctrl = reserving.pooled_factors([data[c]["tri"]["Contrôle"] for c in trusted])

    records, quarters = [], {}
    for c in codes:
        d = data[c]
        est_q = {s: reserving.chain_ladder(d["tri"][s], f_pool[s])
                 for s in ["Oracle", "Brut", "Contrôle", "Redressement"]}
        t = d["tri"]["Redressement"]
        est_q["Benktander"] = reserving.benktander_trust(
            t, f_pool["Redressement"], f_pool["Redressement"], d["premium"], d["elr_prior"], 1.0)
        est_q["Réputation"] = reserving.benktander_trust(
            t, f_pool["Redressement"], f_trusted, d["premium"], d["elr_prior"], d["Z"])
        quarters[c] = {s: (v - d["ult_true_q"]) / np.where(d["ult_true_q"] > 0, d["ult_true_q"], np.nan)
                       for s, v in est_q.items()}
        est = {s: v.sum() for s, v in est_q.items()}
        tc = d["tri"]["Contrôle"]
        est["Benktander sans rapprochement"] = reserving.benktander_trust(
            tc, f_pool["Contrôle"], f_pool["Contrôle"], d["premium"], d["elr_prior"], 1.0).sum()
        est["Réputation sans rapprochement"] = reserving.benktander_trust(
            tc, f_pool["Contrôle"], f_trusted_ctrl, d["premium"], d["elr_prior"], d["Z"]).sum()
        reserve_true = d["ult_true"] - d["paid_true"]
        for s in STRATEGIES + SENSITIVITY:
            records.append({"seed": seed, "partner": c, "strategy": s, "ult_hat": est[s],
                            "ult_true": d["ult_true"], "reserve_true": reserve_true, "Z": d["Z"]})

    # détection des dérives et fausses alertes (suivi en continu)
    def first_alert(code, start):
        mon = data[code]["monitor"].loc[start:]
        hit = mon.index[mon["status"].isin(["orange", "rouge"])]
        return int(hit[0]) if len(hit) else None

    def first_reject(code, start):
        s = data[code]["scores"].loc[start:]
        hit = s.index[s["critical"].astype(bool)]
        return int(hit[0]) if len(hit) else None

    false_alerts = {}
    for c in ["P1", "P2", "P4", "P6"]:
        start = max(PARTNERS[codes.index(c)].start_month, 1)
        mon = data[c]["monitor"].dropna(subset=["q05"]).loc[start + 6:]   # hors démarrage à froid
        false_alerts[c] = float((mon["status"] != "vert").mean())

    out = {
        "records": records,
        "alert": {"P3": first_alert("P3", 19), "P8": first_alert("P8", 24)},
        "reject": {"P3": first_reject("P3", 19), "P8": first_reject("P8", 24)},
        "false_alerts": false_alerts,
        "Z": {c: data[c]["Z"] for c in codes},
        "quarters": quarters,
    }
    if keep_details:
        out["details"] = data
    return out


def summarize(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = pd.DataFrame([r for res in results for r in res["records"]])
    rec["err"] = rec["ult_hat"] - rec["ult_true"]
    rec["rel_err"] = rec["err"] / rec["reserve_true"]
    port = rec.groupby(["seed", "strategy"])[["err", "reserve_true"]].sum().reset_index()
    port["rel_err"] = port["err"] / port["reserve_true"]
    return rec, port
