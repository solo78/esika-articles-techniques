"""Expérience : provisionner à partir des mêmes bordereaux en ajoutant les couches une à une.

Oracle       : données parfaites, Chain-Ladder par partenaire (erreur propre à la méthode).
Brut         : bordereaux tels que reçus, Chain-Ladder.
Contrôle     : contrôle ponctuel sans mémoire à réception (doublons exacts supprimés, lignes
               invalides ou sans survenance écartées, bordereau rejeté si son total s'écarte de
               plus de 30 % du flux comptable du mois).
Redressement : Contrôle + exploitation du rapprochement comptable (couche 2) : chaque bordereau
               incomplet est remis à l'échelle du flux comptable du mois ; un bordereau non reçu
               à la clôture est estimé à partir du mois précédent, mis à l'échelle du flux comptable.
Benktander   : Redressement + Benktander (crédibilité = part déjà payée selon le profil mutualisé ;
               partie Chain-Ladder sur les facteurs propres du partenaire, complétés si besoin).
               C'est le témoin : la bonne pratique actuarielle, sans notion de confiance.
Réputation   : même calcul, la crédibilité étant multipliée par la confiance Z du partenaire à la
               clôture (couche 3). Avec Z = 1, on retrouve exactement le témoin.

Sensibilités : flux comptable bruité (5 % par mois) ; a priori de sinistralité biaisé de -15 % ;
               absence de flux comptable détaillé (pas de redressement).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import quality, reputation, reserving, simulation
from .simulation import N_MONTHS, PARTNERS

STRATEGIES = ["Oracle", "Brut", "Contrôle", "Redressement", "Benktander", "Réputation"]
# Sensibilité : même comparaison quand aucun flux comptable détaillé n'est disponible
SENSITIVITY = ["Redressement, flux bruité", "Benktander, a priori -15 %", "Réputation, a priori -15 %",
               "Benktander sans rapprochement", "Réputation sans rapprochement"]
CASH_NOISE_SD = 0.05  # bruit du flux comptable observé (sensibilité)
PRIOR_BIAS = 0.85     # a priori de sinistralité sous-estimé de 15 % (sensibilité)
ALERT_SCORE = 0.85    # témoin de détection : alerte dès qu'une note atomique passe sous ce seuil
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
        # contrôle ponctuel sans mémoire : seul le rapprochement comptable du mois est utilisé
        gap = scores["gap"].reindex(range(N_MONTHS + 1), fill_value=0.0).to_numpy()
        reject = gap > quality.CRITICAL_GAP
        ctrl = clean(rows, avail & ~reject)
        # pour le redressement, le bordereau critique n'est pas jeté : il est remis à l'échelle
        red = redress(clean(rows, avail), tx["cash"], avail, tx["active"])
        cash_noisy = tx["cash"] * np.exp(rng.normal(0, CASH_NOISE_SD, tx["cash"].size))
        red_noisy = redress(clean(rows, avail), cash_noisy, avail, tx["active"])
        tri = {
            "Oracle": reserving.triangle(truth["acc"], truth["pay"], truth["amount"]),
            "Brut": reserving.triangle(raw["acc"], raw["pay"], raw["amount"]),
            "Contrôle": reserving.triangle(ctrl["acc"], ctrl["pay"], ctrl["amount"]),
            "Redressement": reserving.triangle(red["acc"], red["pay"], red["amount"]),
            "Redressement, flux bruité": reserving.triangle(red_noisy["acc"], red_noisy["pay"],
                                                            red_noisy["amount"]),
        }
        data[p.code] = {
            "partner": p, "tri": tri, "Z": Z, "ult_true": ult_true, "paid_true": paid_true,
            "ult_true_q": ult_true_q,
            "premium": simulation.premiums_by_quarter(p),
            "elr_prior": p.loss_ratio * float(np.exp(rng.normal(0, ELR_PRIOR_SD))),
            "scores": scores, "monitor": monitor, "at_close": at_close, "reject": reject,
        }

    codes = [p.code for p in PARTNERS]
    f_pool = {s: reserving.pooled_factors([data[c]["tri"][s] for c in codes])
              for s in ["Oracle", "Brut", "Contrôle", "Redressement", "Redressement, flux bruité"]}

    records, quarters = [], {}
    for c in codes:
        d = data[c]
        est_q = {s: reserving.chain_ladder(d["tri"][s], f_pool[s])
                 for s in ["Oracle", "Brut", "Contrôle", "Redressement"]}
        t, fr = d["tri"]["Redressement"], f_pool["Redressement"]
        est_q["Benktander"] = reserving.benktander_trust(t, fr, fr, d["premium"], d["elr_prior"], 1.0)
        est_q["Réputation"] = reserving.benktander_trust(t, fr, fr, d["premium"], d["elr_prior"], d["Z"])
        quarters[c] = {s: (v - d["ult_true_q"]) / np.where(d["ult_true_q"] > 0, d["ult_true_q"], np.nan)
                       for s, v in est_q.items()}
        est = {s: v.sum() for s, v in est_q.items()}
        est["Redressement, flux bruité"] = reserving.chain_ladder(
            d["tri"]["Redressement, flux bruité"], f_pool["Redressement, flux bruité"]).sum()
        elr_b = d["elr_prior"] * PRIOR_BIAS
        est["Benktander, a priori -15 %"] = reserving.benktander_trust(
            t, fr, fr, d["premium"], elr_b, 1.0).sum()
        est["Réputation, a priori -15 %"] = reserving.benktander_trust(
            t, fr, fr, d["premium"], elr_b, d["Z"]).sum()
        tc, fc = d["tri"]["Contrôle"], f_pool["Contrôle"]
        est["Benktander sans rapprochement"] = reserving.benktander_trust(
            tc, fc, fc, d["premium"], d["elr_prior"], 1.0).sum()
        est["Réputation sans rapprochement"] = reserving.benktander_trust(
            tc, fc, fc, d["premium"], d["elr_prior"], d["Z"]).sum()
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
        r = data[code]["reject"]
        hit = [m for m in range(start, N_MONTHS + 1) if r[m]]
        return hit[0] if hit else None

    def first_threshold(code, start):
        s = data[code]["scores"].loc[start:]
        hit = s.index[(s["score"] < ALERT_SCORE) | s["critical"].astype(bool)]
        return int(hit[0]) if len(hit) else None

    # fausses alertes : partenaires sans défaut structurel (P4 est inclus : ses doublons exacts
    # sont corrigés automatiquement, ils ne doivent pas déclencher d'alerte)
    false_alerts, false_alerts_thr = {}, {}
    for c in ["P1", "P2", "P4", "P6", "P5"]:
        start = max(PARTNERS[codes.index(c)].start_month, 1)
        mon = data[c]["monitor"].dropna(subset=["q05"]).loc[start + 6:]   # hors démarrage à froid
        false_alerts[c] = float((mon["status"] != "vert").mean())
        sc = data[c]["scores"].dropna(subset=["score"]).loc[start + 6:]
        false_alerts_thr[c] = float(((sc["score"] < ALERT_SCORE) | sc["critical"].astype(bool)).mean())

    # stabilité du signal après le début d'une dérive : nombre de bascules alerte / pas d'alerte
    def flips(code, start):
        mon = data[code]["monitor"].loc[start:]
        rep_alert = (mon["status"] != "vert").astype(int).to_numpy()
        sc = data[code]["scores"].loc[start:]
        thr_alert = ((sc["score"] < ALERT_SCORE) | sc["critical"].astype(bool)).astype(int).to_numpy()
        return {"reputation": int(np.abs(np.diff(rep_alert)).sum()),
                "seuil_note": int(np.abs(np.diff(thr_alert)).sum()),
                "mois_sans_alerte_apres_1re_alerte_seuil": int(
                    (thr_alert[np.argmax(thr_alert):] == 0).sum()) if thr_alert.any() else None,
                "mois_sans_alerte_apres_1re_alerte_rep": int(
                    (rep_alert[np.argmax(rep_alert):] == 0).sum()) if rep_alert.any() else None}

    # retour au vert du nouvel apporteur et statut de P8 en fin de période
    mon6 = data["P6"]["monitor"].dropna(subset=["q05"])
    green6 = mon6.index[mon6["status"] == "vert"]
    p8_status = data["P8"]["monitor"]["status"].iloc[-1]
    p8_worst = data["P8"]["monitor"]["q05"].loc[24:].min()

    out = {
        "records": records,
        "alert": {"P3": first_alert("P3", 19), "P8": first_alert("P8", 24)},
        "reject": {"P3": first_reject("P3", 19), "P8": first_reject("P8", 24)},
        "threshold": {"P3": first_threshold("P3", 19), "P8": first_threshold("P8", 24)},
        "false_alerts": false_alerts,
        "false_alerts_thr": false_alerts_thr,
        "P6_first_green": int(green6[0]) if len(green6) else None,
        "P6_start_status": mon6["status"].iloc[0],
        "flips": {"P3": flips("P3", 19), "P8": flips("P8", 24)},
        "P8_end_status": p8_status, "P8_min_q05": float(p8_worst),
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
