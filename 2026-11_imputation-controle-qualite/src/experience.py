"""Une simulation complète : données, contrôles, mesures."""
from __future__ import annotations

import copy

import numpy as np

from .controles import appliquer_chaine
from .evaluation import ImputeurEngagement, corriger, engagement, metriques, precision_moyenne
from .simulation import simuler

TRAITEMENTS = ["C0", "C1", "C2", "C3", "C4"]


def fusion(base: dict, surcharge: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in surcharge.items():
        if isinstance(v, dict):
            out[k] = fusion(out.get(k, {}), v)
        else:
            out[k] = v
    return out


def une_simulation(cfg: dict, scenario: str, i: int) -> list[dict]:
    graine = cfg["simulation"]["graine_base"] + i  # mêmes graines pour tous les scénarios
    d = simuler(cfg, graine)
    verite = d["anormal"]
    res = appliquer_chaine(d["observe"], cfg, graine)
    lignes = []

    base = {"scenario": scenario, "sim": i, "n_anormaux": int(verite.sum()),
            "part_lignes_incompletes": float(d["observe"][["salaire", "cotisation", "anciennete"]].isna().any(axis=1).mean())}

    # Engagement : vérité, sans contrôle, chaque traitement, oracle (toutes erreurs corrigées)
    e_vrai = engagement(d["vrai"], cfg)
    imputeur = ImputeurEngagement(res["imputeurs"], res["colonnes"])
    memo = {}

    def erreur_engagement(revue):
        # Seules comptent les lignes erronées revues : deux traitements qui corrigent les
        # mêmes lignes donnent le même engagement (mise en cache).
        cle = np.flatnonzero(revue & d["anormal"]).tobytes()
        if cle not in memo:
            tirages = imputeur.completer(corriger(d["observe"], d["vrai"], d["champs_errones"], revue))
            memo[cle] = float(np.mean([engagement(t, cfg) for t in tirages]))
        return memo[cle] / e_vrai - 1.0

    n = len(verite)
    lignes.append({**base, "traitement": "Sans contrôle", "erreur_engagement": erreur_engagement(np.zeros(n, bool))})
    e_corr = erreur_engagement(np.ones(n, bool))
    lignes.append({**base, "traitement": "Toutes erreurs corrigées", "erreur_engagement": e_corr, "erreur_residuelle": 0.0})
    lignes[0]["erreur_residuelle"] = (1 + lignes[0]["erreur_engagement"]) / (1 + e_corr) - 1.0
    for t in TRAITEMENTS:
        m = metriques(res[t], verite)
        e_t = erreur_engagement(res[t])
        # erreur_residuelle : écart dû aux seules erreurs non détectées (la non-réponse est neutralisée)
        lignes.append({**base, "traitement": t, **m, "erreur_engagement": e_t,
                       "erreur_residuelle": (1 + e_t) / (1 + e_corr) - 1.0})

    # Pouvoir de classement des scores seuls (sans seuil)
    lignes.append({**base, "traitement": "score_IF_naif", "precision_moyenne": precision_moyenne(res["score_if_naif"], verite)})
    lignes.append({**base, "traitement": "score_IF_impute", "precision_moyenne": precision_moyenne(res["score_if_impute"], verite)})
    lignes.append({**base, "traitement": "score_PTC", "precision_moyenne": precision_moyenne(res["z_ptc"], verite)})
    lignes.append({**base, "traitement": "score_PTC_generique", "precision_moyenne": precision_moyenne(res["z_ptc_generique"], verite)})
    lignes.append({**base, "traitement": "C4_generique", **metriques(res["C4_generique"], verite)})
    # Précision à budget de revue identique (mêmes nombres de dossiers revus)
    b = int(round(cfg["evaluation"]["budget_comparaison"] * n))
    for nom, sc in [("IF_naif", res["score_if_naif"]), ("IF_impute", res["score_if_impute"]), ("PTC", res["z_ptc"])]:
        top = np.zeros(n, bool); top[np.argsort(-sc, kind="stable")[:b]] = True
        lignes.append({**base, "traitement": f"budget_{nom}", **metriques(top, verite)})

    # Sensibilité au seuil du predict-then-check (même simulation)
    for k in cfg["predict_then_check"]["seuils_sensibilite"]:
        m = metriques(res["C1"] | (res["z_ptc"] > k), verite)
        lignes.append({**base, "traitement": f"C4_k{k}", **m})

    # Oracle : mêmes erreurs, aucune valeur manquante
    if scenario == "base":
        vc = d["anormal_complet"]
        rc = appliquer_chaine(d["observe_complet"], cfg, graine, complet=True)
        for t in ["C1", "C2", "C4", "C4_generique"]:
            lignes.append({**base, "traitement": f"Oracle_{t}", **metriques(rc[t], vc)})
        lignes.append({**base, "traitement": "Oracle_score_PTC", "precision_moyenne": precision_moyenne(rc["z_ptc"], vc)})
        lignes.append({**base, "traitement": "Oracle_score_IF", "precision_moyenne": precision_moyenne(rc["score_if_naif"], vc)})
        lignes.append({**base, "traitement": "Oracle_score_PTC_generique", "precision_moyenne": precision_moyenne(rc["z_ptc_generique"], vc)})
        # Détail par type d'erreur (rappel), scénario de base uniquement
        for t in TRAITEMENTS + ["C4_generique"]:
            for typ in cfg["erreurs"]["types"]:
                mt = (d["type_erreur"] == typ) & verite
                if mt.sum():
                    lignes.append({**base, "traitement": f"rappel_type_{t}", "type_erreur": typ,
                                   "detectes": int((res[t] & mt).sum()), "total": int(mt.sum())})
        # Fausses alertes selon que la ligne est complète ou non
        saine = ~verite
        inc = d["observe"][["salaire", "cotisation", "anciennete"]].isna().any(axis=1).to_numpy()
        for t in TRAITEMENTS:
            for lab, m in [("incompletes", saine & inc), ("completes", saine & ~inc)]:
                lignes.append({**base, "traitement": f"fa_{t}", "groupe": lab,
                               "fausses_alertes": int((res[t] & m).sum()), "total": int(m.sum())})
    return lignes
