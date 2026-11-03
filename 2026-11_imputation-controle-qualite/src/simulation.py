"""Génération d'un recensement synthétique d'affiliés, injection d'erreurs et de non-réponse.

Trois couches de données sont produites pour chaque simulation :
- ``vrai`` : les valeurs exactes (inconnues en pratique) ;
- ``observe_complet`` : les valeurs saisies, erreurs comprises, sans aucune valeur manquante ;
- ``observe`` : ``observe_complet`` après non-réponse (NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIQUES = ["age", "anciennete", "salaire", "cotisation", "nb_mois"]


def generer_population(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    p = cfg["population"]
    n = cfg["simulation"]["n_affilies"]
    cats = np.array(p["categories"])
    idx_cat = rng.choice(len(cats), size=n, p=p["poids_categories"])
    age = rng.uniform(p["age_min"], p["age_max"], size=n)
    age_entree = p["age_entree_min"] + rng.gamma(p["age_entree_gamma_forme"], p["age_entree_gamma_echelle"], size=n)
    age_entree = np.minimum(age_entree, age - 0.5)
    anc = age - age_entree

    base = np.log(np.array(p["salaire_base_mensuel"], dtype=float))[idx_cat]
    if p.get("mal_specifie", False):
        # Pentes différentes par catégorie, plafond de carrière et queues épaisses :
        # le modèle de contrôle (additif, linéaire) est alors mal spécifié.
        pentes = np.array([0.010, 0.025, 0.045])[idx_cat]
        effet_anc = pentes * np.minimum(anc, 25) + 0.5 * pentes * np.maximum(anc - 25, 0) ** 0.5
        bruit = p["bruit_log_salaire"] * rng.standard_t(3, size=n) / np.sqrt(3.0)
    else:
        effet_anc = p["pente_anciennete"] * anc
        bruit = rng.normal(0.0, p["bruit_log_salaire"], size=n)
    log_sal = base + effet_anc + p["pente_age"] * (age - p["age_min"]) + bruit
    salaire = np.exp(log_sal)

    complet = rng.random(n) < p["proba_annee_complete"]
    nb_mois = np.where(complet, 12, rng.integers(1, 12, size=n)).astype(float)
    cotisation = p["taux_cotisation"] * salaire * nb_mois * np.exp(rng.normal(0.0, p["bruit_log_cotisation"], size=n))

    return pd.DataFrame({
        "categorie": cats[idx_cat],
        "age": np.round(age, 1),
        "anciennete": np.round(anc, 1),
        "salaire": np.round(salaire, -2),
        "cotisation": np.round(cotisation, 0),
        "nb_mois": nb_mois,
    })


def _poids_lignes_erreurs(cfg, manquant_ligne):
    conc = cfg["erreurs"].get("concentration_incomplets", 1.0)
    w = np.where(manquant_ligne, conc, 1.0)
    return w / w.sum()


def injecter_erreurs(vrai: pd.DataFrame, cfg: dict, rng: np.random.Generator, manquant_ligne: np.ndarray):
    """Renvoie (observe_complet, type_erreur, champs_errones) ; type_erreur vaut '' pour une ligne saine."""
    e = cfg["erreurs"]
    n = len(vrai)
    obs = vrai.copy()
    n_err = int(round(e["taux_lignes"] * n))
    lignes = rng.choice(n, size=n_err, replace=False, p=_poids_lignes_erreurs(cfg, manquant_ligne))
    types = rng.choice(e["types"], size=n_err)
    type_erreur = np.array([""] * n, dtype=object)
    champs = {c: np.zeros(n, dtype=bool) for c in NUMERIQUES}
    lo_i, hi_i = e["facteur_salaire_incoherent"]
    lo_d, hi_d = e["facteur_declaration"]
    lo_a, hi_a = e["decalage_annees"]
    for i, t in zip(lignes, types):
        type_erreur[i] = t
        signe = rng.choice([-1.0, 1.0])
        if t == "unite_salaire":
            obs.at[i, "salaire"] = vrai.at[i, "salaire"] * (1000.0 if signe > 0 else 0.001)
            champs["salaire"][i] = True
        elif t == "salaire_incoherent":
            obs.at[i, "salaire"] = round(vrai.at[i, "salaire"] * rng.uniform(lo_i, hi_i) ** signe, -2)
            champs["salaire"][i] = True
        elif t == "declaration_coherente":
            f = rng.uniform(lo_d, hi_d) ** signe
            obs.at[i, "salaire"] = round(vrai.at[i, "salaire"] * f, -2)
            obs.at[i, "cotisation"] = round(vrai.at[i, "cotisation"] * f, 0)
            champs["salaire"][i] = True
            champs["cotisation"][i] = True
        elif t == "date_naissance":
            obs.at[i, "age"] = round(vrai.at[i, "age"] + signe * rng.uniform(lo_a, hi_a), 1)
            champs["age"][i] = True
        elif t == "date_embauche":
            obs.at[i, "anciennete"] = round(max(vrai.at[i, "anciennete"] + signe * rng.uniform(lo_a, hi_a), 0.0), 1)
            champs["anciennete"][i] = True
    return obs, type_erreur, champs


def masque_non_reponse(vrai: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict:
    """Masque booléen (True = manquant) par champ. Le mécanisme MAR dépend de l'âge et de la
    catégorie, toujours renseignés (le masque est tiré sur l'âge exact, qui ne diffère de l'âge
    observé que pour les dates de naissance erronées) ; le MNAR dépend de la valeur elle-même."""
    nr = cfg["non_reponse"]
    n = len(vrai)
    taux = nr["taux"]
    masques = {}
    for champ in nr["champs"]:
        if nr["mecanisme"] == "MCAR":
            w = np.ones(n)
        elif nr["mecanisme"] == "MAR":
            w = np.exp(nr["mar_effet_age"] * (vrai["age"].to_numpy() - 40.0) / 10.0
                       + nr["mar_effet_employe"] * (vrai["categorie"].to_numpy() == "employe"))
        elif nr["mecanisme"] == "MNAR":
            ref = "salaire" if champ in ("salaire", "cotisation") else champ
            x = np.log(vrai[ref].to_numpy() + 1.0) if ref != "anciennete" else vrai[ref].to_numpy()
            w = np.exp(nr["mnar_effet"] * (x - x.mean()) / x.std())
        else:
            raise ValueError(nr["mecanisme"])
        prob = np.clip(taux * w / w.mean(), 0.0, 0.95)
        masques[champ] = rng.random(n) < prob
    return masques


def simuler(cfg: dict, graine: int) -> dict:
    rng = np.random.default_rng(graine)
    vrai = generer_population(cfg, rng)
    masques = masque_non_reponse(vrai, cfg, rng)
    manquant_ligne = np.zeros(len(vrai), dtype=bool)
    for m in masques.values():
        manquant_ligne |= m
    obs_complet, type_erreur, champs = injecter_erreurs(vrai, cfg, rng, manquant_ligne)

    obs = obs_complet.copy()
    for champ, m in masques.items():
        obs.loc[m, champ] = np.nan
    if cfg.get("donnees", {}).get("sans_cotisation", False):
        obs["cotisation"] = np.nan
        obs_complet = obs_complet.copy()
        obs_complet["cotisation"] = np.nan

    # Une ligne est anormale si au moins une valeur erronée reste observée.
    anormal = np.zeros(len(vrai), dtype=bool)
    anormal_complet = np.zeros(len(vrai), dtype=bool)
    for c, flag in champs.items():
        anormal |= flag & obs[c].notna().to_numpy()
        anormal_complet |= flag & obs_complet[c].notna().to_numpy()
    return {
        "vrai": vrai, "observe": obs, "observe_complet": obs_complet,
        "anormal": anormal, "anormal_complet": anormal_complet,
        "type_erreur": type_erreur, "champs_errones": champs, "manquant_ligne": manquant_ligne,
    }
