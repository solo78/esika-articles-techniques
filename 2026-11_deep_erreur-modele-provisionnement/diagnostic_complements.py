"""Compléments demandés par la relecture critique (23/11) : écrit results/complements.json et results/realisme_glm.csv.

1. Fil rouge : provision vraie si l'inflation de 3 % continuait après l'année 10.
2. Témoin GLM Gamma de structure connue (sans bootstrap) sur les 20 triangles du contrôle de réalisme.
3. Intervalles de confiance à 95 % de l'écart moyen (n = 20), test t, et comptes hors d'une bande commune.
4. Décomptes : modèles éliminés mieux ajustés que le 1se ; poids du 1se dans le vote du jeu 2 ; cellules plafonnées.
"""
import json
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.data import grid, linear_spline, triangle
from src.witnesses import known_structure

out = {}
# 1. fil rouge
k, j, t = grid(10)
past = t <= 10
mu_cont = np.exp(np.log(1000) + 0.05 * linear_spline(k, 3, 10) + 2.0 * np.log(j) - 0.65 * j + 0.03 * (t - 1))
mu_stop = np.exp(np.log(1000) + 0.05 * linear_spline(k, 3, 10) + 2.0 * np.log(j) - 0.65 * j + 0.03 * linear_spline(t, 1, 10))
out["fil_rouge_provision_vraie_inflation_arretee"] = float(mu_stop[~past].sum())
out["fil_rouge_provision_vraie_si_inflation_continue"] = float(mu_cont[~past].sum())

# 2. GLM de structure connue sur les triangles du contrôle de réalisme
rows = []
for ds in (1, 2):
    X = known_structure(40, ds)
    for seed in range(100, 120):
        y, mu, pst = triangle(ds, seed)
        p = pst.ravel()
        g = sm.GLM(y.ravel()[p], X[p], family=sm.families.Gamma(sm.families.links.Log())).fit()
        R = g.predict(X)[~p].sum()
        rows.append(dict(jeu=ds, graine=seed, provision_glm=R, paiements_futurs_realises=y[~pst].sum()))
G = pd.DataFrame(rows)
G["erreur_relative_glm"] = G.provision_glm / G.paiements_futurs_realises - 1
G.to_csv("results/realisme_glm.csv", index=False)

# 3. intervalles, tests, bandes communes
Rr = pd.read_csv("results/realisme.csv")
Bt = pd.read_csv("results/bootstrap.csv")
for ds in (1, 2):
    d = Rr[Rr.jeu == ds]
    sub = Bt[(Bt.jeu == ds) & (Bt.apriori == "1se")].sous_total_cov.iloc[0]
    for col, name in (("erreur_relative_bma", "vote"), ("erreur_relative_mack", "mack")):
        x = d[col].values
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        tq = stats.t.ppf(0.975, len(x) - 1)
        out[f"jeu{ds}_{name}_moyenne"] = float(m)
        out[f"jeu{ds}_{name}_ic95"] = [float(m - tq * se), float(m + tq * se)]
        out[f"jeu{ds}_{name}_t"] = float(m / se)
        out[f"jeu{ds}_{name}_p"] = float(2 * stats.t.sf(abs(m / se), len(x) - 1))
        out[f"jeu{ds}_{name}_hors_bande_commune"] = int((np.abs(x) > 1.96 * sub).sum())
    out[f"jeu{ds}_mack_cov_annonce_min_med_max"] = [float(d.cov_mack.min()), float(d.cov_mack.median()), float(d.cov_mack.max())]
    x = G[G.jeu == ds].erreur_relative_glm.values
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    out[f"jeu{ds}_glm_moyenne"] = float(m)
    out[f"jeu{ds}_glm_ecart_type"] = float(x.std(ddof=1))
    out[f"jeu{ds}_glm_ic95"] = [float(m - 2.093 * se), float(m + 2.093 * se)]
    out[f"jeu{ds}_glm_surestimations"] = int((x > 0).sum())

# 4. décomptes
for ds in (1, 2):
    prim = pickle.load(open(f"work/prim_{ds}.pkl", "rb"))
    ll, adm, i1 = prim["loglik"], prim["admissible"], prim["i1se"]
    elim = prim["valid"] & ~adm
    out[f"jeu{ds}_elimines"] = int(elim.sum())
    out[f"jeu{ds}_elimines_mieux_ajustes_que_1se"] = int((elim & (ll >= ll[i1])).sum())
    pr = prim["priors"]["1se"]
    out[f"jeu{ds}_poids_1se_dans_vote"] = float(pr["w"][i1])
    out[f"jeu{ds}_mode_vote_q"] = int(pr["mode"]) + 1
    out[f"jeu{ds}_q_1se"] = int(i1) + 1
    boot = pickle.load(open(f"work/boot_{ds}_300_p1.pkl", "rb"))
    out[f"jeu{ds}_part_cellules_plancher"] = float(boot["n_floor"] / boot["Yb"].size)
    del boot
json.dump(out, open("results/complements.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
