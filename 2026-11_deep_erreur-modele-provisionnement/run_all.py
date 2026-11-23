"""Exécute toute l'analyse : fil rouge, jeux 1 et 2 (modèle principal, bootstrap, témoins, sensibilité),
contrôle de réalisme sur 20 triangles par jeu, critères, figures.

Usage : python run_all.py [--B 300] [--n-real 20]
Durée indicative : environ 1 h 30 sur 2 cœurs (le bootstrap et le contrôle de réalisme dominent).
Les calculs lourds sont mis en cache dans work/ (supprimer le dossier pour tout recalculer).
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import bootstrap as BS
from src import fil_rouge
from src.data import triangle
from src.primary import analyse
from src.witnesses import mack, glm_bootstrap

OUT, WORK = "results", "work"
SEEDS = {1: 1, 2: 2}          # graine du triangle observé de chaque jeu
BOOT_SEED = 2023
N = 40
WIDENS = (1.0, 1.1, 1.2, 1.3)   # élargissement des portes de la Table 2 (sensibilité)
WORK_WIDEN = 1.1                # configuration de travail : avec les portes de la Table 2, 3 réplications sur 300 passent (jeu 1)


def cached(name, fn):
    path = os.path.join(WORK, name + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    obj = fn()
    pickle.dump(obj, open(path, "wb"))
    return obj


def primary_summary(ds, prim, mu, past):
    rows = []
    for name, pr in prim["priors"].items():
        rows.append(dict(jeu=ds, apriori=name, s_laplace=pr["s"],
                         provision_vraie=mu[~past].sum(),
                         provision_1se_brute=prim["reserves"][prim["i1se"]],
                         moyenne_aposteriori=pr.get("mean", np.nan), imse_cov=pr.get("cov", np.nan),
                         nb_modeles_proba_sup_1e4=pr.get("n_models", np.nan),
                         mode_q=pr.get("mode", -1) + 1, calibrage_atteint=pr.get("hit", np.nan)))
    return rows


def realism_one(args):
    """Un triangle indépendant : modèle principal (BMA a priori 1se) et Mack. Résultat mis en cache par triangle."""
    ds, seed = args
    wd = os.path.join(WORK, f"real_{ds}_{seed}")
    os.makedirs(wd, exist_ok=True)
    cache = os.path.join(wd, "result.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    y, mu, past = triangle(ds, seed)
    try:
        prim = analyse(y, past, N, wd)
        R_mack, cov_mack = mack(y, past)
        pr = prim["priors"]["1se"]
        res = dict(jeu=ds, graine=seed, provision_vraie=mu[~past].sum(), paiements_futurs_realises=y[~past].sum(),
                   provision_1se_brute=prim["reserves"][prim["i1se"]], moyenne_aposteriori=pr["mean"],
                   imse_cov=pr["cov"], provision_mack=R_mack, cov_mack=cov_mack, phi=prim["fam"]["phi"],
                   phi_methode=prim["fam"]["glm_converged"], erreur="")
    except Exception as e:  # noqa: BLE001 : on garde la trace de l'échec plutôt que de perdre le lot
        res = dict(jeu=ds, graine=seed, erreur=repr(e)[:200])
    pickle.dump(res, open(cache, "wb"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--n-real", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    print("1. Fil rouge")
    fil_rouge.run(WORK, OUT)

    prim_rows, boot_rows, sens_rows, wit_rows, crit, homog, lit_rows = [], [], [], [], [], [], []
    for ds in (1, 2):
        print(f"2. Jeu {ds} : modèle principal")
        y, mu, past = triangle(ds, SEEDS[ds])
        prim = cached(f"prim_{ds}", lambda: analyse(y, past, N, WORK))
        prim_rows += primary_summary(ds, prim, mu, past)
        # élagage : modèles éliminés par les portes et leur poids a posteriori avant élagage (C2)
        w_all = prim["priors"]["1se"]["w_all"]
        out_mask = prim["valid"] & ~prim["admissible"]
        pd.DataFrame(dict(q=np.arange(1, prim["Q"] + 1), nb_variables=prim["fam"]["df"],
                          provision=prim["reserves"], admis=prim["admissible"],
                          poids_avant_elagage=w_all, poids_apres=prim["priors"]["1se"]["w"])
                     ).to_csv(f"{OUT}/jeu{ds}_famille.csv", index=False)

        print(f"3. Jeu {ds} : bootstrap B = {a.B} (résidus ODP)")
        homog.append(dict(jeu=ds, **{f"gamma {k}": v for k, v in BS.homogeneity(prim, y, past, N, 2).items()},
                          **{f"odp {k}": v for k, v in BS.homogeneity(prim, y, past, N, 1).items()}))
        boot = cached(f"boot_{ds}_{a.B}_p1", lambda: BS.fit(prim, y, past, N, a.B, BOOT_SEED + ds, WORK, power=1))
        for prior in ("simple", "1se", "minCV", "complexe"):
            if not np.isfinite(prim["priors"][prior]["s"]):
                continue
            for widen in WIDENS:
                m = BS.matrix(prim, boot, past, N, prior, widen)
                row = dict(jeu=ds, apriori=prior, elargissement_portes=widen, B=m["B"],
                           replications_gardees=m["n_kept"], facteur_echelle=m["scale"],
                           provision_moyenne=m["m"], imse_cov=m["w_imse"], parametre_cov=m["w_pa"],
                           parametre_plus_imse_cov=m["w_pa_imse"], processus_cov=m["w_pr"],
                           sous_total_cov=m["w_sub"], provision_vraie=mu[~past].sum(),
                           modeles_admis_median=float(np.median(m["n_adm"])))
                sens_rows.append(row)
                if widen == WORK_WIDEN:
                    boot_rows.append(row)
                    if prior == "1se":
                        np.save(f"{WORK}/matrix_{ds}_res.npy", m["res"])
                        np.save(f"{WORK}/matrix_{ds}_W.npy", m["W"])
        del boot
        if ds == 1 and os.path.exists(os.path.join(WORK, f"boot_1_{a.B}.pkl")):
            lit = pickle.load(open(os.path.join(WORK, f"boot_1_{a.B}.pkl"), "rb"))   # protocole littéral (résidus Gamma)
            for widen in (1.0, 1.1):
                m = BS.matrix(prim, lit, past, N, "1se", widen)
                fut = ~past.ravel()
                r1 = np.nansum(lit["preds"][:, prim["i1se"], :][:, fut], 1)
                lit_rows.append(dict(jeu=1, apriori="1se", residus="Gamma (protocole littéral)",
                                     elargissement_portes=widen, replications_gardees=m["n_kept"], B=m["B"],
                                     provision_moyenne=m["m"], sous_total_cov=m["w_sub"],
                                     provision_1se_bootstrap_p05=np.percentile(r1, 5),
                                     provision_1se_bootstrap_p50=np.percentile(r1, 50),
                                     provision_1se_bootstrap_p95=np.percentile(r1, 95)))
            del lit
        print(f"4. Jeu {ds} : témoins")
        R_m, cov_m = mack(y, past)
        g = glm_bootstrap(y, past, N, ds, a.B, BOOT_SEED + 10 + ds, power=1, widen=WORK_WIDEN)
        wit_rows += [dict(jeu=ds, methode="Mack (chain ladder)", provision=R_m, parametre_plus_processus_cov=cov_m,
                          parametre_cov=np.nan, processus_cov=np.nan, provision_vraie=mu[~past].sum()),
                     dict(jeu=ds, methode="GLM Gamma de structure connue (bootstrap)", provision=g["R0"],
                          parametre_cov=g["w_pa"], processus_cov=g["w_pr"],
                          parametre_plus_processus_cov=np.sqrt(g["w_pa"] ** 2 + g["w_pr"] ** 2),
                          replications_gardees=g["n_kept"], provision_vraie=mu[~past].sum())]

    P, Bt, S, Wt = map(pd.DataFrame, (prim_rows, boot_rows, sens_rows, wit_rows))
    P.to_csv(f"{OUT}/modele_principal.csv", index=False)
    Bt.to_csv(f"{OUT}/bootstrap.csv", index=False)
    S.to_csv(f"{OUT}/sensibilite_portes.csv", index=False)
    Wt.to_csv(f"{OUT}/temoins.csv", index=False)
    pd.DataFrame(homog).to_csv(f"{OUT}/homogeneite_residus.csv", index=False)
    pd.DataFrame(lit_rows).to_csv(f"{OUT}/bootstrap_protocole_litteral.csv", index=False)

    print(f"5. Contrôle de réalisme : {a.n_real} triangles par jeu")
    jobs = [(ds, 100 + i) for ds in (1, 2) for i in range(a.n_real)]

    def run_real():
        with ProcessPoolExecutor(2) as ex:
            return list(ex.map(realism_one, jobs))
    R = pd.DataFrame(run_real())
    R["erreur_relative_bma"] = R.moyenne_aposteriori / R.paiements_futurs_realises - 1
    R["erreur_relative_mack"] = R.provision_mack / R.paiements_futurs_realises - 1
    R["ecart_esperance_vs_realise"] = R.provision_vraie / R.paiements_futurs_realises - 1
    R.to_csv(f"{OUT}/realisme.csv", index=False)

    print("6. Critères")
    from src.criteria import evaluate
    crit = evaluate(P, Bt, S, Wt, OUT)
    crit.to_csv(f"{OUT}/criteres.csv", index=False)
    print(crit.to_string())

    print("7. Figures")
    from src import figures
    figures.all_figures(OUT, WORK)
    json.dump(dict(B=a.B, n_real=a.n_real, seeds=SEEDS, boot_seed=BOOT_SEED, widens=WIDENS, work_widen=WORK_WIDEN,
                   bootstrap_residus="ODP (variance proportionnelle a mu)", npad=5),
              open(f"{OUT}/parametres.json", "w"))


if __name__ == "__main__":
    main()
