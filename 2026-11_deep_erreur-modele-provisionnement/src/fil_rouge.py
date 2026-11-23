"""Fil rouge : petit triangle annuel 10 x 10, BMA complète puis BMA réduite à 5 modèles, calculable à la main."""
import numpy as np
import pandas as pd
from .data import fil_rouge
from .primary import analyse
from . import bma as B


def run(workdir, outdir):
    y, mu, past = fil_rouge()
    n = 10
    r = analyse(y, past, n, workdir)
    p = past.ravel()
    pd.DataFrame(np.where(past, y, np.nan).round(0), index=[f"S{i}" for i in range(1, n + 1)],
                 columns=[f"D{j}" for j in range(1, n + 1)]).to_csv(f"{outdir}/fil_rouge_triangle.csv")
    pr = r["priors"]["1se"]
    fam = r["fam"]
    adm = r["admissible"]
    i1, im = r["i1se"], r["imin"]
    # 5 modèles autour du 1se, espacés de 4 en 4 sur la grille (deux plus simples, deux plus complexes),
    # parmi les modèles admis : ils couvrent la plage des modèles qui comptent dans la BMA complète,
    # et le calcul des poids reste faisable à la main.
    cand = set(np.where(adm)[0].tolist())
    picks = [q for q in (i1 - 8, i1 - 4, i1, i1 + 4, i1 + 8) if q in cand]
    lp = B.log_prior(fam["a0"], fam["l1"], pr["s"], r["m0"])
    ll = r["loglik"]
    tab = pd.DataFrame(dict(modele=[f"M{q + 1}" for q in picks], q=picks,
                            nb_variables=fam["df"][picks].astype(int),
                            norme_L1=fam["l1"][picks], log_apriori=lp[picks], log_vraisemblance=ll[picks],
                            provision=r["reserves"][picks]))
    tab["log_aposteriori"] = tab.log_apriori + tab.log_vraisemblance
    tab["ecart_au_max"] = tab.log_aposteriori - tab.log_aposteriori.max()
    tab["exp_ecart"] = np.exp(tab.ecart_au_max)
    tab["poids"] = tab.exp_ecart / tab.exp_ecart.sum()
    m = (tab.poids * tab.provision).sum()
    sd = np.sqrt((tab.poids * (tab.provision - m) ** 2).sum())
    tab.to_csv(f"{outdir}/fil_rouge_5_modeles.csv", index=False)
    # vérification indépendante : recalcul par la fonction posterior/bma du code principal
    mask = np.zeros(r["Q"], bool)
    mask[picks] = True
    w = B.posterior(ll + lp, mask)
    m2, v2, c2 = B.bma(w, r["reserves"])
    assert abs(m2 - m) < 1e-6 * m and abs(np.sqrt(v2) - sd) < 1e-6 * m
    summ = dict(provision_vraie=mu[~past].sum(), s_laplace=pr["s"], phi=fam["phi"], Q=r["Q"],
                n_admis=int(adm.sum()), q_1se=i1 + 1, q_minCV=im + 1,
                provision_1se_brute=r["reserves"][i1], bma_complete_moyenne=pr["mean"],
                bma_complete_cov=pr["cov"], bma_complete_nb_modeles=pr["n_models"],
                bma5_moyenne=m, bma5_ecart_type=sd, bma5_cov=sd / m)
    pd.Series(summ).to_csv(f"{outdir}/fil_rouge_resume.csv", header=["valeur"])
    path = pd.DataFrame(dict(q=np.arange(1, r["Q"] + 1), lambda_=fam["lam"], nb_variables=fam["df"],
                             provision=r["reserves"], log_vraisemblance=ll, log_apriori=lp,
                             admis=adm, poids_aposteriori=pr["w"]))
    path.to_csv(f"{outdir}/fil_rouge_famille.csv", index=False)
    return r, tab, summ, y, mu, past
