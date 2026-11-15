"""Lance les 300 simulations Monte Carlo, écrit results/*.csv, results/*.json et results/figures/*.png.

Usage : python run_all.py [--sims 300] [--workers 8]
Le volet LLM (appels payants) est séparé : python run_llm.py
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, wilcoxon

from src.evaluate import run_sim
from src import figures

ROOT = Path(__file__).parent
RES = ROOT / "results"


def clopper_pearson(k: int, n: int, a: float = .05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


DEFAULT = {  # réglages de référence : configuration du papier (k = 8) et précision publiée (0,92)
    "BM25": 8, "RAG vectoriel": 8, "Vue structurée": .92, "Double vue": .92,
    "Double vue (balayage 0.95)": .92, "Oracle": 1.0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    RES.mkdir(exist_ok=True); (RES / "figures").mkdir(exist_ok=True)

    with Pool(args.workers) as pool:
        rows = [r for sim in pool.map(run_sim, range(1, args.sims + 1)) for r in sim]
    df = pd.DataFrame(rows)
    df.to_csv(RES / "simulations_detail.csv.gz", index=False, compression="gzip")

    # Tableau principal (réglages de référence)
    q_prec_k = {"BM25": 10, "RAG vectoriel": 10}
    main_rows = []
    for (q, m), g in df.groupby(["q", "method"]):
        par = q_prec_k.get(m, DEFAULT.get(m)) if q == "Q_prec" else DEFAULT.get(m)
        if par is None:
            continue
        g = g[np.isclose(g.param, par)]
        if g.empty:
            continue
        k, n = int(g.success.sum()), len(g)
        lo, hi = clopper_pearson(k, n)
        main_rows.append(dict(question=q, dispositif=m, parametre=par, n_questions=n,
                              succes=k / n, ic95_bas=lo, ic95_haut=hi,
                              erreur_ou_ratio_moyen=g.err.mean(), docs_lus_median=g.docs_read.median()))
    for code, lab in [(0.0, " [calibré E1/E2, FP 2,6 %]"), (-1.0, " [calibré E1/E2, FP 0,46 %]")]:
      for (q, m), g in df[np.isclose(df.param, code)].groupby(["q", "method"]):
        k, n = int(g.success.sum()), len(g)
        lo, hi = clopper_pearson(k, n)
        main_rows.append(dict(question=q, dispositif=m + lab, parametre=code, n_questions=n,
                              succes=k / n, ic95_bas=lo, ic95_haut=hi, erreur_ou_ratio_moyen=g.err.mean(),
                              docs_lus_median=g.docs_read.median()))
    main = pd.DataFrame(main_rows)
    main.to_csv(RES / "resultats_principaux.csv", index=False)

    # Sensibilités : toutes les combinaisons (dispositif, paramètre)
    sens = (df.groupby(["q", "method", "param"]).agg(n=("success", "size"), succes=("success", "mean"),
                                                     erreur_moy=("err", "mean"),
                                                     erreur_abs_moy=("err", lambda x: x.abs().mean()),
                                                     docs_lus_median=("docs_read", "median"))
            .reset_index())
    sens.to_csv(RES / "sensibilites.csv", index=False)

    # Tests appariés : taux de succès par simulation, Wilcoxon signé
    def per_sim(q, m, p):
        g = df[(df.q == q) & (df.method == m) & np.isclose(df.param, p)]
        return g.groupby("sim").success.mean()
    tests = []
    comps = [
        ("Q_part", ("Vue structurée", .92), ("RAG vectoriel", 8)),
        ("Q_part", ("Vue structurée", .92), ("BM25", 8)),
        ("Q_part", ("Vue structurée", .92), ("BM25", 50)),
        ("Q_top3", ("Vue structurée", .92), ("RAG vectoriel", 8)),
        ("Q_top3", ("Vue structurée", .92), ("BM25", 50)),
        ("Q_rare", ("Double vue (balayage 0.95)", .92), ("Vue structurée", .92)),
        ("Q_rare", ("Double vue (balayage 0.95)", .92), ("RAG vectoriel", 8)),
        ("Q_rare", ("Double vue (balayage 0.95)", .92), ("BM25", 50)),
        ("Q_rare", ("Double vue (balayage 0.80)", .92), ("BM25", 50)),
        ("Q_prec", ("Double vue", .92), ("Vue structurée", .92)),
        ("Q_prec", ("Double vue", .92), ("RAG vectoriel", 10)),
        ("Q_part", ("Vue structurée", .98), ("Vue structurée", .92)),
        # scénario calibré sur le volet LLM (param 0 = CALIB), ajouté après E1/E2
        ("Q_part", ("Vue structurée", 0.0), ("Vue structurée", .92)),
        ("Q_top3", ("Vue structurée", 0.0), ("Vue structurée", .92)),
        ("Q_top3", ("Vue structurée", 0.0), ("RAG vectoriel", 8)),
        ("Q_rare", ("Vue structurée", 0.0), ("BM25", 50)),
        ("Q_prec", ("Double vue", 0.0), ("Double vue", .92)),
        ("Q_rare", ("Vue structurée", -1.0), ("BM25", 50)),
        ("Q_rare", ("Vue structurée", -1.0), ("Vue structurée", 0.0)),
    ]
    for q, a, b in comps:
        sa, sb = per_sim(q, *a), per_sim(q, *b)
        d = (sa - sb).dropna()
        stat = wilcoxon(d) if (d != 0).any() else None
        se = d.std(ddof=1) / np.sqrt(len(d))
        tests.append(dict(question=q, A=f"{a[0]} ({a[1]})", B=f"{b[0]} ({b[1]})", n_sims=len(d),
                          succes_A=sa.mean(), succes_B=sb.mean(), ecart_moyen=d.mean(),
                          ic95_ecart_bas=d.mean() - 1.96 * se, ic95_ecart_haut=d.mean() + 1.96 * se,
                          p_value=None if stat is None else float(stat.pvalue)))
    pd.DataFrame(tests).to_csv(RES / "tests_apparies.csv", index=False)

    # Ratio continu sur Q_prec (moyenne et écart-type)
    prec = (df[df.q == "Q_prec"].groupby(["method", "param"]).err.agg(["mean", "std", "size"]).reset_index())
    prec.to_csv(RES / "q_prec_ratio.csv", index=False)

    # Schéma induit
    sch = df.drop_duplicates("sim").schema_size
    meta = dict(n_sims=args.sims, n_dossiers=5000, questions_par_type_et_sim=5,
                taille_schema_facteurs=dict(min=int(sch.min()), median=float(sch.median()), max=int(sch.max()),
                                             part_sims_avec_au_moins_un_rare=float((sch > 9).mean())),
                audit_oracle={q: float(df[(df.q == q) & (df.method == "Oracle")].success.mean())
                              for q in ["Q_part", "Q_top3", "Q_rare", "Q_prec"]})
    (RES / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    figures.make_all(df, main, sens, RES / "figures")
    print(main.to_string())
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
