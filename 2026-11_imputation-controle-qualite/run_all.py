"""Lance toutes les simulations puis l'analyse.

Usage :
    python run_all.py               # simulations + analyse + figures
    python run_all.py --analyse     # analyse et figures seulement (à partir de results/simulations.csv)
    python run_all.py --rapide      # 20 simulations par scénario, pour tester
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.analyse import analyser
from src.experience import fusion, une_simulation

RACINE = Path(__file__).parent
RESULTATS = RACINE / "results"


def lancer(cfg: dict, n_sims: int, n_jobs: int) -> pd.DataFrame:
    lignes = []
    for nom, surcharge in cfg["scenarios"].items():
        c = fusion(cfg, surcharge or {})
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(une_simulation)(c, nom, i) for i in range(n_sims))
        for r in res:
            lignes.extend(r)
        print(f"{nom:28s} {n_sims} simulations en {time.time() - t0:6.0f} s", flush=True)
    return pd.DataFrame(lignes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--rapide", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(RACINE / "config.yaml", encoding="utf-8"))
    RESULTATS.mkdir(exist_ok=True)
    (RESULTATS / "figures").mkdir(exist_ok=True)
    fichier = RESULTATS / "simulations.csv"
    if not a.analyse:
        n = 20 if a.rapide else cfg["simulation"]["n_simulations"]
        df = lancer(cfg, n, a.jobs)
        df.to_csv(fichier, index=False)
    df = pd.read_csv(fichier)
    analyser(df, cfg, RESULTATS)


if __name__ == "__main__":
    main()
