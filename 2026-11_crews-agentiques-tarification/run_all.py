"""Point d'entrée.

  python run_all.py --prepare   télécharge freMTPL2freq, construit les témoins et les dossiers sabotés
  python run_all.py --analyse   recalcule tous les résultats et figures (après les exécutions de crews)

Les exécutions de crews (payantes, API Claude) se lancent avec run_mrm_batch.sh et run_crew_batch.sh.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "runs"
URL = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/freMTPL2freq.rda"


def prepare():
    raw = ROOT / "data" / "freMTPL2freq.parquet"
    if not raw.exists():
        import pyreadr
        rda = ROOT / "data" / "freMTPL2freq.rda"
        urllib.request.urlretrieve(URL, rda)
        df = list(pyreadr.read_r(str(rda)).values())[0]
        df["IDpol"] = df["IDpol"].astype(str).astype(float).astype("int64")
        for c in ["VehBrand", "VehGas", "Area", "Region"]:
            df[c] = df[c].astype(str)
        df.to_parquet(raw, index=False)
        rda.unlink()
    from src.baseline import build_baselines
    from src.sabotage import build_packages
    base = build_baselines(RUNS / "baselines")
    (ROOT / "results" / "baselines.json").write_text(json.dumps(base, indent=1))
    truth = build_packages(RUNS / "packages")
    (ROOT / "results" / "packages_truth.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False))
    print(json.dumps(base, indent=1))


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    if "--analyse" in sys.argv:
        import analyse
        analyse.main()
