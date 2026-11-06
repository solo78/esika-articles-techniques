"""Lance la crew MRM sur un dossier de modèle (saboté ou issu d'une crew).

Usage : python run_mrm.py <run_id> <dossier_package> <dossier_data>
"""
import json
import sys
import time
from pathlib import Path

from src.crews.mrm import MRMCrew
from src.llm import LLM, default_ledger, require_key

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "runs"

if __name__ == "__main__":
    require_key()
    run_id, package, data = sys.argv[1:4]
    ledger = default_ledger()
    llm = LLM(ledger, mode="route", run_id=run_id)
    t0, before = time.time(), ledger.spent
    log = MRMCrew(llm, RUNS / run_id, Path(package), Path(data)).run()
    log.update(run_id=run_id, package=Path(package).name, minutes=round((time.time() - t0) / 60, 1),
               cost_usd=round(ledger.spent - before, 4))
    out = ROOT / "results" / "mrm_runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    v = log.get("verdict") or {}
    print(json.dumps({"run_id": run_id, "decision": v.get("decision"),
                      "categories": [(c["categorie"], c["gravite"]) for c in v.get("constats", [])],
                      "minutes": log["minutes"], "cost_usd": log["cost_usd"]}, ensure_ascii=False))
