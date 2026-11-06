"""Lance une exécution de crew de modélisation.

Usage : python run_crew.py <run_id> <variante: autonome|naif> <routage: route|tout_opus>
Les dossiers de travail vont dans ../runs/<run_id> (hors dépôt), les journaux dans results/.
"""
import json
import sys
import time
from pathlib import Path

from src.crews.modeling import ModelingCrew
from src.llm import LLM, default_ledger, require_key

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "runs"

if __name__ == "__main__":
    require_key()
    run_id, variant, mode = sys.argv[1:4]
    ledger = default_ledger()
    llm = LLM(ledger, mode=mode, run_id=run_id)
    t0 = time.time()
    before = ledger.spent
    log = ModelingCrew(llm, RUNS / run_id, variant=variant).run()
    log.update(run_id=run_id, mode=mode, minutes=round((time.time() - t0) / 60, 1),
               cost_usd=round(ledger.spent - before, 4))
    out = ROOT / "results" / "crew_runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps({k: log.get(k) for k in ("run_id", "status", "final_verdict", "minutes", "cost_usd")}))
