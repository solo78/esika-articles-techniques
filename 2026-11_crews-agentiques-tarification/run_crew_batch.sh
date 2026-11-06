#!/bin/bash
# Exécutions des crews de modélisation, l'une après l'autre (mémoire limitée à 7 Go).
# Ordre entrelacé pour garder un plan équilibré si le budget est atteint avant la fin.
cd "$(dirname "$0")"
while pgrep -f run_mrm_batch.sh > /dev/null; do sleep 30; done
for spec in "auto_r1 autonome route" "naif_r1 naif route" "auto_r2 autonome route" "naif_r2 naif route" \
            "auto_opus1 autonome tout_opus" "auto_r3 autonome route" "naif_r3 naif route"; do
  set -- $spec
  [ -f results/crew_runs/$1.json ] && continue
  spent=$(python3 -c "import json;print(sum(json.loads(l)['cost_usd'] for l in open('results/llm_calls.jsonl') if l.strip()))")
  if python3 -c "import sys;sys.exit(0 if $spent > 16.5 else 1)"; then echo "BUDGET : arrêt avant $1 ($spent $)"; break; fi
  python3 run_crew.py $1 $2 $3 || echo "ECHEC $1"
done
