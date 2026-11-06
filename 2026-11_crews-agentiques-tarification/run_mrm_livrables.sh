#!/bin/bash
# Revue MRM autonome des modèles livrés par les crews de modélisation (bout en bout, sans humain).
cd "$(dirname "$0")"
L=/home/claude/work/runs/livrables
for r in auto_r3 naif_r1 auto_opus1 naif_r2; do
  [ -f results/mrm_runs/mrm_livrable_$r.json ] && continue
  python3 run_mrm.py mrm_livrable_$r $L/$r $L/${r}_data || echo "ECHEC $r"
done
