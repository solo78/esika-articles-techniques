#!/bin/bash
# Seconde revue des deux livrables refusés, pour mesurer la stabilité de la gravité attribuée.
cd "$(dirname "$0")"
L=/home/claude/work/runs/livrables
for r in auto_r3 naif_r2; do
  [ -f results/mrm_runs/mrm_livrable_${r}_b.json ] && continue
  python3 run_mrm.py mrm_livrable_${r}_b $L/$r $L/${r}_data || echo "ECHEC $r"
done
