#!/bin/bash
# Lance la crew MRM sur les 7 dossiers sabotés, 2 répétitions (la 1re répétition de P2 = pilote).
cd "$(dirname "$0")"
PK=/home/claude/work/runs/packages
for rep in a b; do
  for p in P0_sain P1_fuite P2_exposition P3_contamination P4_calibration P5_documentation P6_metriques; do
    [ "$rep" = a ] && [ "$p" = P2_exposition ] && continue
    [ -f results/mrm_runs/mrm_${p}_${rep}.json ] && continue
    python3 run_mrm.py mrm_${p}_${rep} $PK/$p $PK/${p}_data || echo "ECHEC $p $rep"
  done
done
