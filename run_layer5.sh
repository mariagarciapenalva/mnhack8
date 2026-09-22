#!/usr/bin/env bash

echo "=== pre-flight ==="
for f in build/heat_solver_het scripts/ensemble/run_ensemble.py; do
  [ -e "$f" ] && echo "OK       $f" || echo "MISSING  $f  <-- si falta esto, para aquí y dímelo"
done

echo ""
echo "=== Layer 5: lanzando ensemble pilot en background: $(date) ==="
mkdir -p results
nohup python3 scripts/ensemble/run_ensemble.py --bin build/heat_solver_het \
    --out results/ensemble_N32 --detect --seeds 1 --reps 2 --targets 2 \
    > results/ensemble_N32.log 2>&1 &
echo "Layer 5 PID: $!"
echo "(~70 min, ~430 runs -- versión reducida, no el grid completo de ~2900)"
