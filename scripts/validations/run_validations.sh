#!/usr/bin/env bash
# =============================================================================
# run_validations.sh -- build + run the baseline validation battery.
#
#   scripts/validations/run_validations.sh <label> <solver.cu> [ref.cu] [ARCH] [MAXRANKS]
#
#   label      name of the results folder: tests/baseline_validation/<label>/
#   solver.cu  candidate (e.g. src/heat_solver_het.cu, or the original file)
#   ref.cu     MNHack7 reference for V1 (default src/heat_solver.cu)
#   ARCH       sm_86 (RTX 3060), sm_75 (T4), sm_90 (H100)     default sm_86
#   MAXRANKS   ranks available for V3/V4 (0 disables MPI tests) default 4
#
# Each V* writes report.json (pass/fail + numbers), run.log (every command and
# its output) and figures. summary.txt collects the verdicts.
# A binary without --verify/--kernel/--scatter (the original file) gets
# UNSUPPORTED on V0, V2-V6 and only V1 runs -- that IS the result for it.
# =============================================================================
set -euo pipefail
LABEL=${1:?label}; SRC=${2:?solver.cu}; REF=${3:-src/heat_solver.cu}; ARCH=${4:-sm_86}; MAXR=${5:-4}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
OUT=tests/baseline_validation/$LABEL; mkdir -p "$OUT" build
GEN=scripts/gen_field.py; V=scripts/validations
BIN=build/heat_solver_$LABEL; REFBIN=build/heat_solver_ref

echo "== build =="
nvcc -O3 -arch=$ARCH -ccbin=mpicxx "$SRC" -o "$BIN" 2>&1 | tee "$OUT/build.log"
[ -x "$REFBIN" ] || nvcc -O3 -arch=$ARCH -ccbin=mpicxx "$REF" -o "$REFBIN" 2>&1 | tee -a "$OUT/build.log"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.txt" 2>/dev/null || true
nvcc --version | tail -1 >> "$OUT/gpu.txt"; git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true

echo "== V0 (all four kernel/scatter combinations) =="
for k in fast gauss; do for s in colored atomic; do
  python3 $V/v0_convergence.py --bin "$BIN" --gen $GEN --out "$OUT" --kernel $k --scatter $s || true
done; done
echo "== V1 =="; python3 $V/v1_regression.py --bin "$BIN" --ref "$REFBIN" --gen $GEN --out "$OUT" || true
echo "== V2 =="; python3 $V/v2_noise_floor.py --bin "$BIN" --gen $GEN --out "$OUT" || true
echo "== V3 =="; python3 $V/v3_injection.py  --bin "$BIN" --gen $GEN --out "$OUT" --mpi $MAXR || true
echo "== V4/V5 =="; python3 $V/v4_v5_equivalence.py --bin "$BIN" --gen $GEN --out "$OUT" --mpi $MAXR || true
echo "== V6 =="; python3 $V/v6_detection.py --bin "$BIN" --gen $GEN --out "$OUT" --mpi $MAXR || true
echo "== V7 (quantum substrate, no GPU) =="; python3 scripts/quantum/v7_quantum.py --out "$OUT/v7" || true
echo "== V8 (hybrid coupling, NumPy) =="; python3 scripts/quantum/v8_hybrid_mock.py --out "$OUT/v8" || true

echo "== summary =="; : > "$OUT/summary.txt"
for r in $(find "$OUT" -name report.json | sort); do
  python3 - "$r" >> "$OUT/summary.txt" <<'EOF'
import json,sys; d=json.load(open(sys.argv[1])); s=d["details"].get("status","")
print(f"{sys.argv[1]:70s} {'PASS' if d['pass'] else 'FAIL'} {s}")
EOF
done
cat "$OUT/summary.txt"
