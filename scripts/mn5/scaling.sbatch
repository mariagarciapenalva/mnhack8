#!/bin/bash
# Layer 9 -- classical scaling strand on MN5 ACC (H100 x4 per node).
#   strong: fixed N, ranks 4,8,16,32     sbatch --export=MODE=strong,N=256 -N 8 scaling.sbatch
#   weak  : cells/rank fixed, N grows    sbatch --export=MODE=weak,CELLS=64 -N 8 scaling.sbatch
# Verify module names on the system first:  module avail 2>&1 | grep -i -E "cuda|openmpi|nvhpc"
#SBATCH -J mnhack8-scaling
#SBATCH -A ACCOUNT_TODO
#SBATCH -q acc_debug
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 20
#SBATCH -t 01:00:00
#SBATCH -o slurm-%j.out
set -e
module purge; module load cuda openmpi     # TODO: exact names
MODE=${MODE:-strong}; N=${N:-256}; CELLS=${CELLS:-64}
OUT=results/mn5_scaling_${MODE}_$SLURM_JOB_ID; mkdir -p $OUT fields
python3 scripts/gen_field.py --n $N --model lognormal --sigma 1.0 --L 0.125 --upscale 2 --seed 7 --out fields/ln$N
for R in 4 8 16 32; do
  [ $R -gt $SLURM_NTASKS ] && break
  if [ "$MODE" = weak ]; then N=$((CELLS*R)); python3 scripts/gen_field.py --n $N --model lognormal --sigma 1.0 --L 0.125 --seed 7 --out fields/ln$N; fi
  d=$OUT/R$R; mkdir -p $d
  srun -n $R --gpus-per-task=1 build/heat_solver_het $N 0.01 1e-4 fields/ln$N $d --tag sc --snap 0 --kernel fast --scatter atomic
done
python3 scripts/mn5/scaling_plot.py --root $OUT --mode $MODE
