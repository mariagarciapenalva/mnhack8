#!/bin/bash
# Layer 5 at scale: job array, one shard per task.   sbatch --array=0-31 ensemble_array.sbatch
#SBATCH -J mnhack8-ens
#SBATCH -A ACCOUNT_TODO
#SBATCH -q acc_bsccs
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH -t 04:00:00
#SBATCH -o slurm-%A_%a.out
set -e
module purge; module load cuda openmpi     # TODO
NS=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-0}
python3 scripts/ensemble/run_ensemble.py --bin build/heat_solver_het --out results/ensemble_N128 \
    --N 128 --dt 2e-4 --t 0.02 --seeds 1,2,3,4,5 --reps 4 --targets 4 --mpi 2 --detect --shard $I/$NS
# after all tasks:  python3 scripts/ensemble/aggregate.py --csv results/ensemble_N128/ensemble.csv --out results/ensemble_N128/agg
