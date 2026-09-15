#!/bin/bash
# Layer 8 on MN5: MPMD launch, P solver ranks + 1 quantum rank, H2 crosses a real NIC when -N 2.
#   sbatch --export=P=4,M=3,BLOCK="9 9 9" -N 2 hybrid.sbatch
#SBATCH -J mnhack8-hybrid
#SBATCH -A ACCOUNT_TODO
#SBATCH -q acc_debug
#SBATCH --ntasks=5
#SBATCH --gres=gpu:4
#SBATCH -c 20
#SBATCH -t 00:30:00
#SBATCH -o slurm-%j.out
set -e
module purge; module load cuda openmpi python     # TODO; python env must have qiskit, qiskit-aer(-gpu), mpi4py
P=${P:-4}; M=${M:-3}; BLOCK=${BLOCK:-"9 9 9"}; N=${N:-64}
OUT=results/mn5_hybrid_$SLURM_JOB_ID; mkdir -p $OUT fields
python3 scripts/gen_field.py --n $N --model homogeneous --out fields/homog$N
cat > $OUT/mpmd.conf <<EOC
0-$((P-1))  build/heat_solver_het $N 0.01 1e-4 fields/homog$N $OUT --tag hy --snap 1 --ic gaussian --no-source --kernel fast --scatter colored --qrank --qblock $BLOCK $M --detect
$P          python3 scripts/quantum/qrank.py --backend varqite --reps 2 --log $OUT/qrank_log.csv
EOC
srun --multi-prog $OUT/mpmd.conf
