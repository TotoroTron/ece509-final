#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --output=logs/slurm.%j.%N.out   # Output file name
#SBATCH --error=logs/slurm.%j.%N.err     # STDERR output file (optional)

#SBATCH --partition=main
#SBATCH --mem=16000                 # Real memory (RAM) required (MB)
#SBATCH --ntasks=10                 # 10 total tasks

#SBATCH --time=01:00:00             # Total run time limit (HH:MM:SS)

module purge

source /projects/community/anaconda/2020.07/gc563/etc/profile.d/conda.sh
# source /home/bbc33/.conda/envs/hello/bin/activate
conda activate /home/bbc33/.conda/envs/hello

mpirun python /scratch/bbc33/ece509-final/dist_lasso/lasso_mpi.py
# mpirun -np $SLURM_NTASKS /scratch/bbc33/ece509-final/dist_lasso/hello_matdotvec.py
