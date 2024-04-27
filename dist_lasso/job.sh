#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --output=logs/slurm.%j.%N.out   # Output file name
#SBATCH --error=logs/slurm.%j.%N.err     # STDERR output file (optional)

#SBATCH --partition=main
#SBATCH --mem=8000                  # Real memory (RAM) required (MB)
#SBATCH --constraint=broadwell      # Require a Broadwell node
#SBATCH --ntasks=14                 # 14 total tasks

#SBATCH --time=01:00:00             # Total run time limit (HH:MM:SS)

module purge
source activate /home/bbc33/.conda/envs/hello

mpirun python /scratch/bbc33/ece509-final/dist_lasso/hello_matdotvec.py