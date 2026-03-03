#!/bin/bash

#SBATCH --job-name="M01_H"
#SBATCH --output=avp_%J.out
#SBATCH --error=avp_%J.err
#SBATCH --account=upc27
#SBATCH -N 5                      
#SBATCH --ntasks-per-node=56      # asi usas los 2 nodos enteros ( c/nodo tiene 112 cores)
##SBATCH --constraint=highmem
#SBATCH --qos=gp_resa
##SBATCH --qos=gp_debug
##SBATCH --time=02:00:00

ulimit -s unlimited

export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=1
##export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge

module load singularity openmpi/4.1.5-gcc

mpirun --bind-to none \
    singularity exec \
    -B /apps \
    /apps/GPP/SINGULARITY/images/underworld_v2.16.1b.sif \
    python M01_H.py

#srun singularity exec -B /apps/GPP/SINGULARITY/images/underworld2-2.15.1b.sif python H1.py

