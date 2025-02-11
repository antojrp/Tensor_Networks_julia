#!/bin/bash
 
#SBATCH -c 20
#SBATCH -p GPGPU
#SBATCH --mem=100GB
#SBATCH -t 24:00:00 

OMP_NUM_THREADS=18 /home/ajrp/julia-1.11.1/bin/julia -t 1 --heap-size-hint=45G overlap.jl
