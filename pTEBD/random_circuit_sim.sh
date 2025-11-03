#!/bin/bash
 
#SBATCH -c 20
#SBATCH -p GPGPU
#SBATCH --mem=100GB
#SBATCH -t 72:00:00 

OMP_NUM_THREADS=1 /home/ajrp/julia-1.11.1/bin/julia -t 3 --heap-size-hint=45G random_circuit_sim.jl