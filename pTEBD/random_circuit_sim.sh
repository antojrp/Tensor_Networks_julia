#!/bin/bash
 
#SBATCH -c 30
#SBATCH -p GPGPU
#SBATCH --mem=100GB
#SBATCH -t 72:00:00 

OMP_NUM_THREADS=4 /home/ajrp/julia-1.11.1/bin/julia -t 6 random_circuit_sim.jl