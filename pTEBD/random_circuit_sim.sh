#!/bin/bash
 
#SBATCH -c 20
#SBATCH -p GPGPU
#SBATCH --mem=100GB
#SBATCH -t 72:00:00 

OMP_NUM_THREADS=2 /home/ajrp/julia-1.11.1/bin/julia -t 9 random_circuit_sim.jl