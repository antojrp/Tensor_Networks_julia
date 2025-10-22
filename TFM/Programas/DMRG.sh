#!/bin/bash

#SBATCH -c 20
#SBATCH -p GPGPU
#SBATCH --mem=50GB
#SBATCH -t 72:00:00 

OMP_NUM_THREADS=15 /home/ajrp/julia-1.11.1/bin/julia -t 1 --heap-size-hint=45G DMRG.jl
