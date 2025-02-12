#!/bin/bash

#SBATCH -c 2
#SBATCH -p GPGPU
#SBATCH --mem=70GB
#SBATCH -t 02:00:00 

OMP_NUM_THREADS=1 /home/ajrp/julia-1.11.1/bin/julia -t 1 --heap-size-hint=45G DMRG.jl
