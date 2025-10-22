#!/bin/bash

#SBATCH -c 2
#SBATCH -p GPGPU
#SBATCH --mem=70GB
#SBATCH -t 02:00:00 

#module load julia
/home/ajrp/julia-1.11.1/bin/julia --heap-size-hint=68G -t 2 evolution_separate.jl
