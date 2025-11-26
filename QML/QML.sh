#!/bin/bash
 
#SBATCH -c 1
#SBATCH -p GPGPU
#SBATCH --mem=10GB
#SBATCH -t 72:00:00 

OMP_NUM_THREADS=1 /home/ajrp/julia-1.11.1/bin/julia -t 1 QML.jl