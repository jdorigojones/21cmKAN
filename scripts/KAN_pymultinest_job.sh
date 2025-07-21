#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=21cmKAN_pymultinest_1
#SBATCH --output=21cmKAN_pymultinest_1.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jodo2960@colorado.edu

ml purge
ml miniforge
mamba activate 21cm-kan-env

export PATH=$PATH:$HOME/.local/bin/
export UCX_TLS=ud,sm,self
export SLURM_EXPORT_ENV=ALL
export LD_LIBRARY_PATH=/projects/jodo2960/KAN/MultiNest/lib:$LD_LIBRARY_PATH
export PERSES=/projects/jodo2960/KAN/perses
export PYLINEX=/projects/jodo2960/KAN/pylinex
export DISTPY=/projects/jodo2960/KAN/distpy
export ARES=/projects/jodo2960/KAN/ares

cd 21cmKAN/

python pymultinest_21cmKAN_21cmGEM.py
