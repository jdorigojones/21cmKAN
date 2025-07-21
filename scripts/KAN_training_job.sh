#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --job-name=21cmKAN_train_21cmGEM_default2_20
#SBATCH --output=21cmKAN_train_21cmGEM_default2_20.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jodo2960@colorado.edu

ml purge
ml miniforge
mamba activate 21cm-kan-env

export PATH=$PATH:$HOME/.local/bin/
export UCX_TLS=ud,sm,self
export SLURM_EXPORT_ENV=ALL

python 21cmKAN/run_batch_training_21cmGEM.py
python 21cmKAN/evaluate_test_set_21cmGEM.py
