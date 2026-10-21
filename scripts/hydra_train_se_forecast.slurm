#!/bin/bash
# Train GENCO SE with day-ahead forecast inputs (scripts/train_genco_se_forecast.py) on one Hydra GPU.
#   sbatch scripts/hydra_train_se_forecast.slurm
# Expects $VSC_DATA/gencodoe with .venv, the repo's src/ scripts/ configs/, data_se/case33bw/raw and
# se_fluvius/states.npz.
#SBATCH --job-name=se-forecast
#SBATCH --partition=ampere_gpu,hopper_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/gencodoe"
module load Python/3.12.3-GCCcore-13.3.0
nvidia-smi --query-gpu=name,memory.total --format=csv
export PYTHONPATH=src
.venv/bin/python scripts/train_genco_se_forecast.py --config configs/genco_se_forecast_case33bw.yaml \
  --states se_fluvius/states.npz --raw data_se/case33bw/raw --out runs/genco-se-forecast-$SLURM_JOB_ID \
  --workers 6
