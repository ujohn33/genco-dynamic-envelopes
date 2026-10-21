#!/bin/bash
# Train a GENCO model with gridfm-graphkit on one Hydra A100.
#   sbatch scripts/hydra_train_genco.slurm <config> <data_path> <exp_name>
# Expects $VSC_DATA/gencodoe with .venv (torch 2.11.0+cu126, torch_scatter, gridfm-graphkit 0.9.0.3),
# configs/ and the raw datakit data under <data_path>/<network>/raw.
#SBATCH --job-name=genco-train
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$VSC_DATA/gencodoe"
module load Python/3.12.3-GCCcore-13.3.0
nvidia-smi --query-gpu=name,memory.total --format=csv
CONFIG="$1"; DATA="$2"; EXP="$3"
echo "config=$CONFIG data=$DATA exp=$EXP job=$SLURM_JOB_ID"
export MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_AGENT_HINT=1
.venv/bin/gridfm_graphkit train --config "$CONFIG" --data_path "$DATA" --log_dir mlruns --exp_name "$EXP" \
  --num_workers "$SLURM_CPUS_PER_TASK"
