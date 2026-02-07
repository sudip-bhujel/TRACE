#!/bin/bash
# Usage: sbatch --account=$ACCOUNT_NAME attacker/scripts/train.sh
# source .env && sbatch -A $ACCOUNT_NAME attacker/scripts/train.sh

#SBATCH --time=3-00:00:00
#SBATCH --job-name=train_temporal_lm_108_1024
#SBATCH --ntasks=1
#SBATCH --partition=H8V141_SAP112M2000_L
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -e ./logs/err_%j.log
#SBATCH -o ./logs/out_%j.log
#SBATCH --export=NONE

unset LD_LIBRARY_PATH

module load ccs/Miniconda3
module load ccs/singularity

IMG=/share/singularity/images/ccs/rocky/rocky8.sinf
# IMG=/share/singularity/images/ccs/conda/lcc-jupyter-rocky8.sinf

# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Export WANDB_API_KEY to Singularity using SINGULARITYENV_ prefix
if [ -n "${WANDB_API_KEY:-}" ]; then
  export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"
fi

echo "---- Running inside container ----"
singularity exec --nv "$IMG" bash -lc '
  set -euo pipefail
  echo "Container OS: $(grep PRETTY_NAME /etc/os-release)"
  echo "whoami: $(whoami)"
  echo "pwd: $(pwd)"
  echo "python: $(which python || true)"
  echo "uv: $(which uv || true)"
  echo "TMPDIR=${TMPDIR:-unset}"
  echo "ulimit -a:"
  echo "ulimit -a:"
  ulimit -a

  # Print job info (from SBATCH directives)
  echo "Job ID: ${SLURM_JOB_ID:-unknown}"
  echo "Job Name: ${SLURM_JOB_NAME:-unknown}"
  echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
  echo "CPUs per Task: ${SLURM_CPUS_PER_TASK:-unknown}"
  echo "GPUs: ${SLURM_GPUS:-unknown}"
  
  # Configure WandB directories
  export WANDB_DIR="${TMPDIR:-/tmp}"
  export WANDB_CACHE_DIR="${TMPDIR:-/tmp}/wandb_cache"
  export WANDB_CONFIG_DIR="${TMPDIR:-/tmp}/wandb_config"
  mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
  
  nvidia-smi || true
  
  # Get number of GPUs
  NGPUS=$(nvidia-smi -L | wc -l)
  echo "==== starting DDP training with $NGPUS GPUs ===="
  
  # Use torch.distributed.run (torchrun) for DDP
  uv sync
  uv run python -m torch.distributed.run --standalone --nproc_per_node=$NGPUS -m attacker.train ./attacker/config/train.yaml
  echo "---- Container execution completed ----"
'
