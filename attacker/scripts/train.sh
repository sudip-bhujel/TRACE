#!/bin/bash
# Usage: source .env && sbatch -A $ACCOUNT_NAME attacker/scripts/train.sh <config_path>
# Example: cd $SCRATCH/projects/grad_inversion && source .env && sbatch -A $ACCOUNT_NAME attacker/scripts/train.sh attacker/config/train_layers_dino.yaml

#SBATCH --time=12:00:00
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --partition=H8V141_SAP112M2000_L
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=768G
#SBATCH -e ./logs/err_%j.log
#SBATCH -o ./logs/out_%j.log
#SBATCH --export=NONE

module load ccs/Miniconda3
source activate inversion

CONFIG="$1"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Derive job name from config filename (e.g., train_layers_dino.yaml -> train_layers_dino)
JOB_NAME=$(basename "$CONFIG" .yaml)
scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME"
echo "Job name set to: $JOB_NAME"

# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Configure WandB directories
export WANDB_DIR="${TMPDIR:-/tmp}"
export WANDB_CACHE_DIR="${TMPDIR:-/tmp}/wandb_cache"
export WANDB_CONFIG_DIR="${TMPDIR:-/tmp}/wandb_config"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Configure environment variables for PyTorch and NCCL
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Job id and job name
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $JOB_NAME"

# Debug GPU allocation
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi failed"

# Get number of usable GPUs (ask PyTorch directly — nvidia-smi is unreliable on this cluster)
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "==== Detected $NGPUS GPUs ===="

if [ "$NGPUS" -gt 1 ]; then
    echo "==== Starting DDP training with $NGPUS GPUs ===="
    # Use torch.distributed.run (torchrun) for DDP
    python -m torch.distributed.run --standalone --nproc_per_node=$NGPUS -m attacker.training.train $CONFIG
else
    echo "==== Starting single-GPU/CPU training ===="
    # Standard python execution for single device
    python -m attacker.training.train $CONFIG
fi

echo "---- Container execution completed ----"
