#!/bin/bash
#SBATCH --job-name=simai_bench
#SBATCH --output=/gpfs/home6/tiberiui/simai/simai_bench_%j.out
#SBATCH --error=/gpfs/home6/tiberiui/simai/simai_bench_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_h100

echo "=== SLURM Job Starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Initial working directory: $(pwd)"
echo "Starting time: $(date)"
echo ""

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

echo "Modules loaded successfully"
echo ""

# Change to project directory
echo "Changing to project directory..."
cd /gpfs/home6/tiberiui/simai || exit 1
echo "Current directory: $(pwd)"
echo ""

echo "Activating virtual environment..."
source /gpfs/home6/tiberiui/simai/.venv/bin/activate || exit 1
echo "Virtual environment activated"
echo ""

# Derive MASTER_ADDR from the first node in the allocation.
# scontrol expands compact nodelist notation (e.g. node[01-02]) to one hostname per line.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}

# Print environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo ""
echo "CUDA devices:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo ""

# Launch one simai task per node. Each task reads the SLURM env vars as defaults
# and spawns torchrun locally to drive $SLURM_GPUS_PER_NODE GPU workers.
echo "Launching benchmark..."
srun simai bench training \
    --framework Megatron \
    --tensor-parallel 4 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-heads 16 \
    --global-batch-size 8 \
    --micro-batch-size 1 \
    --epochs 1 \
    --output /gpfs/home6/tiberiui/simai/results/bench/
exit $?
