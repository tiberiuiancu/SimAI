#!/bin/bash
#SBATCH --job-name=simai_profile_test
#SBATCH --output=/gpfs/home6/tiberiui/simai/simai_profile_test_%j.out
#SBATCH --error=/gpfs/home6/tiberiui/simai/simai_profile_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
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

# Print environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo ""
echo "CUDA devices:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo ""

# Run tests
echo "Running integration tests..."
uv run bash tests/test_profile_integration.sh
exit $?
