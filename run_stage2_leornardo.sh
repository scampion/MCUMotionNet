#!/bin/bash
#SBATCH --job-name=train_stage2_mcunet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=aifac_p01_168  # <-- IMPORTANT: Please replace with your account ID
#SBATCH --output=logs/slurm/stage2_training_%j.out
#SBATCH --error=logs/slurm/stage2_training_%j.err

# --- Environment Setup ---
echo "Setting up the environment..."
module load profile/deeplrn
module load cudnn

if [ ! -f "libdevice.10.bc" ]; then
    cp /leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/cuda-12.2.0-o6rr2unwsp4e4av6ukobro6plj7ceeos/nvvm/libdevice/libdevice.10.bc ./libdevice.10.bc
fi
source .venv/bin/activate

# --- Set CUDA and Data Paths ---
export CUDA_HOME=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/cuda-12.2.0-o6rr2unwsp4e4av6ukobro6plj7ceeos
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc

export ANNOTATION_DATA_DIR="/leonardo/home/userexternal/scampion/work/sport_video_scrapper/camera_movement_reports/"
export VIDEO_DATA_DIR="/leonardo/home/userexternal/scampion/work/sport_video_scrapper/data/videos/"

# --- Training Script Execution ---
echo "Starting the training script..."

# Check if arguments are provided, otherwise use defaults
NUM_VIDEOS=${1:-1000} # Default to 1000 if first argument is not provided
EXPERIMENT_NAME=${2:-"all"} # Default to "all" if second argument is not provided

echo "Running with --num-videos=${NUM_VIDEOS} and --experiment-name=${EXPERIMENT_NAME}"

python train_combined_model_stage2.py \
    --num-videos "$NUM_VIDEOS" \
    --experiment-name "$EXPERIMENT_NAME"

echo "Training script finished."
