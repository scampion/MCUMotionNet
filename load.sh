#!/bin/bash
module load profile/deeplrn
module load cudnn
cp /leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/cuda-12.2.0-o6rr2unwsp4e4av6ukobro6plj7ceeos/nvvm/libdevice/libdevice.10.bc ./libdevice.10.bc
source .venv/bin/activate
# Set CUDA paths
export CUDA_HOME=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/cuda-12.2.0-o6rr2unwsp4e4av6ukobro6plj7ceeos
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Also set these for TensorFlow to find the libraries
export CUDA_ROOT=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc


export ANNOTATION_DATA_DIR="/leonardo/home/userexternal/scampion/work/sport_video_scrapper/camera_movement_reports/"
export VIDEO_DATA_DIR="/leonardo/home/userexternal/scampion/work/sport_video_scrapper/data/videos/"

