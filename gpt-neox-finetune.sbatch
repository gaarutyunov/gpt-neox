#!/bin/bash
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name="neox-ft"
#SBATCH --time=7-02:00
#SBATCH --gpus-per-node=4
#SBATCH --output="gpt-neox-ft-"%j.out
#SBATCH --error="gpt-neox-ft-"%j.err
#SBATCH --constraint="type_e"
#SBATCH --nodes=2


source ~/.bashrc
module restore default

#Executable
source activate gpt-neox
/opt/hse/slurm/scripts/get_gpu_indexes.sh

#env
type python
nvcc --version
echo "PATH=${PATH}" > ~/.deepspeed_env

python ./megatron/fused_kernels/setup.py install
python ./deepy.py train.py configs/20B.yml