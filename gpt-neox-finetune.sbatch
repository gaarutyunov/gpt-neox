#!/bin/bash
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name="neox-ft"
#SBATCH --time=0-02:00
#SBATCH --gpus=8
#SBATCH --output="gpt-neox-ft-"%j.out
#SBATCH --error="gpt-neox-ft-"%j.err
#SBATCH --constraint="type_a|type_b|type_c|type_e"
#SBATCH --mail-user=gaarutyunov@edu.hse.ru
#SBATCH --mail-type=ALL


module restore default

#Executable
source activate gpt-neox
/opt/hse/slurm/scripts/get_gpu_indexes.sh

python ./megatron/fused_kernels/setup.py install
python ./deepy.py train.py configs/20B.yml