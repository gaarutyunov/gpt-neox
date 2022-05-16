#!/bin/bash
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name="neox-evl"
#SBATCH --time=0-02:00
#SBATCH --gpus=2
#SBATCH --output="gpt-neox-eval-"%j.out
#SBATCH --error="gpt-neox-eval-"%j.err
#SBATCH --constraint="type_a|type_b|type_c|type_e"
#SBATCH --mail-user=gaarutyunov@edu.hse.ru
#SBATCH --mail-type=ALL


module restore default

#Executable
source activate gpt-neox
/opt/hse/slurm/scripts/get_gpu_indexes.sh

python ./megatron/fused_kernels/setup.py install
python ./deepy.py evaluate.py -d configs 20B.yml 20B-apps.yml 20B-eval.yml --eval_tasks apps
