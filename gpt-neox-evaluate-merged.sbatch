#!/bin/bash
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name="neox-merged"
#SBATCH --time=1-12:00
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --output="gpt-neox-merged-"%j.out
#SBATCH --error="gpt-neox-merged-"%j.err
#SBATCH --constraint="type_e"
#SBATCH --mail-user=gaarutyunov@edu.hse.ru
#SBATCH --mail-type=ALL

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
python ./deepy.py evaluate.py -d 20B_checkpoints_finetuned_merged/configs 20B.yml --eval_tasks apps_introductory apps_interview apps_competition