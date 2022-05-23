#!/bin/bash
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name="neox-eval-ft"
#SBATCH --time=0-04:00
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --output="gpt-neox-eval-ft-"%j.out
#SBATCH --error="gpt-neox-eval-ft-"%j.err
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
python ./deepy.py evaluate.py -d configs 20B-apps-eval-finetuned.yml --eval_tasks apps_introductory apps_interview apps_competition