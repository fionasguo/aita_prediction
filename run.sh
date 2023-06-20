#!/bin/bash
#SBATCH --partition=donut-default
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60GB

source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

python src/aita_classifiers.py --mode concat_text --output_dir /nas/home/siyiguo/aita_prediction/concat_text_output
#python -m torch.distributed.launch --nproc-per-node 4 src/aita_classifiers.py --mode concat_text --output_dir /nas/home/siyiguo/aita_prediction/concat_text_output
#torchrun --standalone --nnodes=1 --nproc-per-node=4 src/aita_classifiers.py --mode concat_text --output_dir /nas/home/siyiguo/aita_prediction/concat_text_output