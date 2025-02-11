#!/bin/bash
#SBATCH --partition=long
#SBATCH --account=long
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem 0
#SBATCH --time=72:00:00
# #SBATCH --output=test.out

python src/train_and_test_outcome.py \
    --mode concat_embedding \
    --output_dir /nas/home/siyiguo/aita_prediction/test_DANN \
    -b 48

python src/baseline_train_and_test.py \
    --mode concat_embedding \
    --output_dir /nas/home/siyiguo/aita_prediction/test_baseline_DANN \
    -b 32
