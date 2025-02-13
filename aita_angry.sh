#!/bin/bash
#SBATCH --partition=medium-lg
#SBATCH --account=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem 0
#SBATCH --time=24:00:00

source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

for SEED in 3 2024 812 99 706; \
do \
    python src/train_and_test_effect.py \
        --model none \
        --outcome_model bert \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_angry/aita_angry.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_angry/seed_$SEED/effect_none_bert \
        --cov_dim 30 \
        --ite True \
        -s $SEED; \
        # -b 64 \

    python src/train_and_test_effect.py \
        --model none \
        --outcome_model DANN \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_angry/aita_angry.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_angry/seed_$SEED/effect_none_DANN \
        --cov_dim 30 \
        --ite True \
        -s $SEED; \

    python src/train_and_test_effect.py \
        --model DR \
        --outcome_model bert \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_angry/aita_angry.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_angry/seed_$SEED/effect_DR_bert \
        --cov_dim 30 \
        --ite True \
        -s $SEED; \

    python src/train_and_test_effect.py \
        --model DR \
        --outcome_model DANN \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_angry/aita_angry.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_angry/seed_$SEED/effect_DR_DANN \
        --cov_dim 30 \
        --ite True \
        -s $SEED; \
done



# python src/train_and_test_propensity.py \
#     --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_angry/aita_angry.csv \
#     --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_angry/seed_$SEED \
#     -b 460 \
#     -s $SEED \
#     # -f "[4]"

# python src/train_and_test_outcome.py \
#     --model bert \
#     --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_angry/aita_angry.csv \
#     --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_angry/seed_$SEED \
#     -b 460 \
#     -s $SEED

# python src/train_and_test_outcome.py \
#     --model DANN \
#     --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_angry/aita_angry.csv \
#     --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_angry/seed_$SEED/outcome_DANN \
#     -b 128 \
#     -s $SEED \
#     # -f "[4]" 
