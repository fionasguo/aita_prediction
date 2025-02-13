#!/bin/bash
#SBATCH --partition=long-lg
#SBATCH --account=long-lg
#SBATCH --gres=gpu:quadrortx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem 0
#SBATCH --time=72:00:00

# #SBATCH --gres=gpu:rtxa6000:1

source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib


# 3 2024 812 99 706
for SEED in 3 2024 706; \
do \
    python src/train_and_test_propensity.py \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_gender_age_comments/aita_gender_age_comments.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_gender_age_comments/seed_$SEED/propensity_model \
        -b 256 \
        -s $SEED; \
        # -f "[4]"

    python src/train_and_test_outcome.py \
        --model bert \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_gender_age_comments/aita_gender_age_comments.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_gender_age_comments/seed_$SEED/outcome_bert \
        -b 256 \
        -s $SEED; \

    python src/train_and_test_outcome.py \
        --model DANN \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_gender_age_comments/aita_gender_age_comments.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_gender_age_comments/seed_$SEED/outcome_DANN \
        -b 64 \
        -s $SEED; \
        # -f "[4]" 
done

# 3 2024 812 99 706
for SEED in 3 2024 706; \
do \
    python src/train_and_test_effect_genderage.py \
        --model none \
        --outcome_model bert \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_gender_age_comments/aita_gender_age_comments.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_gender_age_comments/seed_$SEED/effect_none_bert \
        --cov_dim 30 \
        --ite True \
        -s $SEED; \
        # -b 64 \

    python src/train_and_test_effect_genderage.py \
        --model none \
        --outcome_model DANN \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_gender_age_comments/aita_gender_age_comments.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_gender_age_comments/seed_$SEED/effect_none_DANN \
        --cov_dim 30 \
        --ite True \
        -s $SEED; \

    python src/train_and_test_effect_genderage.py \
        --model DR \
        --outcome_model bert \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_gender_age_comments/aita_gender_age_comments.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_gender_age_comments/seed_$SEED/effect_DR_bert \
        --cov_dim 30 \
        --ite True \
        -s $SEED; \

    python src/train_and_test_effect_genderage.py \
        --model DR \
        --outcome_model DANN \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/aita_gender_age_comments/aita_gender_age_comments.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/aita_gender_age_comments/seed_$SEED/effect_DR_DANN \
        --cov_dim 30 \
        --ite True \
        -s $SEED; \
done

