#!/bin/bash
#SBATCH --partition=medium-lg
#SBATCH --account=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
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
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/amazon_reviews_vader/music.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/amazon_reviews_vader/seed_$SEED/effect_model_none_bert \
        -s $SEED \
        --cov_dim 2; \
        # -b 64 \

    python src/train_and_test_effect.py \
        --model none \
        --outcome_model DANN \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/amazon_reviews_vader/music.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/amazon_reviews_vader/seed_$SEED/effect_model_none_DANN \
        -s $SEED \
        --cov_dim 2; \

    python src/train_and_test_effect.py \
        --model DR \
        --outcome_model bert \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/amazon_reviews_vader/music.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/amazon_reviews_vader/seed_$SEED/effect_model_DR_bert \
        -s $SEED \
        --cov_dim 2; \

    python src/train_and_test_effect.py \
        --model DR \
        --outcome_model DANN \
        --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/amazon_reviews_vader/music.csv \
        --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/amazon_reviews_vader/seed_$SEED/effect_model_DR_DANN \
        -s $SEED \
        --cov_dim 2; \
done



# python src/train_and_test_propensity.py \
#     --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/amazon_reviews_vader/music.csv \
#     --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/amazon_reviews_vader/seed_$SEED \
#     -b 460 \
#     -s $SEED

# python src/train_and_test_outcome.py \
#     --model bert \
#     --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/amazon_reviews_vader/music.csv \
#     --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/amazon_reviews_vader/seed_$SEED \
#     -b 460 \
#     -s $SEED

# python src/train_and_test_outcome.py \
#     --model DANN \
#     --input_dir /nas/eclairnas01/users/siyiguo/aita_prediction/data/amazon_reviews_vader/music.csv \
#     --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/amazon_reviews_vader/seed_$SEED/outcome_DANN \
#     -b 32 \
#     -s $SEED










# python src/baseline_train_and_test.py \
#     --mode concat_embedding \
#     --output_dir /nas/eclairnas01/users/siyiguo/aita_prediction/test_baseline_DANN_$SEED_gamma10 \
#     -b 6 \
#     -s $SEED

# 7166 b=6 baseline
# 7183 fold 0
# 7190 fold 1


# 7243 baseline DANN seed=3
# 7245 baseline DANN seed=3
# 7247 baseline DANN seed=3
# 7248 baseline DANN seed=$SEED

# 7257 DR seed=3 calibrate propensity
# 7260 baseline DANN seed=3 finishing up evaluation
# 7265 baseline bert seed=3