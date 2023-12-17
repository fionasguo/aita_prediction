#!/bin/bash
#SBATCH --partition=long
#SBATCH --account=long
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem 0
#SBATCH --time=72:00:00
# #SBATCH --output=test.out

source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

export CUDA_VISIBLE_DEVICES=0,1,2,3

# python src/outcome_predictor/outcome_predictor.py --mode concat_embeddings --output_dir /home1/siyiguo/aita_prediction/outcome_predictor_concat_embeddings_output -b 32
# python src/effect_predictor.py --mode concat_embeddings --output_dir effect_prediction_out -b 32
# python src/outcome_predictor.py --mode concat_embeddings --output_dir tmp -b 8

# python src/train_and_test_outcome.py \
#     --mode concat_embedding \
#     --output_dir /nas/home/siyiguo/aita_prediction/test_DANN \
#     -b 48

python src/baseline_train_and_test.py \
    --mode concat_embedding \
    --output_dir /nas/home/siyiguo/aita_prediction/test_baseline_DANN \
    -b 32
# python src/baseline_train_and_test.py --mode concat_embedding --output_dir tmp_baseline -b 12

# 2644 5-fold propensity b=12 concat embedding
# 2643 5-fold outcome b=12 concat embedding
# 2670 baseline b=12
# 2705 propensity fold 3,4
# 2789 effect prediction