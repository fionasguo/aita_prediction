#!/bin/bash
#SBATCH --partition=medium
#SBATCH --account=medium
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --ntasks=1
#SBATCH --mem 0
#SBATCH --time=24:00:00
# #SBATCH --output=test.out

source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

# python src/outcome_predictor/outcome_predictor.py --mode concat_embeddings --output_dir /home1/siyiguo/aita_prediction/outcome_predictor_concat_embeddings_output -b 32
# python src/effect_predictor.py --mode concat_embeddings --output_dir effect_prediction_out -b 32
# python src/outcome_predictor.py --mode concat_embeddings --output_dir tmp -b 8

python src/train_and_test.py --mode concat_text --output_dir tmp -b 8
# python src/baseline_train_and_test.py --mode concat_text --output_dir tmp_baseline -b 8

# 2266 - train_and_test
# 2265 - baseline