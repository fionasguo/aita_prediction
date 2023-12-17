#!/bin/bash
#SBATCH --partition=donut-default
#SBATCH --gpus=1
#SBATCH --nodelist donut-gpu01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60GB

source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

# python src/outcome_predictor/outcome_predictor.py --mode concat_embeddings --output_dir /home1/siyiguo/aita_prediction/outcome_predictor_concat_embeddings_output -b 32
# python src/effect_predictor.py --mode concat_embeddings --output_dir effect_prediction_out -b 32
# python src/propensity_predictor.py --mode concat_embeddings --output_dir tmp -b 32
python src/trainer.py --mode concat_embeddings --output_dir tmp -b 32