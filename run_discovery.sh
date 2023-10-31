#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=11GB
#SBATCH --account=lerman_316
#SBATCH --time=48:00:00

source /home1/siyiguo/anaconda3/bin/activate base

#source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home1/siyiguo/anaconda3/lib

#python src/aita_classifiers.py --mode concat_text --output_dir /home1/siyiguo/aita_prediction/concat_text_output -b 64
##python src/outcome_predictor/outcome_predictor.py --mode concat_embeddings --output_dir tmp -b 32

# python src/outcome_predictor/outcome_predictor.py --mode concat_embeddings --output_dir /home1/siyiguo/aita_prediction/outcome_predictor_concat_embeddings_output -b 32
python src/propensity_predictor/propensity_predictor.py --mode concat_embeddings --output_dir propensity_predictor_concat_embeddings_output -b 64
