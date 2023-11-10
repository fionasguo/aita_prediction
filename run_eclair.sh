#!/bin/bash
#SBATCH --partition=short
#SBATCH --account=short
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --ntasks=1
#SBATCH --mem 0
#SBATCH --time=6:00:00
# #SBATCH --output=test.out

source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

# python src/outcome_predictor/outcome_predictor.py --mode concat_embeddings --output_dir /home1/siyiguo/aita_prediction/outcome_predictor_concat_embeddings_output -b 32
# python src/effect_predictor.py --mode concat_embeddings --output_dir effect_prediction_out -b 32
# python src/outcome_predictor.py --mode concat_embeddings --output_dir tmp -b 8
python src/trainer.py --mode concat_text --output_dir tmp -b 16

#export CUDA_VISIBLE_DEVICES=0,1
#python -m torch.distributed.launch --nproc_per_node 2 src/trainer.py --mode concat_embeddings --output_dir tmp -b 32

# 544 trainer 16
# 543 trainer 8
# 542 outcome 8*4
# 539 outcome 8*1
# 538 propensity 8*1
