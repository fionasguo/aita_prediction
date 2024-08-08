"""
Train the model to predict the causal effect where text is the intervention.

1. run the outcome predictor with 5-fold splitting and predict the outcome for all training data
2. run the propensity predictor with 5-fold splitting and predict the outcome for all training data
3. compute the effect using doubly robust
4. train the effect_predictor using story as input, and the effect from dobuly robust as the outcome
"""

import os
import argparse
import pandas as pd
import numpy as np
import logging

from effect_predictor import train_effect_predictor,test_effect_predictor
from utils.data_processing import *
from utils.compute_effect import doubly_robust
from utils.utils import create_logger


# MODEL = 'bert-base-uncased'
# tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


if __name__ == "__main__":
    ## command args
    parser = argparse.ArgumentParser(description='AITA Classifier.')

    parser.add_argument('-i','--input_dir', type=str, required=True, help='input dir of data')
    parser.add_argument('-o','--output_dir', type=str, default='./output', help='output dir to be written')
    parser.add_argument('-l','--lr', type=float, default=0.00002, help='learning rate')
    parser.add_argument('-e','--num_epoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('-b','--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('-s','--seed', type=int, default=3, help='random seed')
    parser.add_argument('--model', type=str, default='DR', help='DR for doubly robust or none for simple subtraction between predicted outcomes for treated and control')
    parser.add_argument('--outcome_model', type=str, default='bert', help='bert or DANN')
    parser.add_argument('--cov_dim', type=int, required=True, help='number of dimensions of covariates X')
    parser.add_argument('--ite', type=bool, required=False, default=False, help="wheter to compute Individual Treatment Effects")
    
    args = parser.parse_args()
    data_folder_dir = os.path.dirname(args.input_dir)

    # logger
    create_logger(args.output_dir)

    logging.info(f"Data: {args.input_dir}. Training Effect prector with {args.model} model, potential outcome predicted by {args.outcome_model}, seed={args.seed}")
    logging.info(args)

    # load and preprocessing text data
    df = load_text_data(args.input_dir)
    df = df.set_index('id')

    # # split into train and test
    # train_data, test_data = train_test_split(df,tr_frac=0.8,seed=args.seed)
    # logging.info(f'sizes of train data: {train_data.shape}, test data: {test_data.shape}')

    # n_folds = 5
    # train_data = train_data.sample(frac=1,random_state=args.seed) # shuffle
    # N_train = len(train_data)
    # N_fold = int(N_train/n_folds)
    # logging.info(f"number of folds: {n_folds}, size of each fold: {N_fold}")

    # read outcome, treated, propensity
    train_data_outcome = np.loadtxt(f'{data_folder_dir}/seed_{args.seed}/train_{args.outcome_model}_outcome.csv',delimiter=',')
    train_data_id = np.loadtxt(f'{data_folder_dir}/seed_{args.seed}/train_{args.outcome_model}_id.csv',delimiter=',')
    logging.info(f"loaded train_data outcome shape={train_data_outcome.shape}, id shape={train_data_id.shape}")

    train_data = df.loc[train_data_id,]
    test_data = df.drop(train_data.index)

    if args.model == 'DR':
        train_data_propensity = np.loadtxt(f'{data_folder_dir}/seed_{args.seed}/train_propensity.csv',delimiter=',')
        logging.info(f"loaded train_data propensity shape={train_data_propensity.shape}")
        # effect using doubly robust 
        # Y = train_data['y'].apply(lambda x: [1 if x==i else 0 for i in range(2)]).tolist()
        # Y = np.array(Y)
        # logging.info(f"Y size: {Y.shape}")
        # np.savetxt(f'data/seed_{args.seed}/train_Y.csv',Y,delimiter=',')

        cate = CATE_doubly_robust(train_data['y'], train_data_outcome, train_data_propensity, train_data['T']).tolist() # shape: N_train * 4
        if args.ite:
            # TODO: make sure id matches
            ite = ITE_doubley_robust()
        
    elif args.model == 'none':
        treated_train_data_outcome = train_data_outcome[treated==1,1]
        control_train_data_outcome = train_data_outcome[treated==0,1]
        # train_data['effect'] = (treated_train_data_outcome - control_train_data_outcome).tolist()
        effects = (treated_train_data_outcome - control_train_data_outcome).tolist()
        logging.info(f"effects shape: {effects.shape}")

    train_data_ = pd.DataFrame(train_data.loc[treated==0,'X'])
    train_data_['effect'] = effects
    logging.info(f"preparing training data for effect prediction: shape={train_data_.shape}")

    # train the effect predictor
    val_data = train_data_.sample(frac=0.2,random_state=args.seed)
    train_data_ = train_data_.drop(val_data.index)

    train_dataset, val_dataset, test_dataset = process_data_effect_prediction(train_data_,val_data,test_data,args.cov_dim)
    # train_dataset = data_loader(train_dataset, tokenizer, mode='story_only')
    # val_dataset = data_loader(val_dataset, tokenizer, mode='story_only')
    # test_dataset = data_loader(test_dataset, tokenizer, mode='story_only')

    logging.info('training effect predictor')

    trainer = train_effect_predictor(args,train_dataset,val_dataset)
    test_data_effects = test_effect_predictor(trainer,args,test_dataset)
    test_data['effect_pred'] = test_data_effects.tolist()
    # test_data.to_csv(args.output_dir+'/test_data_prediction.csv',index=False)

    logging.info('Finished training effect predictor')
    
