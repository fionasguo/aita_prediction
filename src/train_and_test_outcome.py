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
from ast import literal_eval
import logging
import gc
import torch

from bert_outcome_predictor import train_bert_outcome_predictor,test_bert_outcome_predictor
from DANN_outcome_predictor import train_DANN_outcome_predictor,test_DANN_outcome_predictor
from evaluate import evaluate
from utils.data_processing import *
from utils.utils import create_logger
from DANN import read_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

MODEL = 'bert-base-uncased'


if __name__ == "__main__":
    ## command args
    parser = argparse.ArgumentParser(description='Potential Outcome Predictor.')

    # parser.add_argument('-m','--mode', type=str, default='concat_embedding', help='choose from concat_text, concat_embeddings, add_embeddings')
    parser.add_argument('-i','--input_dir', type=str, required=True, help='input dir of data')
    parser.add_argument('-o','--output_dir', type=str, default='./output', help='output dir to be written')
    parser.add_argument('-c','--config_dir', type=str, default=None, help='path to the config file')
    parser.add_argument('-l','--lr', type=float, default=0.00002, help='learning rate')
    parser.add_argument('-e','--n_epoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('-b','--batch_size', type=int, default=48, help='mini-batch size')
    parser.add_argument('-s','--seed', type=int, default=3, help='random seed')
    parser.add_argument('-f','--fold', type=str, default="[0,1,2,3,4]", help='list of fold number 0 to 4')
    parser.add_argument('--model', type=str, default='bert', help='bert or DANN')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.fold = literal_eval(args.fold)
    data_folder_dir = os.path.dirname(args.input_dir)

    # logger
    create_logger(args.output_dir)

    logging.info(f"Data: {args.input_dir}. Running 5-fold {args.model} outcome predictor training/prediction. seed={args.seed}")
    logging.info(args)

    # load and preprocessing text data
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #, local_files_only=True)
    df = load_text_data(args.input_dir)
    # split into train and test
    train_data, test_data = train_test_split(df,tr_frac=0.8,seed=args.seed)
    logging.info(f'sizes of train data: {train_data.shape}, test data: {test_data.shape}')

    # on the training data, run 5-fold outcome & propensity train/inference
    # the results double in size because for each story there are top and random comments
    # try:
    #     train_data_outcome = np.loadtxt(f'{data_folder_dir}/seed_{args.seed}/train_{args.model}_outcome.csv',delimiter=',')
    #     train_data_id = np.loadtxt(f'{data_folder_dir}/seed_{args.seed}/train_{args.model}_id.csv',delimiter=',')
    #     # train_data_id = train_data_id.reshape(-1,1)
    #     logging.info(f"loading saved train data outcome and id file. prev outcome size={train_data_outcome.shape}, prev train data id size={train_data_id.shape}")
    # except:
    train_data_outcome = np.empty((0,))
    train_data_id = np.empty((0,))
    logging.info("initiating new train data outcome and id")

    n_folds = 5
    train_data = train_data.sample(frac=1,random_state=args.seed) # shuffle
    N_train = len(train_data)
    N_fold = int(N_train/n_folds)
    logging.info(f"number of folds: {n_folds}, size of each fold: {N_fold}")

    for fold in args.fold:
        logging.info(f"####################### FOLD {fold} #######################")
        # if fold < 4: continue
        # get train val test subsets
        if fold == n_folds-1: # last fold
            test_subset = train_data[N_fold*fold:]
        else:
            test_subset = train_data[N_fold*fold:N_fold*(fold+1)]
        train_subset = train_data.drop(test_subset.index)
        logging.info(f"in fold, train size: {train_subset.shape}, test size: {test_subset.shape}")
        val_subset = train_subset.sample(frac=0.2,random_state=args.seed)
        train_subset = train_subset.drop(val_subset.index)

        # outcome prediction - vanilla bert
        if args.model == 'bert':
            logging.info("outcome model: vanilla bert")
            tmp_tr, tmp_val, tmp_te = process_data_bert_outcome_prediction(train_subset,val_subset,test_subset)
            logging.info(f'outcome data processed: tr: {tmp_tr.shape}, val: {tmp_val.shape}, te: {tmp_te.shape}')
            tmp_tr_dataset = data_loader(tmp_tr, tokenizer) #, args.mode)
            tmp_val_dataset = data_loader(tmp_val, tokenizer) #, args.mode)
            tmp_te_dataset = data_loader(tmp_te, tokenizer) #, args.mode)

            trainer = train_bert_outcome_predictor(args, tmp_tr_dataset, tmp_val_dataset)
            fold_outcome = test_bert_outcome_predictor(trainer, args, tmp_te_dataset, save_preds=False)
            
        # # outcome prediction - DANN
        elif args.model == 'DANN':
            logging.info("outcome model: DANN")
            datasets = read_data(
                tokenizer,
                train=train_subset,
                val=val_subset,
                test=test_subset,
                train_frac=0.8,
                seed=args.seed
                )
            trainer = train_DANN_outcome_predictor(vars(args), datasets)
            fold_outcome = test_DANN_outcome_predictor(trainer,vars(args),datasets,save_preds=False)
        else:
            raise ValueError("Please specify model as either bert or DANN")

        fold_outcome = fold_outcome[:,1]
        logging.info(f"fold outcome size: {fold_outcome.shape}")

        #train_data_outcome.append(fold_outcome)
        train_data_outcome = np.hstack((train_data_outcome,fold_outcome))
        train_data_id = np.hstack((train_data_id,test_subset['id'].values))
        
        if not os.path.exists(f'{data_folder_dir}/seed_{args.seed}'):
            os.makedirs(f'{data_folder_dir}/seed_{args.seed}')
        np.savetxt(f'{data_folder_dir}/seed_{args.seed}/train_{args.model}_outcome.csv',train_data_outcome,delimiter=',')
        np.savetxt(f'{data_folder_dir}/seed_{args.seed}/train_{args.model}_id.csv',train_data_id,delimiter=',')

        logging.info(f"fold outcome saved - outcome size={train_data_outcome.shape}, train_data id size={len(train_data_id)}")

        del trainer.model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
