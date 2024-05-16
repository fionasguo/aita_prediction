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
from ast import literal_eval
import gc
import torch

from propensity_predictor import train_propensity_predictor,test_propensity_predictor
from utils.data_processing import *
from utils.utils import create_logger


MODEL = 'bert-base-uncased'
DATADIR = 'data/fiona-aita-verdicts.csv'
# DATADIR = 'data/test.csv'


if __name__ == "__main__":
    ## command args
    parser = argparse.ArgumentParser(description='AITA Classifier.')

    parser.add_argument('-m','--mode', type=str, default='concat_text', help='choose from concat_text, concat_embeddings, add_embeddings')
    parser.add_argument('-o','--output_dir', type=str, default='./output', help='output dir to be written')
    parser.add_argument('-l','--lr', type=float, default=0.00002, help='learning rate')
    parser.add_argument('-e','--num_epoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('-b','--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('-s','--seed', type=int, default=3, help='random seed')
    parser.add_argument('-f','--fold', type=str, default="[0,1,2,3,4]", help='list of fold number 0 to 4')
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.fold = literal_eval(args.fold)

    # logger
    create_logger()

    logging.info(f"Running 5-fold propensity predictor training/prediction. seed={args.seed}")
    logging.info(args)

    # load and preprocessing text data
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #, local_files_only=True)
    df = load_text_data(DATADIR)
    # split into train and test
    train_data, test_data = train_test_split(df,tr_frac=0.8,seed=args.seed)
    logging.info(f'sizes of train data: {train_data.shape}, test data: {test_data.shape}')

    # on the training data, run 5-fold outcome & propensity train/inference
    # the results double in size because for each story there are top and random comments
    try:
        train_data_propensity = np.loadtxt(f'data/seed_{args.seed}/train_propensity.csv',delimiter=',')
        logging.info(f"loading saved train data propensity file. prev propensity size={train_data_propensity.shape}")
    except:
        train_data_propensity = np.empty((0,2))
        logging.info("initiating new train data propensity")

    n_folds = 5
    train_data = train_data.sample(frac=1,random_state=args.seed) # shuffle
    N_train = len(train_data)
    N_fold = int(N_train/n_folds)
    logging.info(f"number of folds: {n_folds}, size of each fold: {N_fold}")

    for fold in args.fold:
        logging.info(f"####################### FOLD {fold} #######################")
        # if fold < 3: continue
        # get train val test subsets
        if fold == n_folds-1: # last fold
            test_subset = train_data[N_fold*fold:]
        else:
            test_subset = train_data[N_fold*fold:N_fold*(fold+1)]
        train_subset = train_data.drop(test_subset.index)
        logging.info(f"in fold, train size: {train_subset.shape}, test size: {test_subset.shape}")
        val_subset = train_subset.sample(frac=0.2,random_state=args.seed)
        train_subset = train_subset.drop(val_subset.index)

        # propensity
        tmp_tr, tmp_val, tmp_te = process_data_propensity_prediction(train_subset,val_subset,test_subset)
        logging.info(f'propensity data processed: tr: {tmp_tr.shape}, val: {tmp_val.shape}, te: {tmp_te.shape}')
        tmp_tr_dataset = data_loader(tmp_tr, tokenizer, mode='concat_text')
        tmp_val_dataset = data_loader(tmp_val, tokenizer, mode='concat_text')
        tmp_te_dataset = data_loader(tmp_te, tokenizer, mode='concat_text')

        trainer,temperature = train_propensity_predictor(args, tmp_tr_dataset, tmp_val_dataset)
        fold_propensity = test_propensity_predictor(trainer,args,tmp_te_dataset,temperature,save_preds=False)

        logging.info(f"fold propensity size: {fold_propensity.shape}")

        train_data_propensity = np.vstack((train_data_propensity,fold_propensity))
        np.savetxt(f'data/seed_{args.seed}/train_propensity.csv',train_data_propensity,delimiter=',')

        logging.info(f"fold propensity saved - size={train_data_propensity.shape}")

        del trainer.model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    