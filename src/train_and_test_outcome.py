"""
Train the model to predict the causal effect where text is the intervention.

1. run the outcome predictor with 5-fold splitting and predict the outcome for all training data
2. run the propensity predictor with 5-fold splitting and predict the outcome for all training data
3. compute the effect using doubly robust
4. train the effect_predictor using story + comment as input, and the effect from dobuly robust as the outcome
"""

import argparse
import pandas as pd
import numpy as np
import logging
import gc
import torch

from bert_outcome_predictor import train_bert_outcome_predictor,test_bert_outcome_predictor
from DANN_outcome_predictor import train_DANN_outcome_predictor,test_DANN_outcome_predictor
from utils.data_processing import *
from utils.utils import create_logger
from DANN import read_data


MODEL = 'bert-base-uncased'
DATADIR = 'data/fiona-aita-verdicts.csv'
# DATADIR = 'data/test.csv'


if __name__ == "__main__":
    ## command args
    parser = argparse.ArgumentParser(description='AITA Classifier.')

    parser.add_argument('-m','--mode', type=str, default='concat_embedding', help='choose from concat_text, concat_embeddings, add_embeddings')
    parser.add_argument('-o','--output_dir', type=str, default='./output', help='output dir to be written')
    parser.add_argument('-c','--config_dir', type=str, default=None, help='path to the config file')
    parser.add_argument('-l','--lr', type=float, default=0.00002, help='learning rate')
    parser.add_argument('-e','--n_epoch', type=int, default=40, help='number of epochs to train for')
    parser.add_argument('-b','--batch_size', type=int, default=48, help='mini-batch size')
    parser.add_argument('-s','--seed', type=int, default=3, help='random seed')

    args = parser.parse_args()

    # logger
    create_logger()

    logging.info(args)

    # load and preprocessing text data
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    df = load_text_data(DATADIR)
    # split into train and test
    train_data, test_data = train_test_split(df,tr_frac=0.8,seed=args.seed)
    logging.info(f'sizes of train data: {train_data.shape}, test data: {test_data.shape}')

    # on the training data, run 5-fold outcome & propensity train/inference
    # the results double in size because for each story there are top and random comments
    train_data_outcome = np.empty((0,4))
    train_data_propensity = np.empty((0,2))
    treated = np.empty((0,1))

    n_folds = 5
    train_data = train_data.sample(frac=1,random_state=args.seed) # shuffle
    N_train = len(train_data)
    N_fold = int(N_train/n_folds)
    logging.info(f"number of folds: {n_folds}, size of each fold: {N_fold}")

    for fold in range(n_folds):
        logging.info(f"FOLD# {fold}")
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

        # # outcome prediction - vanilla bert
        # tmp_tr, tmp_val, tmp_te = process_data_bert_outcome_prediction(train_subset,val_subset,test_subset)
        # logging.info(f'outcome data processed: tr: {tmp_tr.shape}, val: {tmp_val.shape}, te: {tmp_te.shape}')
        # tmp_tr_dataset = data_loader(tmp_tr, tokenizer, args.mode)
        # tmp_val_dataset = data_loader(tmp_val, tokenizer, args.mode)
        # tmp_te_dataset = data_loader(tmp_te, tokenizer, args.mode)

        # trainer = train_bert_outcome_predictor(args, tmp_tr_dataset, tmp_val_dataset)
        # fold_outcome = test_bert_outcome_predictor(trainer, args, tmp_te_dataset, save_preds=False)

        # outcome prediction - DANN
        datasets = read_data(
            tokenizer,
            train=train_subset,
            val=val_subset,
            test=test_subset,
            mode=args.mode,
            train_frac=0.8,
            seed=args.seed
            )
        trainer = train_DANN_outcome_predictor(vars(args), datasets)
        fold_outcome = test_DANN_outcome_predictor(trainer,vars(args),datasets,save_preds=False)

        logging.info(f"fold outcome size: {fold_outcome.shape}")

        #train_data_outcome.append(fold_outcome)
        train_data_outcome = np.vstack((train_data_outcome,fold_outcome))
        treated = np.vstack((treated,np.array([1 for _ in range(len(fold_outcome)//2)] + [0 for _ in range(len(fold_outcome)//2)]).reshape(-1,1)))
        np.savetxt(f'data/train_DANN_outcome_{fold}.csv',train_data_outcome,delimiter=',')
        # np.savetxt(f'data/train_DANN_treated_{fold}.csv',treated,delimiter=',')

        logging.info(f"binary treated var size: {len(treated)}")

        del trainer.model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

