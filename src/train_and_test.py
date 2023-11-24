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

from outcome_predictor import train_outcome_predictor,test_outcome_predictor
from propensity_predictor import train_propensity_predictor,test_propensity_predictor
from effect_predictor import train_effect_predictor,test_effect_predictor
from utils.data_processing import *
from utils.doubly_robust import doubly_robust
from utils.utils import create_logger


# MODEL = 'allenai/longformer-base-4096'
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

    args = parser.parse_args()

    # logger
    create_logger()

    # load and preprocessing text data
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    df = load_text_data(DATADIR)
    # split into train and test
    train_data, test_data = train_test_split(df,tr_frac=0.8,seed=args.seed)
    logging.info(f'sizes of train data: {train_data.shape}, test data: {test_data.shape}')

    # on the training data, run 5-fold outcome & propensity train/inference
    # the results double in size because for each story there are top and random comments
    train_data_outcome = []
    train_data_propensity = []
    treated = []

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

        # outcome prediction
        tmp_tr, tmp_val, tmp_te = process_data_outcome_prediction(train_subset,val_subset,test_subset)
        logging.info(f'outcome data processed: tr: {tmp_tr.shape}, val: {tmp_val.shape}, te: {tmp_te.shape}')
        tmp_tr_dataset = data_loader(tmp_tr, tokenizer, args.mode)
        tmp_val_dataset = data_loader(tmp_val, tokenizer, args.mode)
        tmp_te_dataset = data_loader(tmp_te, tokenizer, args.mode)

        trainer = train_outcome_predictor(args, tmp_tr_dataset, tmp_val_dataset)
        fold_outcome = test_outcome_predictor(trainer, args, tmp_te_dataset, save_preds=False)

        logging.info(f"fold outcome size: {fold_outcome.shape}")

        #train_data_outcome.append(fold_outcome)
        train_data_outcome = np.vstack((train_data_outcome,fold_outcome))
        treated = np.vstack((treated,np.array([1 for _ in range(len(fold_outcome)//2)] + [0 for _ in range(len(fold_outcome)//2)]).reshape(-1,1)))
        np.savetxt('data/train_outcome.csv',train_data_outcome,delimiter=',')
        np.savetxt('data/train_treated.csv',treated,delimiter=',')

        logging.info(f"binary treated var size: {len(treated)}")

        del trainer.model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        # propensity
        tmp_tr, tmp_val, tmp_te = process_data_propensity_prediction(train_subset,val_subset,test_subset)
        logging.info(f'outcome data processed: tr: {tmp_tr.shape}, val: {tmp_val.shape}, te: {tmp_te.shape}')
        tmp_tr_dataset = data_loader(tmp_tr, tokenizer, mode='concat_text')
        tmp_val_dataset = data_loader(tmp_val, tokenizer, mode='concat_text')
        tmp_te_dataset = data_loader(tmp_te, tokenizer, mode='concat_text')

        trainer = train_propensity_predictor(args, tmp_tr_dataset, tmp_val_dataset)
        fold_propensity = test_propensity_predictor(trainer,args,tmp_te_dataset,save_preds=False)

        logging.info(f"fold propensity size: {fold_propensity.shape}")

        train_data_propensity = np.vstack((train_data_propensity,fold_propensity))
        np.savetxt('data/train_propensity_1.csv',train_data_propensity,delimiter=',')

        del trainer.model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    train_data_outcome = np.vstack(train_data_outcome) # N datapoints * 4 - softmax 4 different classes of verdicts
    train_data_propensity = np.vstack(train_data_propensity) # N datapoints * 2 - softmax 2 classes top/rand for propensity
    treated = np.array(treated)
    logging.info(f"all train data - outcome size: {train_data_outcome.shape}, propensity: {train_data_propensity.shape}, binary treated var: {treated.shape}")

    # effect using doubly robust 
    Y = train_data['top_verdict'].apply(lambda x: [1 if x==i else 0 for i in range(4)]).tolist()
    Y = np.array(Y)
    logging.info(f"Y size: {Y.shape}")

    np.savetxt('data/test_outcome.csv', train_data_outcome, delimiter=',')
    np.savetxt('data/test_propensity.csv',train_data_propensity,delimiter=',')
    np.savetxt('data/test_Y.csv',Y,delimiter=',')
    np.savetxt('data/test_treated.csv',treated,delimiter=',')

    train_data['effect'] = doubly_robust(Y, train_data_outcome, train_data_propensity, treated).tolist() # shape: N_train * 4
    logging.info(f"added effects train data: {train_data.shape}")

    # train the effect predictor
    val_data = train_data.sample(frac=0.2,random_state=args.seed)
    train_data = train_data.drop(val_data.index)

    train_dataset, val_dataset, test_dataset = process_data_effect_prediction(train_data,val_data,test_data)
    train_dataset = data_loader(train_dataset, tokenizer, mode='story_only')
    val_dataset = data_loader(val_dataset, tokenizer, mode='story_only')
    test_dataset = data_loader(test_dataset, tokenizer, mode='story_only')

    logging.info('training effect predictor')

    trainer = train_effect_predictor(args,train_dataset,val_dataset)
    test_data_effects = test_effect_predictor(trainer,args,test_dataset)
    test_data['effect_pred'] = test_data_effects.tolist()
    test_data.to_csv(args.output_dir+'/test_data_prediction.csv',index=False)

    logging.info('Finished training effect predictor')

    # effect predictor - only story
    