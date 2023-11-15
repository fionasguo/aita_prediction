"""
Train the baseline model to predict the causal effect where text is the intervention.

1. run the outcome predictor on training data with top comment + story
2. predict on test data with both top comment + story and rand comment + story
3. compute the effect by subtraction
"""

import argparse
import pandas as pd
import numpy as np
import logging
import gc
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from outcome_predictor import train_outcome_predictor,test_outcome_predictor
from utils.data_processing import *
from utils.utils import create_logger


# MODEL = 'allenai/longformer-base-4096'
MODEL = 'bert-base-uncased'
DATADIR = 'data/fiona-aita-verdicts.csv'
# DATADIR = 'data/test.csv'


def evaluate(labels, preds):
    mse = mean_squared_error(labels, preds)
    rmse = mean_squared_error(labels, preds, squared=False)
    mae = mean_absolute_error(labels, preds)

    return {"mse": mse, "rmse": rmse, "mae": mae}


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
    val_data = train_data.sample(frac=0.2,random_state=args.seed)
    train_data = train_data.drop(val_data.index)

    # outcome prediction
    train_dataset,val_dataset,test_dataset = process_data_outcome_prediction(train_data,val_data,test_data)
    logging.info(f'outcome data processed: tr: {train_data.shape}, val: {val_data.shape}, te: {test_data.shape}')
    train_dataset = data_loader(train_dataset, tokenizer, args.mode)
    val_dataset = data_loader(val_dataset, tokenizer, args.mode)
    test_dataset = data_loader(test_dataset, tokenizer, args.mode)

    trainer = train_outcome_predictor(args, train_dataset, val_dataset)
    outcome = test_outcome_predictor(trainer, args, test_dataset, save_preds=False)
    top_com_outcome = outcome[:len(outcome)//2,:]
    rand_com_outcome = outcome[len(outcome)//2,:]
    logging.info(f"outcome size: top comment: {top_com_outcome.shape}, rand comment: {rand_com_outcome.shape}")

    test_data_effects = top_com_outcome - rand_com_outcome
    test_data['effect_pred'] = test_data_effects.tolist()
    test_data.to_csv(args.output_dir+'/test_data_prediction.csv',index=False)

    # evaluate
    test_data['top_verdict_onehot'] = test_data['top_verdict'].apply(lambda x: [1 if x==i else 0 for i in range(4)])
    test_data['rand_verdict_onehot'] = test_data['rand_verdict'].apply(lambda x: [1 if x==i else 0 for i in range(4)])
    gt = test_data.apply(lambda x: [t-r for t,r in zip(x['top_verdict_onehot'],x['rand_verdict_onehot'])],axis=1)
    logging.info('Evaluation on test data:')
    logging.info(evaluate(np.array(gt.tolist()),test_data_effects))

    logging.info('Finished effect prediction')