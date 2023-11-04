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

from outcome_predictor import train_outcome_predictor,test_outcome_predictor
from propensity_predictor import train_propensity_predictor,test_propensity_predictor
from effect_predictor import train_effect_predictor,test_effect_predictor
from utils.data_processing import *
from utils.doubly_robust import doubly_robust


# MODEL = 'allenai/longformer-base-4096'
MODEL = 'bert-base-uncased'
# DATADIR = '/home1/siyiguo/aita_prediction/data/fiona-aita-verdicts.csv'
DATADIR = 'data/test.csv'


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

    # load and preprocessing text data
    df = load_text_data(DATADIR,seed=args.seed)
    # split into train and test
    train_data, test_data = train_test_split(df,tr_frac=0.8,seed=args.seed)
    print(train_data.shape,test_data.shape)

    # on the training data, run 5-fold outcome & propensity train/inference
    # the results double in size because for each story there are top and random comments
    train_data_outcome = []
    train_data_propensity = []
    treated = []

    n_folds = 5
    train_data = train_data.sample(frac=1,random_state=args.seed) # shuffle
    N_train = len(train_data)
    N_fold = int(N_train/n_folds)

    for fold in range(n_folds):
        # get train val test subsets
        test_subset = train_data[N_fold*fold:N_fold*(fold+1)]
        train_subset = train_data.drop(test_subset.index)
        val_subset = train_subset.sample(frac=0.8,random_state=args.seed)
        train_subset = train_subset.drop(val_subset)

        # outcome prediction
        tmp_tr, tmp_val, tmp_te = process_data_outcome_prediction(train_subset,val_subset,test_subset)
        tmp_tr_dataset = data_loader(tmp_tr, mode='concat_text')
        tmp_val_dataset = data_loader(tmp_val, mode='concat_text')
        tmp_te_dataset = data_loader(tmp_te, mode='concat_text')

        outcome_trainer = train_outcome_predictor(args, tmp_tr_dataset, tmp_val_dataset)
        fold_outcome = test_outcome_predictor(outcome_trainer,tmp_te_dataset,save_preds=False)

        train_data_outcome.append(fold_outcome)
        treated.append([1 for _ in range(len(fold_outcome)//2)] + [0 for _ in range(len(fold_outcome)//2)])

        # propensity
        tmp_tr, tmp_val, tmp_te = process_data_propensity_prediction(train_subset,val_subset,test_subset)
        tmp_tr_dataset = data_loader(tmp_tr, mode='concat_text')
        tmp_val_dataset = data_loader(tmp_val, mode='concat_text')
        tmp_te_dataset = data_loader(tmp_te, mode='concat_text')

        propensity_trainer = train_propensity_predictor(args, tmp_tr_dataset, tmp_val_dataset)
        fold_propensity = test_propensity_predictor(outcome_trainer,tmp_te_dataset,save_preds=False)

        train_data_propensity.append(fold_propensity)

    train_data_outcome = np.hstack(train_data_outcome) # N datapoints * 4 - softmax 4 different classes of verdicts
    train_data_propensity = np.hstack(train_data_propensity) # N datapoints * 2 - softmax 2 classes top/rand for propensity
    treated = np.hstack(treated)

    # effect using doubly robust 
    Y = train_data['top_verdict'].apply(lambda x: [1 if x==i else 0 for i in range(4)]).values
    train_data['effects'] = doubly_robust(Y, train_data_outcome, train_data_propensity, treated) # shape: N_train * 4

    # train the effect predictor
    val_data = train_data.sample(frac=0.2,random_state=args.seed)
    train_data = train_data.drop(val_data.index)

    train_data, val_data, test_data = process_data_effect_prediction(train_data,val_data,test_data)

    effect_trainer = train_effect_predictor(args,train_data,val_data)
    test_data_effects = test_effect_predictor(effect_trainer,test_data)