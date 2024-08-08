"""
Given comment and story, predict AITA.
"""

import os
# local_rank = int(os.environ["LOCAL_RANK"])
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

from utils.data_processing import *
from utils.utils import create_logger

import os
import time
from datetime import datetime
import logging
import argparse
import csv
import transformers
from ast import literal_eval

from DANN import read_data
from DANN import DomainAdaptTrainer
from DANN import evaluate
from DANN import set_seed


DATADIR = '/nas/eclairnas01/users/siyiguo/aita_prediction/data/fiona-aita-verdicts.csv'
# DATADIR = '/nas/eclairnas01/users/siyiguo/aita_prediction/data/test.csv'
MODEL = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)



def read_config(args,curr_dir):
    """
    Read arguments from the config file.

    Args:
        args: a dict to store arguments, should at least include 'config_dir'
    """
    # default values
    args['pretrained_dir'] = 'bert-base-uncased'
    args['domain_adapt'] = True
    args['transformation'] = False
    args['reconstruction'] = False
    args['semi_supervised'] = True
    args['weighted_loss'] = True
    args['aflite'] = False
    args['train_domain'] = 'source'
    args['test_domain'] = 'target'
    args['n_mf_classes'] = 2
    args['n_domain_classes'] = 2
    args['alpha'] = 10
    args['beta'] = 0.25
    args['dropout_rate'] = 0.3
    args['lambda_trans'] = 0.0
    args['lambda_rec'] = 0.5
    args['lambda_domain'] = 0.0
    args['num_no_adv'] = 3
    args['gamma'] = 1
    args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # read args in config file
    if args['config_dir'] is not None:
        with open(args['config_dir'], 'r') as f:
            for l in f.readlines():
                # skip comments
                if l.strip() == '' or l.strip().startswith('#'):
                    continue
                # get value
                arg = l.strip().split(" = ")
                arg_name, arg_val = arg[0], arg[1]

                args[arg_name] = arg_val

        args['pretrained_dir'] = os.path.join(curr_dir, args['pretrained_dir'])
        # booleans
        args['domain_adapt'] = literal_eval(args['domain_adapt'])
        args['transformation'] = literal_eval(args['transformation'])
        args['reconstruction'] = literal_eval(args['reconstruction'])
        args['semi_supervised'] = literal_eval(args['semi_supervised'])
        args['weighted_loss'] = literal_eval(args['weighted_loss'])
        args['aflite'] = literal_eval(args['aflite'])
        # train/test domains
        if args['train_domain'][0] == '[':
            args['train_domain'] = literal_eval(args['train_domain'])
        else:
            args['train_domain'] = [args['train_domain']]
        if args['test_domain'][0] == '[':
            args['test_domain'] = literal_eval(args['test_domain'])
        else:
            args['test_domain'] = [args['test_domain']]
        # number of mf and domain classes
        args['n_mf_classes'] = int(args['n_mf_classes'])
        args['n_domain_classes'] = len(
            set(args['train_domain'] + args['test_domain']))
        # hyperparameters
        args['lr'] = float(args['lr'])
        args['alpha'] = float(args['alpha'])
        args['beta'] = float(args['beta'])
        args['batch_size'] = int(args['batch_size'])
        args['n_epoch'] = int(args['n_epoch'])
        args['dropout_rate'] = float(args['dropout_rate'])
        try:
            args['lambda_trans'] = float(args['lambda_trans'])
        except:
            args['lambda_trans'] = 0.0
        if args['lambda_trans'] == 0:
            args['transformation'] = False
        try:
            args['lambda_rec'] = float(args['lambda_rec'])
        except:
            args['lambda_rec'] = 0.0
        if args['lambda_rec'] == 0:
            args['reconstruction'] = False
        try:
            args['lambda_domain'] = 0.0
            args['num_no_adv'] = int(args['num_no_adv'])
            args['gamma'] = float(args['gamma'])
        except:
            args['lambda_domain'] = 0.0
            args['num_no_adv'] = 0
            args['gamma'] = 0.0
        # others
        args['seed'] = int(args['seed'])
        args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"

    logging.info('Configurations:')
    logging.info(args)

    return args


def train_DANN_outcome_predictor(args, datasets):
    ## command args
    root_dir = os.path.dirname(os.path.realpath(__file__))
    args = read_config(args,root_dir)

    logging.info('Start training outcome predictor...')

    start_time = time.time()

    # set up trainer
    trainer = DomainAdaptTrainer(datasets, args)

    trainer.train()

    logging.info(
        f"Finished training {args['train_domain']} data. Time: {time.time()-start_time}"
    )

    return trainer


def test_DANN_outcome_predictor(trainer,args,datasets,save_preds=False):
    start_time = time.time()

    logging.info('Evaluation DANN on Test Data\n')
    # evaluate with the best model just got from training or with the model given from config
    if trainer is not None:
        eval_model_path = args['output_dir'] + '/best_model.pth'
    elif args.get('mf_model_dir') is not None:
        eval_model_path = args['mf_model_dir']
    else:
        raise ValueError('Please provide a model for evaluation.')
    # check if test data is good
    if 'test' not in datasets or datasets['test'].mf_labels is None:
        raise ValueError('Invalid test dataset.')
    # evaluate
    test_accu, test_preds_softmax = evaluate(datasets['test'],
                            args['batch_size'],
                            model_path=eval_model_path,
                            is_adv=args['domain_adapt'],
                            test=True)
    logging.info('Macro F1 of the %s TEST dataset: %f' %
                    ('target', test_accu))

    # output predictions
    if save_preds:
        with open(args['output_dir'] + '/mf_preds.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(test_preds_softmax)

    logging.info(f"Finished evaluating test data {args['test_domain']}. Time: {time.time()-start_time}")

    return test_preds_softmax # return softmax vector len=4


# if __name__ == '__main__':
#     ## logger
#     create_logger()

#     ## command args
#     args = {}

#     parser = argparse.ArgumentParser(description='Unsupervised Time Series Clustering.')
#     parser.add_argument('-m', '--mode', type=str, required=False, default='train_test', help='train,test,or train_test')
#     parser.add_argument( '-c', '--config_dir', type=str, required=False, default=None, help='configuration file dir that specifies hyperparameters etc')
#     parser.add_argument('-i', '--data_dir', type=str, required=True, help='input data directory')
#     parser.add_argument('-o', '--output_dir', type=str, required=False, default='./output', help='output directory')
#     parser.add_argument( '-t', '--trained_model', type=str, required=False, default=None, help='if testing, it is optional to provide a trained model weight dir')
#     command_args = parser.parse_args()

#     # mode
#     mode = command_args.mode

#     # data dir
#     root_dir = os.path.dirname(os.path.realpath(__file__))
#     args['data_dir'] = os.path.join(root_dir, command_args.data_dir) if command_args.data_dir else None
#     args['config_dir'] = os.path.join(root_dir, command_args.config_dir) if command_args.config_dir else None
#     args['trained_model_dir'] = os.path.join(root_dir, command_args.trained_model) if command_args.trained_model else None
#     args['output_dir'] = os.path.join(root_dir, command_args.output_dir)
#     if not os.path.exists(os.path.join(root_dir, args['output_dir'])):
#         os.makedirs(os.path.join(root_dir, args['output_dir']))

#     root_dir = os.path.dirname(os.path.realpath(__file__))
#     args = read_config(args,root_dir)

#     ## process data
#     # set seed
#     set_seed(args['seed'])

#     # load data
#     start_time = time.time()
#     logging.info('Start processing data...')

#     # data should be in a csv file with these columns: 'text','domain' and MF_LABELS
#     datasets = read_data(
#         tokenizer,
#         data_dir=args['data_dir'],
#         mode='concat_text',
#         train_frac=0.8,
#         seed=args.seed
#     )

#     logging.info(f'Finished processing data. Time: {time.time()-start_time}')

#     ## Training
#     trainer = train_DANN_outcome_predictor(args, datasets)

#     ## Test
#     test_preds = test_DANN_outcome_predictor(trainer, args, datasets)

