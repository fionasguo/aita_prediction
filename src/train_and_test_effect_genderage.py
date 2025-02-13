import os
import argparse
import pandas as pd
import numpy as np
import logging

from utils.data_processing import *
from utils.compute_effect import *
from utils.utils import create_logger
from evaluate import evaluate

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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data_folder_dir = os.path.dirname(args.input_dir)

    # logger
    create_logger(args.output_dir)

    logging.info(f"Data: {args.input_dir}. Training Effect prector with {args.model} model, potential outcome predicted by {args.outcome_model}, seed={args.seed}")
    logging.info(args)

    # load and preprocessing text data
    df = load_text_data(args.input_dir)
    df = df.set_index('id')

    # read outcome, id, propensity
    train_data_outcome = np.loadtxt(f'{data_folder_dir}/seed_{args.seed}/train_{args.outcome_model}_outcome.csv',delimiter=',') # N*1
    train_data_id = np.loadtxt(f'{data_folder_dir}/seed_{args.seed}/train_{args.outcome_model}_id.csv',delimiter=',') # N*1
    logging.info(f"loaded train_data outcome shape={train_data_outcome.shape}, id shape={train_data_id.shape}")

    train_data = df.loc[train_data_id,]
    test_data = df.drop(train_data.index)

    if args.cov_dim == 2:
        covariates = train_data['X'].values
    else:
        # one-hot encoding
        covariates = np.array(train_data['X'].apply(lambda x: [1 if x==i else 0 for i in range(args.cov_dim)]).tolist())

    # evaluate
    logging.info(f"\nComputing Oracle CATE:")
    cate_oracle = CATE_naive(train_data['y'].values,train_data['T'].values,covariates)

    if args.model == 'DR':
        train_data_propensity = np.loadtxt(f'{data_folder_dir}/seed_{args.seed}/train_propensity.csv',delimiter=',') # N*2
        # train_data_propensity = estimate_propensities(train_data['T'].tolist(),train_data['X'].tolist()) # for Amazon review data
        logging.info(f"loaded train_data propensity shape={train_data_propensity.shape}")

        # cate = CATE_doubly_robust(train_data['y'].values, train_data_outcome, train_data_propensity, train_data['T'].values, covariates)
        # cate_oracle_, cate = [cate_oracle[key] for key in sorted(cate_oracle.keys())], [cate[key] for key in sorted(cate.keys())]
        # logging.info(f"Evaluate DR CATE: delta={np.mean(cate)-np.mean(cate_oracle_)}, {evaluate(cate_oracle_,cate)}")

        cate = CATE_IPW(train_data['y'].values, train_data_outcome, train_data_propensity, train_data['T'].values, covariates)
        cate_oracle_, cate = [cate_oracle[key] for key in sorted(cate_oracle.keys())], [cate[key] for key in sorted(cate.keys())]
        logging.info(f"Evaluate IPW CATE: delta={np.mean(cate)-np.mean(cate_oracle_)}, {evaluate(cate_oracle_,cate)}")
        
        for gender in [0,1,-1]:
            gender_mask = train_data['gender']==gender
            cate = CATE_IPW(train_data.loc[gender_mask,'y'].values, train_data_outcome[gender_mask.values], train_data_propensity[gender_mask.values], train_data.loc[gender_mask,'T'].values, covariates[gender_mask.values])
            cate_oracle = CATE_naive(train_data.loc[gender_mask,'y'].values,train_data.loc[gender_mask,'T'].values,covariates[gender_mask.values])
            cate_oracle_, cate = [cate_oracle[key] for key in sorted(cate_oracle.keys())], [cate[key] for key in sorted(cate.keys())]
            logging.info(f"Evaluate IPW CATE (Gender={gender}): delta={np.mean(cate)-np.mean(cate_oracle_)}, {evaluate(cate_oracle_,cate)}")
        for age in [1,2,3,4,-1]:
            age_mask = train_data['age']==age
            cate = CATE_IPW(train_data.loc[age_mask,'y'].values, train_data_outcome[age_mask.values], train_data_propensity[age_mask.values], train_data.loc[age_mask,'T'].values, covariates[age_mask.values])
            cate_oracle = CATE_naive(train_data.loc[age_mask,'y'].values,train_data.loc[age_mask,'T'].values,covariates[age_mask.values])
            cate_oracle_, cate = [cate_oracle[key] for key in sorted(cate_oracle.keys())], [cate[key] for key in sorted(cate.keys())]
            logging.info(f"Evaluate IPW CATE (Age={age}): delta={np.mean(cate)-np.mean(cate_oracle_)}, {evaluate(cate_oracle_,cate)}")
        
        if args.ite:
            logging.info("\nComputing ITE:")
            ite_oracle = ITE_naive(train_data['y'].values,train_data['T'].values,train_data['pair_id'].values)
            # ite = ITE_doubly_robust(train_data['y'].values, train_data_outcome, train_data_propensity, train_data['T'].values, train_data['pair_id'].values)
            # logging.info(f"Evaluate DR ITE: {evaluate(ite_oracle,ite)}")
            ite = ITE_IPW(train_data['y'].values, train_data_outcome, train_data_propensity, train_data['T'].values, train_data['pair_id'].values)
            logging.info(f"Evaluate IPW ITE: {evaluate(ite_oracle,ite)}")

            for gender in [0,1,-1]:
                gender_mask = train_data['gender']==gender
                ite_oracle = ITE_naive(train_data.loc[gender_mask,'y'].values,train_data.loc[gender_mask,'T'].values,train_data.loc[gender_mask,'pair_id'].values)
                ite = ITE_IPW(train_data.loc[gender_mask,'y'].values, train_data_outcome[gender_mask.values], train_data_propensity[gender_mask.values], train_data.loc[gender_mask,'T'].values, train_data.loc[gender_mask,'pair_id'].values)
                logging.info(f"Evaluate IPW ITE (Gender={gender}): {evaluate(ite_oracle,ite)}")
            for age in [1,2,3,4,-1]:
                age_mask = train_data['age']==age
                ite_oracle = ITE_naive(train_data.loc[age_mask,'y'].values,train_data.loc[age_mask,'T'].values,train_data.loc[age_mask,'pair_id'].values)
                ite = ITE_IPW(train_data.loc[age_mask,'y'].values, train_data_outcome[age_mask.values], train_data_propensity[age_mask.values], train_data.loc[age_mask,'T'].values, train_data.loc[age_mask,'pair_id'].values)
                logging.info(f"Evaluate IPW ITE(Age={age}): {evaluate(ite_oracle,ite)}")


    elif args.model == 'none':
        cate = CATE_naive(train_data_outcome,train_data['T'].values,covariates)
        cate_oracle, cate = [cate_oracle[key] for key in sorted(cate_oracle.keys())], [cate[key] for key in sorted(cate.keys())]
        logging.info(f"Evaluate CATE: delta={np.mean(cate)-np.mean(cate_oracle)}, {evaluate(cate_oracle,cate)}")

        for gender in [0,1,-1]:
            gender_mask = train_data['gender']==gender
            cate = CATE_naive(train_data_outcome[gender_mask.values],train_data.loc[gender_mask,'T'].values,covariates[gender_mask.values])
            cate_oracle = CATE_naive(train_data.loc[gender_mask,'y'].values,train_data.loc[gender_mask,'T'].values,covariates[gender_mask.values])
            cate_oracle, cate = [cate_oracle[key] for key in sorted(cate_oracle.keys())], [cate[key] for key in sorted(cate.keys())]
            logging.info(f"Evaluate CATE (Gender={gender}): delta={np.mean(cate)-np.mean(cate_oracle)}, {evaluate(cate_oracle,cate)}")
        for age in [1,2,3,4,-1]:
            age_mask = train_data['age']==age
            cate = CATE_naive(train_data_outcome[age_mask.values],train_data.loc[age_mask,'T'].values,covariates[age_mask.values])
            cate_oracle = CATE_naive(train_data.loc[age_mask,'y'].values,train_data.loc[age_mask,'T'].values,covariates[age_mask.values])
            cate_oracle, cate = [cate_oracle[key] for key in sorted(cate_oracle.keys())], [cate[key] for key in sorted(cate.keys())]
            logging.info(f"Evaluate CATE (Age={age}): delta={np.mean(cate)-np.mean(cate_oracle)}, {evaluate(cate_oracle,cate)}")

        if args.ite:
            logging.info("\nComputing ITE:")
            ite = ITE_naive(train_data_outcome,train_data['T'].values,train_data['pair_id'].values)
            ite_oracle = ITE_naive(train_data['y'].values,train_data['T'].values,train_data['pair_id'].values)
            logging.info(f"Evaluate ITE: {evaluate(ite_oracle,ite)}")

            for gender in [0,1,-1]:
                gender_mask = train_data['gender']==gender
                ite = ITE_naive(train_data_outcome[gender_mask],train_data.loc[gender_mask,'T'].values,train_data.loc[gender_mask,'pair_id'].values)
                ite_oracle = ITE_naive(train_data.loc[gender_mask,'y'].values,train_data.loc[gender_mask,'T'].values,train_data.loc[gender_mask,'pair_id'].values)
                logging.info(f"Evaluate ITE (Gender={gender}): {evaluate(ite_oracle,ite)}")
            for age in [1,2,3,4,-1]:
                age_mask = train_data['age']==age
                ite = ITE_naive(train_data_outcome[age_mask.values],train_data.loc[age_mask,'T'].values,train_data.loc[age_mask,'pair_id'].values)
                ite_oracle = ITE_naive(train_data.loc[age_mask,'y'].values,train_data.loc[age_mask,'T'].values,train_data.loc[age_mask,'pair_id'].values)
                logging.info(f"Evaluate ITE (Age={age}): {evaluate(ite_oracle,ite)}")



