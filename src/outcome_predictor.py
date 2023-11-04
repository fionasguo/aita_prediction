"""
Given comment and story, predict AITA.
"""

import os
# local_rank = int(os.environ["LOCAL_RANK"])
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

from utils.data_processing import *

# MODEL = 'allenai/longformer-base-4096'
MODEL = 'bert-base-uncased'
# DATADIR = '/home1/siyiguo/aita_prediction/data/fiona-aita-verdicts.csv'
DATADIR = 'data/test.csv'

def train_outcome_predictor(args, train_dataset, val_dataset):

    print('Start training...')

    training_args = TrainingArguments(
        output_dir=args.output_dir,                        # output directory
        num_train_epochs=args.num_epoch,                  # total number of training epochs
        per_device_train_batch_size=args.batch_size,       # batch size per device during training
        per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
        learning_rate=args.lr,                      # learning rate
        warmup_steps=100,                         # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                        # strength of weight decay
        logging_dir=args.output_dir+'/logs',                     # directory for storing logs
        logging_steps=100,                         # when to print log
        evaluation_strategy='steps',
        eval_steps=100,
        load_best_model_at_end=True,              # load or not best model at the end
        seed = args.seed,
        save_steps = 1000,
        save_total_limit = 5,
        disable_tqdm=True
    )

    num_labels = 4
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)

    trainer = Trainer(
        model=model,                              # the instantiated Transformers model to be trained
        args=training_args,                       # training arguments, defined above
        train_dataset=train_dataset,              # training dataset
        eval_dataset=val_dataset                  # evaluation dataset
    )

    trainer.train()

    trainer.save_model(f"./{args.output_dir}/best_outcome_model")

    print('Finished training.')

    return trainer


def test_outcome_predictor(trainer,test_dataset,save_preds=False):
    test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
    test_preds = np.argmax(test_preds_raw, axis=-1)

    # pred with top and rand comment
    top_com_len = len(test_labels)//2
    report_top = classification_report(test_labels[:top_com_len], test_preds[:top_com_len], digits=3)
    print('preds with top comment')
    print(report_top)
    report_rand = classification_report(test_labels[top_com_len:], test_preds[top_com_len:], digits=3)
    print('preds with random comment')
    print(report_rand)

    if save_preds:
        with open(args.output_dir+'/outcome_preds.txt','w+') as f:
            for i in test_preds:
                f.write(str(i)+'\n')
    
    return test_preds_raw # return softmax vector len=4

if __name__ == '__main__':
    ## command args
    parser = argparse.ArgumentParser(description='AITA Classifier.')

    parser.add_argument('-m','--mode', type=str, default='concat_text', help='choose from concat_text, concat_embeddings, add_embeddings')
    parser.add_argument('-o','--output_dir', type=str, default='./output', help='output dir to be written')
    parser.add_argument('-l','--lr', type=float, default=0.00002, help='learning rate')
    parser.add_argument('-e','--num_epoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('-b','--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('-s','--seed', type=int, default=3, help='random seed')

    args = parser.parse_args()

    ## process data
    print('Start processing data...')
    df = load_text_data(DATADIR,args.mode)
    train_data, val_data, test_data = train_val_test_split(df,seed=args.seed)
    train_data, val_data, test_data = process_data_outcome_prediction(train_data,val_data,test_data)
    train_dataset = data_loader(train_data, mode=args.mode)
    val_dataset = data_loader(val_data, mode=args.mode)
    test_dataset = data_loader(test_data, mode=args.mode)

    ## Training
    trainer = train_outcome_predictor(args, train_dataset, val_dataset)

    ## Test
    test_preds = test_outcome_predictor(trainer,test_dataset)

    # with open(args.output_dir+'/classification_report.txt','w+') as f:
    #     f.write(report)
