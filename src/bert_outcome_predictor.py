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


MODEL = 'bert-base-uncased'
DATADIR = '/nas/home/siyiguo/aita_prediction/data/fiona-aita-verdicts.csv'
# DATADIR = 'data/test.csv'

def train_bert_outcome_predictor(args, train_dataset, val_dataset):

    logging.info('Start training outcome predictor...')

    training_args = TrainingArguments(
        output_dir=args.output_dir+'/outcome_bert',                        # output directory
        num_train_epochs=args.n_epoch,                  # total number of training epochs
        per_device_train_batch_size=args.batch_size,       # batch size per device during training
        per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
        learning_rate=args.lr,                      # learning rate
        warmup_steps=100,                         # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                        # strength of weight decay
        logging_dir=args.output_dir+'/logs',                     # directory for storing logs
        logging_steps=500,                         # when to logging.info log
        evaluation_strategy='steps',
        eval_steps=500,
        load_best_model_at_end=True,              # load or not best model at the end
        seed = args.seed,
        save_steps = 5000,
        save_total_limit = 3,
        disable_tqdm=True
    )

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)

    trainer = Trainer(
        model=model,                              # the instantiated Transformers model to be trained
        args=training_args,                       # training arguments, defined above
        train_dataset=train_dataset,              # training dataset
        eval_dataset=val_dataset                  # evaluation dataset
    )

    trainer.train()

    trainer.save_model(f"{args.output_dir}/outcome_bert/best_model")

    logging.info('Finished training outcome predictor.')

    return trainer


def test_bert_outcome_predictor(trainer,args,test_dataset,save_preds=False):
    test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
    test_preds_softmax = torch.nn.functional.softmax(torch.tensor(test_preds_raw)).numpy()
    test_preds = np.argmax(test_preds_softmax, axis=-1)

    # pred with top and rand comment
    report_observed = classification_report(test_labels[np.array(test_dataset.T)==0], test_preds[np.array(test_dataset.T)==0], digits=3)
    logging.info('preds with observed data')
    logging.info(report_observed)
    report_rand = classification_report(test_labels[np.array(test_dataset.T)==1], test_preds[np.array(test_dataset.T)==1], digits=3)
    logging.info('preds with transformed data')
    logging.info(report_rand)

    if save_preds:
        with open(args.output_dir+'/outcome_bert/outcome_preds.txt','w+') as f:
            for i in test_preds:
                f.write(str(i)+'\n')
    
    return test_preds_softmax # return softmax vector len=4

# if __name__ == '__main__':
#     ## command args
#     parser = argparse.ArgumentParser(description='AITA Classifier.')

#     parser.add_argument('-m','--mode', type=str, default='concat_text', help='choose from concat_text, concat_embeddings, add_embeddings')
#     parser.add_argument('-o','--output_dir', type=str, default='./output', help='output dir to be written')
#     parser.add_argument('-l','--lr', type=float, default=0.00002, help='learning rate')
#     parser.add_argument('-e','--n_epoch', type=int, default=20, help='number of epochs to train for')
#     parser.add_argument('-b','--batch_size', type=int, default=128, help='mini-batch size')
#     parser.add_argument('-s','--seed', type=int, default=3, help='random seed')

#     args = parser.parse_args()

#     ## logger
#     create_logger()

#     ## process data
#     logging.info('Start processing data...')
#     tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
#     df = load_text_data(DATADIR)
#     train_data, val_data, test_data = train_val_test_split(df,seed=args.seed)
#     train_data, val_data, test_data = process_data_outcome_prediction(train_data,val_data,test_data)
    
#     train_dataset = data_loader(train_data, tokenizer, mode=args.mode)
#     val_dataset = data_loader(val_data, tokenizer, mode=args.mode)
#     test_dataset = data_loader(test_data, tokenizer, mode=args.mode)

#     ## Training
#     trainer = train_bert_outcome_predictor(args, train_dataset, val_dataset)

#     ## Test
#     test_preds = test_bert_outcome_predictor(trainer, args, test_dataset)

#     # with open(args.output_dir+'/classification_report.txt','w+') as f:
#     #     f.write(report)
