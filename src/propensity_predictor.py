"""
Given comment and story, predict if comment is a top comment or a random comment.
"""

import os
# local_rank = int(os.environ["LOCAL_RANK"])
import argparse
import logging
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

from utils.data_processing import *
from utils.utils import create_logger
from utils.temperature_scaling import ModelWithTemperature


MODEL = 'bert-base-uncased'
DATADIR = '/nas/home/siyiguo/aita_prediction/data/fiona-aita-verdicts.csv'
# DATADIR = '/nas/home/siyiguo/aita_prediction/data/test.csv'


def train_propensity_predictor(args, train_dataset, val_dataset):
    logging.info(f'Start training propensity predictor... GPU: {torch.cuda.is_available()}')

    start_time = time.time()

    training_args = TrainingArguments(
        output_dir=args.output_dir+'/propensity_model',                        # output directory
        num_train_epochs=args.num_epoch,                  # total number of training epochs
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
    # model = AutoModelForSequenceClassification.from_pretrained(f"{args.output_dir}/propensity_model/best_model", num_labels=num_labels)

    trainer = Trainer(
        model=model,                              # the instantiated Transformers model to be trained
        args=training_args,                       # training arguments, defined above
        train_dataset=train_dataset,              # training dataset
        eval_dataset=val_dataset                  # evaluation dataset
    )

    trainer.train()

    trainer.save_model(f"{args.output_dir}/propensity_model/best_model")

    logging.info(f'Finished training propensity predictor. Time: {time.time()-start_time}')

    logging.info("temperature scaling")
    # temperature scaling on val_dataset
    scaled_model = ModelWithTemperature(trainer.model)
    temperature = scaled_model.set_temperature(trainer,val_dataset)

    return trainer, temperature


def test_propensity_predictor(trainer, args, test_dataset, temperature=1.0, save_preds=False):
    test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
    test_preds_softmax = torch.nn.functional.softmax(torch.tensor(test_preds_raw/temperature)).numpy()
    test_preds = np.argmax(test_preds_softmax, axis=-1)

    # pred performance report
    report = classification_report(test_labels, test_preds, digits=3)
    logging.info('preds with top comment')
    logging.info(report)

    if save_preds:
        with open(args.output_dir+'/propensity_preds.txt','w+') as f:
            for i in test_preds:
                f.write(str(i)+'\n')

    # with open(args.output_dir+'/classification_report.txt','w+') as f:
    #     f.write(report)

    return test_preds_softmax # return softmax vector len=2



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

    ## logger
    create_logger()

    ## process data
    logging.info('Start processing data...')
    # train_dataset, val_dataset, test_dataset = load_data(DATADIR,args.mode,args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    df = load_text_data(DATADIR)
    train_data, val_data, test_data = train_val_test_split(df,seed=args.seed)
    train_data, val_data, test_data = process_data_propensity_prediction(train_data,val_data,test_data)

    train_dataset = data_loader(train_data, tokenizer, mode=args.mode)
    val_dataset = data_loader(val_data, tokenizer, mode=args.mode)
    test_dataset = data_loader(test_data, tokenizer, mode=args.mode)

    ## Training
    trainer, temperature = train_propensity_predictor(args, train_dataset, val_dataset)
    
    ## Test
    test_preds = test_propensity_predictor(trainer, args, test_dataset, temperature, save_preds=False)
