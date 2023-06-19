import argparse

import pandas as pd
import numpy as np
import torch

from dataloader import *

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from sklearn.metrics import classification_report

MODEL = 'allenai/longformer-base-4096'
DATADIR = ''


if __name__ == '__main__':
    ## command args
    parser = argparse.ArgumentParser(description='AITA Classifier.')

    parser.add_argument('-m','--mode', type=str, default='concat_text', help='choose from concat_text, concat_embeddings, add_embeddings')
    parser.add_argument('-o','--output_dir', type=str, default='./aita_output', help='output dir to be written')
    parser.add_argument('-l','--lr', type=float, default=0.00002, help='learning rate')
    parser.add_argument('-e','--num_epoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('-b','--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('-s','--seed', type=int, default=3, help='random seed')

    args = parser.parse_args()

    ## process data
    print('Start processin data...')
    train_dataset, val_dataset, test_dataset = load_data(DATADIR,args.mode,args.seed)

    ## Training
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
    )

    num_labels = 4
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels,local_files_only=True)

    trainer = Trainer(
        model=model,                              # the instantiated Transformers model to be trained
        args=training_args,                       # training arguments, defined above
        train_dataset=train_dataset,              # training dataset
        eval_dataset=val_dataset                  # evaluation dataset
    )

    trainer.train()

    trainer.save_model(f"./{args.output_dir}/best_model")

    print('Finished training.')

    ## Test
    test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
    test_preds = np.argmax(test_preds_raw, axis=-1)
    report = classification_report(test_labels, test_preds, digits=3)
    print(report)

    with open(args.output_dir+'/test_preds.txt','w+') as f:
        for i in test_preds:
            f.write(i+'\n')

    with open(args.output_dir+'/classification_report.txt','w+') as f:
        f.write(report)
