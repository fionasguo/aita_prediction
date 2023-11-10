"""
Predict the causal effect using doubly robust: 

input: story+comment
output: effect estimated by doubly robust N*4
loss: MSE
"""

import os
# local_rank = int(os.environ["LOCAL_RANK"])
import time
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.data_processing import *

# MODEL = 'allenai/longformer-base-4096'
MODEL = 'bert-base-uncased'
DATADIR = '/nas/home/siyiguo/aita_prediction/data/fiona-aita-verdicts.csv'


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)

    return {"mse": mse, "rmse": rmse, "mae": mae}


def train_effect_predictor(args, train_dataset, val_dataset):
    logging.info(f'Start training effect predictor... GPU: {torch.cuda.is_available()}')
    start_time = time.time()

    training_args = TrainingArguments(
        output_dir=args.output_dir,                        # output directory
        num_train_epochs=args.num_epoch,                  # total number of training epochs
        per_device_train_batch_size=args.batch_size,       # batch size per device during training
        per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
        learning_rate=args.lr,                      # learning rate
        warmup_steps=100,                         # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                        # strength of weight decay
        logging_dir=args.output_dir+'/logs',                     # directory for storing logs
        logging_steps=5000,                         # when to logging.info log
        evaluation_strategy='steps',
        eval_steps=5000,
        load_best_model_at_end=True,              # load or not best model at the end
        seed = args.seed,
        save_steps = 5000,
        save_total_limit = 3,
        disable_tqdm=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=4)
    model.config.problem_type = 'regression'

    trainer = Trainer(
        model=model,                              # the instantiated Transformers model to be trained
        args=training_args,                       # training arguments, defined above
        train_dataset=train_dataset,              # training dataset
        eval_dataset=val_dataset,                 # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(f"./{args.output_dir}/best_model")

    logging.info(f'Finished training effect predictor. Time: {time.time()-start_time}')

    return trainer


def test_effect_predictor(trainer, args, test_dataset, save_preds=False):
    test_preds_raw, test_labels, _ = trainer.predict(test_dataset)
    logging.info(f"effect predictor testing - test_preds_raw size: {test_preds_raw.shape}, test_preds_raw: {test_preds_raw[0]}, test_labels size: {test_labels.shape}, test_labels: {test_labels[0]}")

    trainer.eval_dataset = test_dataset
    trainer.evaluate()

    if save_preds:
        with open(args.output_dir+'/effect_preds.txt','w+') as f:
            for i in test_preds_raw:
                f.write(str(i)+'\n')

        # with open(args.output_dir+'/classification_report.txt','w+') as f:
        #     f.write(report)
    
    return test_preds_raw

# if __name__ == '__main__':
#     ## command args
#     parser = argparse.ArgumentParser(description='AITA Classifier.')

#     parser.add_argument('-m','--mode', type=str, default='concat_text', help='choose from concat_text, concat_embeddings, add_embeddings')
#     parser.add_argument('-o','--output_dir', type=str, default='./output', help='output dir to be written')
#     parser.add_argument('-l','--lr', type=float, default=0.00002, help='learning rate')
#     parser.add_argument('-e','--num_epoch', type=int, default=20, help='number of epochs to train for')
#     parser.add_argument('-b','--batch_size', type=int, default=128, help='mini-batch size')
#     parser.add_argument('-s','--seed', type=int, default=3, help='random seed')

#     args = parser.parse_args()

#     ## process data
#     logging.info('Start processing data...')
#     df = load_text_data(DATADIR,args.mode)
#     train_data, val_data, test_data = train_val_test_split(df,seed=args.seed)
#     train_data, val_data, test_data = process_data_effect_prediction(train_data,val_data,test_data) # input: top_comment + story, outcome: top_verdict - rand_verdict
#     train_dataset = data_loader(train_data, mode=args.mode)
#     val_dataset = data_loader(val_data, mode=args.mode)
#     test_dataset = data_loader(test_data, mode=args.mode)

#     ## Training
#     trainer = train_effect_predictor(args, train_dataset, val_dataset)

#     ## Test
#     test_effect_predictor(trainer, test_dataset, save_preds=False)