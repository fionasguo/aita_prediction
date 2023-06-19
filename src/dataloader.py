import pandas as pd
import numpy as np
import torch
from preprocessing import preprocess_text
from transformers import AutoTokenizer


MODEL = 'allenai/longformer-base-4096'

def gen_encodings_labels(train, test, mode='concat_text'):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True,local_files_only=True)

    if mode == 'concat_text':
        train['encodings1'] = tokenizer(train.apply(lambda x: x['top_comment']+x['story'],axis=0).tolist(), truncation=True, max_length=4096, padding="max_length")
        train['encodings2'] = np.nan
        train['label'] = train['top_verdict']
        
        test1,test2 = pd.DataFrame(),pd.DataFrame()
        test1['encodings1'] = tokenizer(test.apply(lambda x: x['top_comment']+x['story'],axis=0).tolist(), truncation=True, max_length=4096, padding="max_length")
        test1['encodings2'] = np.nan
        test1['label'] = test['top_verdict']
        test2['encodings1'] = tokenizer(test.apply(lambda x: x['rand_comment']+x['story'],axis=0).tolist(), truncation=True, max_length=4096, padding="max_length")
        test2['encodings2'] = np.nan
        test2['label'] = test['rand_verdict']

    elif mode == 'concat_embeddings':
        train['encodings1'] = tokenizer(train['top_comment'].tolist(),train['story'].tolist(), truncation=True, max_length=4096, padding="max_length")
        train['encodings2'] = np.nan
        train['label'] = train['top_verdict']

        test1,test2 = pd.DataFrame(),pd.DataFrame()
        test1['encodings1'] = tokenizer(test['top_comment'].tolist(),test['story'].tolist(), truncation=True, max_length=4096, padding="max_length")
        test1['encodings2'] = np.nan
        test1['label'] = test['top_verdict']
        test2['encodings1'] = tokenizer(test['rand_comment'].tolist(),test['story'].tolist(), truncation=True, max_length=4096, padding="max_length")
        test2['encodings2'] = np.nan
        test2['label'] = test['rand_verdict']

    elif mode == 'add_embeddings':
        train['encodings1'] = tokenizer(train['top_comment'].tolist(), truncation=True, max_length=4096, padding="max_length")
        train['encodings2'] = tokenizer(train['story'].tolist(), truncation=True, max_length=4096, padding="max_length")
        train['label'] = train['top_verdict']

        test1,test2 = pd.DataFrame(),pd.DataFrame()
        test1['encodings1'] = tokenizer(test['top_comment'].tolist(), truncation=True, max_length=4096, padding="max_length")
        test1['encodings2'] = tokenizer(test['story'].tolist(), truncation=True, max_length=4096, padding="max_length")
        test1['label'] = test['top_verdict']
        test2['encodings1'] = tokenizer(test['rand_comment'].tolist(), truncation=True, max_length=4096, padding="max_length")
        test2['encodings2'] = tokenizer(test['story'].tolist(), truncation=True, max_length=4096, padding="max_length")
        test2['label'] = test['rand_verdict']

    test = pd.concat([test1,test2],axis=0)

    return train,test

def load_data(data_dir,mode='concat_text',seed=3):
    df = pd.read_csv(data_dir)

    df = df.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)

    # remove empty lines, preprocess text
    for c in ['story','top_comment','rand_comment']:
        df[c] = df[c].str.replace('\n\n',' ')
        df[c] = df[c].str.replace('\n',' ')

        df[c] = df[c].apply(preprocess_text)

    # split into train val test
    # take out 80% posts first and train with top comment, and for 20% test set I will treat top and random comments as separate data points and shuffle everything
    train = df.sample(frac=0.8,random_state=seed)
    test = df.drop(train.index)

    # process texts and labels based on mode
    train,test = gen_encodings_labels(train, test, mode=mode)
    val = train.sample(frac=0.2,random_state=seed)
    train = train.drop(val.index)

    # create dataset objects
    datasets = []
    for d in [train,val,test]:
        dataset = AITAData(d['encodings1'], d['encodings2'], d['label'])
        datasets.append(dataset)

    return datasets
    

class AITAData(torch.utils.data.Dataset):
    def __init__(self, encodings1, encodings2, labels):
        self.encodings1 = encodings1
        self.encodings2 = encodings2
        self.labels = labels

    def __getitem__(self, idx):
        ## TODO: write the add_embeddings situation
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings1.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)