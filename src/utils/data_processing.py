import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from .preprocessing import preprocess_text


MODEL = 'bert-base-uncased'


def load_text_data(data_dir,mode='concat_text'):
    """
    load from csv and preprocess
    """
    df = pd.read_csv(data_dir,lineterminator='\n')

    # df = df.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)
    df = df[df['top_verdict']!=-1]
    df = df[df['rand_verdict']!=-1]

    # remove empty lines, preprocess text
    for c in ['story','top_comment','rand_comment']:
        df[c] = df[c].str.replace('\n\n',' ')
        df[c] = df[c].str.replace('\n',' ')

        df[c] = df[c].apply(preprocess_text)

    return df


def train_val_test_split(df,tr_frac=0.8,seed=3):
    train = df.sample(frac=tr_frac,random_state=seed)
    test = df.drop(train.index)
    val = train.sample(frac=1-tr_frac,random_state=seed)
    train = train.drop(val.index)

    return train, val, test


def train_test_split(df,tr_frac=0.8,seed=3):
    train = df.sample(frac=tr_frac,random_state=seed)
    test = df.drop(train.index)

    return train, test


def process_data_outcome_prediction(train,val,test):
    train = train[['top_comment','story','top_verdict']]
    train = train.rename(columns={'top_comment':'comment','top_verdict':'label'})

    val = val[['top_comment','story','top_verdict']]
    val = val.rename(columns={'top_comment':'comment','top_verdict':'label'})

    test = pd.concat([test[['top_comment','story','top_verdict']].rename(columns={'top_comment':'comment','top_verdict':'label'}),test[['rand_comment','story','rand_verdict']].rename(columns={'rand_comment':'comment','rand_verdict':'label'})],axis=0)

    return train, val, test


def process_data_propensity_prediction(train,val,test):
    def helper(df):
        N = len(df)
        df = pd.concat([df[['top_comment','story']].rename(columns={'top_comment':'comment'}),df[['rand_comment','story']].rename(columns={'rand_comment':'comment'})],axis=0)
        df['label'] = [1 for _ in range(N)] + [0 for _ in range(N)]
        df = df.reset_index(drop=True)
        return df
    
    return helper(train), helper(val), helper(test)


def process_data_effect_prediction(train,val,test):
    train = train[['top_comment','story','effect']]
    train = train.rename(columns={'top_comment':'comment','effect':'label'})

    val = val[['top_comment','story','effect']]
    val = val.rename(columns={'top_comment':'comment','effect':'label'})

    # labels are one hot of length 4
    test['top_verdict_onehot'] = test['top_verdict'].apply(lambda x: [1 if x==i else 0 for i in range(4)])
    test['rand_verdict_onehot'] = test['rand_verdict'].apply(lambda x: [1 if x==i else 0 for i in range(4)])
    test['label'] = test.apply(lambda x: [t-r for t,r in zip(x['top_verdict_onehot'],x['rand_verdict_onehot'])],axis=1)
    test = test[['top_comment','story','label']]
    test = test.rename(columns={'top_comment':'comment'})

    return train, val, test


def data_loader(df, tokenizer, mode='concat_text'):
    if mode == 'concat_text':
        encodings = tokenizer(df.apply(lambda x: x['comment']+x['story'],axis=1).tolist(), truncation=True, max_length=512, padding="max_length")
        labels = df['label'].tolist()

    elif mode == 'concat_embedding':
        encodings = tokenizer(df['comment'].tolist(),df['story'].tolist(), truncation=True, max_length=512, padding="max_length")
        labels = df['label'].tolist()
    
    elif mode == 'story_only':
        encodings = tokenizer(df['story'].tolist(), truncation=True, max_length=512, padding="max_length")
        labels = df['label'].tolist()

    dataset = AITAData(encodings, labels)

    return dataset


class AITAData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)