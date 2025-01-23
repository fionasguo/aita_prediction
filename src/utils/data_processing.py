import pandas as pd
import numpy as np
import logging
import torch
from transformers import AutoTokenizer
from .preprocessing import preprocess_text


MODEL = 'bert-base-uncased'


def load_text_data(data_dir):
    """
    load from csv and preprocess
    """
    df = pd.read_csv(data_dir,lineterminator='\n')
    try:
        df = df[~df['pair_id'].isin(df.loc[df['text'].isnull(),'pair_id'].tolist())]
    except:
        pass

    # # df = df.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)
    # df = df[df['top_verdict']!=-1]
    # df = df[df['rand_verdict']!=-1]

    # remove empty lines, preprocess text
    # for c in ['story','top_comment','rand_comment']:
    #     df[c] = df[c].str.replace('\n\n',' ')
    #     df[c] = df[c].str.replace('\n',' ')

    df = df[df['text'].str.split(' ').apply(len)>=5]
    # df['text'] = df['text'].apply(preprocess_text)
    logging.info(f"df loaded shape={df.shape}")
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


def process_data_bert_outcome_prediction(train,val,test):
    # train = train[['top_comment','story','top_verdict']]
    # train = train.rename(columns={'top_comment':'comment','top_verdict':'label'})

    # val = val[['top_comment','story','top_verdict']]
    # val = val.rename(columns={'top_comment':'comment','top_verdict':'label'})

    # test = pd.concat([test[['top_comment','story','top_verdict']].rename(columns={'top_comment':'comment','top_verdict':'label'}),test[['rand_comment','story','rand_verdict']].rename(columns={'rand_comment':'comment','rand_verdict':'label'})],axis=0)

    train = train.loc[train['T']==0,['text','T','X','y']] # assume we only observe untreated units
    train['label'] = train['y']
    val = val.loc[val['T']==0,['text','T','X','y']]
    val['label'] = val['y']
    test = test[['text','T','X','y']]
    test['label'] = test['y']

    return train, val, test


def process_data_propensity_prediction(train,val,test):
    def helper(df):
        df = df[['text','T','X','y']]
        df['label'] = df['T']
        df = df.reset_index(drop=True)
        return df
    
    return helper(train), helper(val), helper(test)


def process_data_effect_prediction(train,val,test,cov_dim):
    # train = train[['top_comment','story','effect']]
    # train = train.rename(columns={'top_comment':'comment','effect':'label'})

    # val = val[['top_comment','story','effect']]
    # val = val.rename(columns={'top_comment':'comment','effect':'label'})

    # # labels are one hot of length 4
    # test['top_verdict_onehot'] = test['top_verdict'].apply(lambda x: [1 if x==i else 0 for i in range(4)])
    # test['rand_verdict_onehot'] = test['rand_verdict'].apply(lambda x: [1 if x==i else 0 for i in range(4)])
    # test['label'] = test.apply(lambda x: [t-r for t,r in zip(x['top_verdict_onehot'],x['rand_verdict_onehot'])],axis=1)
    # test = test[['top_comment','story','label']]
    # test = test.rename(columns={'top_comment':'comment'})

    train = train[['X','effect']]
    train['X_onehot'] = train['X'].apply(lambda x: [1 if x==i else 0 for i in range(cov_dim)])
    train = train.rename(columns={'effect':'label'})

    val = val[['X','effect']]
    val['X_onehot'] = val['X'].apply(lambda x: [1 if x==i else 0 for i in range(cov_dim)])
    val = val.rename(columns={'effect':'label'})

    test_0 = test[test['T']==0].sort_values('id')
    test_1 = test[test['T']==1].sort_values('id')
    test_ = test_0[['id','X','y']]
    
    return train, val, test


def data_loader(df, tokenizer): #, mode='concat_text'):
    # if mode == 'concat_text':
    #     encodings = tokenizer(df.apply(lambda x: x['comment']+x['story'],axis=1).tolist(), truncation=True, max_length=512, padding="max_length")
    #     labels = df['label'].tolist()

    # elif mode == 'concat_embedding':
    #     encodings = tokenizer(df['comment'].tolist(),df['story'].tolist(), truncation=True, max_length=512, padding="max_length")
    #     labels = df['label'].tolist()
    
    # elif mode == 'story_only':
    #     encodings = tokenizer(df['story'].tolist(), truncation=True, max_length=512, padding="max_length")
    #     labels = df['label'].tolist()

    encodings = tokenizer(df['text'].tolist(), truncation=True, max_length=200, padding="max_length")
    dataset = AITAData(encodings, df['label'].tolist(), df['T'].tolist())

    return dataset


class AITAData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, T):
        self.encodings = encodings
        self.labels = labels
        self.T = T

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)