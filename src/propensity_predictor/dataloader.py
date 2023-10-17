"""
input: comment + story
output: whether this comment is a top (1) or random (0) comment
"""

import sys
sys.path.append('/home1/siyiguo/aita_prediction/src/utils')

import pandas as pd
import numpy as np
import torch
from preprocessing import preprocess_text
from transformers import AutoTokenizer


#MODEL = 'allenai/longformer-base-512'
MODEL = 'bert-base-uncased'

def gen_encodings_labels(data, mode='concat_text'):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #, local_files_only=True

    data_dict = {}
    if mode == 'concat_text':
        data1 = data.apply(lambda x: x['top_comment']+x['story'],axis=1).tolist()
        data2 = data.apply(lambda x: x['rand_comment']+x['story'],axis=1).tolist()
        data1.extend(data2)
        data_dict['encodings1'] = tokenizer(data1, truncation=True, max_length=512, padding="max_length")
        data_dict['encodings2'] = np.nan
        data_dict['label'] = [1 for _ in range(len(data))] + [0 for _ in range(len(data))]

    elif mode == 'concat_embeddings':
        data_comment = data['top_comment'].tolist() + data['rand_comment'].tolist()
        data_story = data['story'].tolist() + data['story'].tolist()
        data_dict['encodings1'] = tokenizer(data_comment,data_story, truncation=True, max_length=512, padding="max_length")
        data_dict['encodings2'] = np.nan
        data_dict['label'] = [1 for _ in range(len(data))] + [0 for _ in range(len(data))]

    elif mode == 'add_embeddings':
        # TODO: change this
        data1,data2 = pd.DataFrame(),pd.DataFrame()
        data1['encodings1'] = tokenizer(data['top_comment'].tolist(), truncation=True, max_length=512, padding="max_length")
        data1['encodings2'] = tokenizer(data['story'].tolist(), truncation=True, max_length=512, padding="max_length")
        data1['label'] = [1 for _ in range(len(data))]
        data2['encodings1'] = tokenizer(data['rand_comment'].tolist(), truncation=True, max_length=512, padding="max_length")
        data2['encodings2'] = tokenizer(data['story'].tolist(), truncation=True, max_length=512, padding="max_length")
        test2['label'] = [0 for _ in range(len(data))]

    return data_dict

def load_data(data_dir,mode='concat_text',seed=3):
    df = pd.read_csv(data_dir)

    df = df.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)
    df = df[df['top_verdict']!=-1]
    df = df[df['rand_verdict']!=-1]

    # remove empty lines, preprocess text
    for c in ['story','top_comment','rand_comment']:
        df[c] = df[c].str.replace('\n\n',' ')
        df[c] = df[c].str.replace('\n',' ')

        df[c] = df[c].apply(preprocess_text)

    # split into train val test
    # treat top and random comments as separate data points and shuffle everything, 80% train, 20% test 
    train = df.sample(frac=0.8,random_state=seed)
    test = df.drop(train.index)
    val = train.sample(frac=0.2,random_state=seed)
    train = train.drop(val.index)

    # process texts and labels based on mode
    train = gen_encodings_labels(train, mode=mode)
    val = gen_encodings_labels(val, mode=mode)
    test = gen_encodings_labels(test, mode=mode)
    print(train.keys())
    print(train['encodings1'].keys())
    print(len(train['encodings1']['input_ids']),len(train['encodings1']['input_ids'][0]))

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
