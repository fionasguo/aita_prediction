"""
Read train/dev/test datasets from a csv file (data_dir).

data_dir can either be:
- a csv path, where train/val/test are all together and mf labels are present in all data examples, and the functions will split them. This is mostly for testing models.
- a dir where separate csv files exist for train/val/test data. Some sets might be missing (eg test) in this case.

The data csv needs to have these columns: 'text','domain' and all columns in desired mf_label_names.
"""

import os
import logging
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple, Dict

from .preprocessing import preprocess_tweet
from .data_loader import MFData

pd.options.mode.chained_assignment = None


def create_Data(df, tokenizer, mode='concat_text'):
    if mode == 'concat_text':
        encodings = df.apply(lambda x: tokenizer(x['comment']+x['story'], truncation=True, max_length=512, padding="max_length"),axis=1).tolist()

    elif mode == 'concat_embedding':
        encodings = df.apply(lambda x: tokenizer(x['comment'],x['story'], truncation=True, max_length=512, padding="max_length"),axis=1).tolist()
    
    elif mode == 'story_only':
        encodings = df['story'].apply(tokenizer, truncation=True, max_length=512, padding="max_length").tolist()

    labels = df['label'].tolist()
    domain_labels  = df['domain_idx'].tolist()

    dataset = MFData(encodings, labels, domain_labels)

    return dataset


def read_data(
        tokenizer,
        train: pd.DataFrame = None,
        val: pd.DataFrame = None,
        test: pd.DataFrame = None,
        data_dir: str = None,
        mode: str = 'concat_text',
        train_frac: float = 0.8,
        seed: int = 3
    ) -> Dict[str, MFData]:
    """
    Generate train/dev/test datasets.

    Args:
        data_dir: should be a csv with these columns: 'text','domain' and mf_label_names
        tokenizer_path: where to load pretrained tokenizer
        n_mf_classes: number of moral foundation classes in the data
        train_domain: the name of the domains for training, should be in the 'domain' column in df
        test_domain: the name of the domains for testing
        semi_supervised: if true, separate train/dev/test data by source and target domains
        aflite: whether to run AFlite algorithm to denoise the training data
        max_seq_length: max number of tokens from each input text used by the tokenizer
        train_frac: the fraction of number of data points used for training
        seed: random seed

    Returns:
        dictionary of train, val, test MFData objects
    """
    if data_dir is not None:
        # if data_dir is a csv, peform train test split
        df = pd.read_csv(data_dir, lineterminator='\n')
        # preprocess
        # df = df[~df.text.isnull()]
        df.loc[df.text.isnull(),'text'] = ' '
        df.text = df.text.apply(preprocess_tweet)
        df = df.reset_index()

        train = df.sample(train_frac,random_state=seed)
        test = df.drop(train.index)
        val = train.sample(1-train_frac,random_state=seed)
        train = train.drop(val.index)

    # construct the source target train val test
    s_train = train[['top_comment','story','top_verdict']]
    s_train = s_train.rename(columns={'top_comment':'comment','top_verdict':'label'})
    s_train['domain_idx'] = 0

    t_train = train[['rand_comment','story','rand_verdict']]
    t_train = t_train.rename(columns={'rand_comment':'comment','rand_verdict':'label'})
    t_train['domain_idx'] = 1

    s_val = val[['top_comment','story','top_verdict']]
    s_val = s_val.rename(columns={'top_comment':'comment','top_verdict':'label'})
    s_val['domain_idx'] = 0

    t_val = val[['rand_comment','story','rand_verdict']]
    t_val = t_val.rename(columns={'rand_comment':'comment','rand_verdict':'label'})
    t_val['domain_idx'] = 1

    N = len(test)
    test = pd.concat([test[['top_comment','story','top_verdict']].rename(columns={'top_comment':'comment','top_verdict':'label'}),test[['rand_comment','story','rand_verdict']].rename(columns={'rand_comment':'comment','rand_verdict':'label'})],axis=0)
    test['domain_idx'] = [0 for _ in range(N)] + [1 for _ in range(N)]

    logging.info(f"s_train size: {s_train.shape}, t_train size: {t_train.shape}, s_val size: {s_val.shape}, t_val size: {t_val.shape}, test size: {test.shape}")

    # encode the text and take the labels
    datasets = {
        's_train': s_train,
        't_train': t_train,
        's_val': s_val,
        't_val': t_val,
        'test': test
    }
    for k, v in datasets.items():
        datasets[k] = create_Data(v, tokenizer, mode)

    return datasets
