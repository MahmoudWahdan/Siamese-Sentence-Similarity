# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:34:32 2018

@author: mwahdan
"""
import pandas as pd

def read_SICK_data(file_path, normalize_scores=True):
    df = pd.read_csv(file_path, sep='\t')
    df = df[['pair_ID', 'sentence_A', 'sentence_B', 'relatedness_score', 'SemEval_set']]
    # relatedness_score: semantic relatedness gold score (on a 1-5 continuous scale)
    # normalize score to range [0,1]
    if normalize_scores:
        min_score = 1.0
        max_score = 5.0
        df['relatedness_score'] = (df['relatedness_score'] - min_score) / (max_score - min_score)
    
    # split dataset
    train_df = df[df['SemEval_set'] == 'TRAIN']
    dev_df = df[df['SemEval_set'] == 'TRIAL']
    test_df = df[df['SemEval_set'] == 'TEST']
    return train_df, dev_df, test_df
    
def read_SEMEVAL_data(file_path, normalize_scores=True):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['relatedness_score', 'sentence_A', 'sentence_B'])
    # relatedness_score: semantic relatedness gold score (on a 0-5 continuous scale)
    # normalize score to range [0,1]
    if normalize_scores:
        min_score = 0.0
        max_score = 5.0
        df['relatedness_score'] = (df['relatedness_score'] - min_score) / (max_score - min_score)
    return df

if __name__ == '__main__':
    df = read_SEMEVAL_data('./data/sts/semeval-sts/2013/OnWN.test.tsv')