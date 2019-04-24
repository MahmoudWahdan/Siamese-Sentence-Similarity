# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:16:27 2019

@author: mwahdan
"""

import argparse
import time
from scipy import stats
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

from word_embeddings import WordEmbeddings
from tokenizer import Tokenizer
from vectorizer import Vectorizer
from data_reader import read_SICK_data
from utils import pad_tensor
from siamese import SiameseModel


# read command-line parameters
parser = argparse.ArgumentParser('Testing the model with SICK-like data')
parser.add_argument('--model', '-p', help = 'Path to trained model.', type = str, required = True)
parser.add_argument('--word2vec', '-w', help = 'Path to word2vec .bin file with 300 dims.', type = str, required = True)
parser.add_argument('--data', '-d', help = 'Path to SICK data used for testing.', type = str, required = True)
parser.add_argument('--save', '-s', help = 'csv file path to save test output.', type = str, required = False)


args = parser.parse_args()
model_path = args.model
word_embeddings_file_path = args.word2vec
train_df, dev_df, test_df = read_SICK_data(args.data)
save_file_name = args.save


# initialize objects
print('Initializing objects ...')
print('Initializing word embeddings ...')
t1 = time.time()
word_embeddings = WordEmbeddings(word_embeddings_file_path)
t2 = time.time()
print('\tTook %f seconds' % (t2 - t1))
print('Initializing tokenizer ...')
tokenizer = Tokenizer()
print('Initializing vectorizer ...')
vectorizer = Vectorizer(word_embeddings, tokenizer)


#### testing dataset ####
print('Vectorizing testing dataset ...')
ids, test_a_vectors, test_b_vectors, test_gold = vectorizer.vectorize_df(test_df)
test_max_a_length = len(max(test_a_vectors, key=len))
test_max_b_length = len(max(test_b_vectors, key=len))
print('maximum number of tokens per sentence A in testing set is %d' % test_max_a_length)
print('maximum number of tokens per sentence B in testing set is %d' % test_max_b_length)
max_len = max([test_max_a_length, test_max_b_length])

# padding
print('Padding testing dataset ...')
test_a_vectors = pad_tensor(test_a_vectors, max_len)
test_b_vectors = pad_tensor(test_b_vectors, max_len)

print('Loading the model ...')
siamese = SiameseModel(False)
siamese.load(model_path)

print('Testing the model ...')
y = siamese.predict(test_a_vectors, test_b_vectors)
y = [i[0] for i in y]
assert len(test_gold) == len(y)

mse = mean_squared_error(test_gold, y)
print('MSE = %.2f' % mse)

pearsonr = stats.pearsonr(test_gold, y)
print('Pearson correlation (r) = %.2f' % pearsonr[0])

spearmanr = stats.spearmanr(test_gold, y)
print('Spearman’s p = %.2f' % spearmanr.correlation)

# saving
if save_file_name is not None:
    df = pd.DataFrame(np.column_stack((ids, y)), columns=['pair_ID', 'predicted_normalized_similarity'])
    df.to_csv(save_file_name, sep=',', encoding='utf-8', index=False)