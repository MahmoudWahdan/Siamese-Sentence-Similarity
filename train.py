# -*- coding: utf-8 -*-
"""
Created on Fri May 25 00:09:54 2018

@author: mwahdan
"""

import argparse
import time
from scipy import stats
from sklearn.metrics import mean_squared_error

from word_embeddings import WordEmbeddings
from tokenizer import Tokenizer
from vectorizer import Vectorizer
from data_reader import read_SICK_data
from utils import pad_tensor, str2bool
from siamese import SiameseModel


# read command-line parameters
parser = argparse.ArgumentParser('Training the model with SICK-like data')
parser.add_argument('--word2vec', '-w', help = 'Path to word2vec .bin file with 300 dims.', type = str, required = True)
parser.add_argument('--data', '-d', help = 'Path to SICK data used for training.', type = str, required = True)
parser.add_argument('--pretrained', '-p', help = 'Path to pre-trained weights.', type = str, required = False)
parser.add_argument('--epochs', '-e', help = 'Number of epochs.', type = int, default = 100, required = False)
parser.add_argument('--save', '-s', help = 'Folder path to save both the trained model and its weights.', type = str, required = False)
parser.add_argument('--cudnnlstm', '-c', help = 'Use CUDNN LSTM for fast training. This requires GPU and CUDA.', type = str2bool, default='true', required = False)
#print(parser.format_help())

args = parser.parse_args()
word_embeddings_file_path = args.word2vec
train_df, dev_df, test_df = read_SICK_data(args.data)
pretrained = args.pretrained
epochs = args.epochs
save_path = args.save
use_cudnn_lstm = args.cudnnlstm

print('use_cudnn_lstm: ', use_cudnn_lstm)

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

#### training dataset ####
# vectorizing
ids, train_a_vectors, train_b_vectors, train_gold = vectorizer.vectorize_df(train_df)
train_max_a_length = len(max(train_a_vectors, key=len))
train_max_b_length = len(max(train_b_vectors, key=len))
print('maximum number of tokens per sentence A in training set is %d' % train_max_a_length)
print('maximum number of tokens per sentence B in training set is %d' % train_max_b_length)
max_len = max([train_max_a_length, train_max_b_length])

# padding
train_a_vectors = pad_tensor(train_a_vectors, max_len)
train_b_vectors = pad_tensor(train_b_vectors, max_len)


#### development dataset ####
# vectorizing
ids, dev_a_vectors, dev_b_vectors, dev_gold = vectorizer.vectorize_df(dev_df)
dev_max_a_length = len(max(dev_a_vectors, key=len))
dev_max_b_length = len(max(dev_b_vectors, key=len))
print('maximum number of tokens per sentence A in dev set is %d' % dev_max_a_length)
print('maximum number of tokens per sentence B in dev set is %d' % dev_max_b_length)
max_len = max([dev_max_a_length, dev_max_b_length])

# padding
dev_a_vectors = pad_tensor(dev_a_vectors, max_len)
dev_b_vectors = pad_tensor(dev_b_vectors, max_len)


print('Training the model ...')
siamese = SiameseModel(use_cudnn_lstm)
if pretrained is not None:
    siamese.load_pretrained_weights(model_wieghts_path=pretrained)
validation_data = ([dev_a_vectors, dev_b_vectors], dev_gold)
t1 = time.time()
siamese.fit(train_a_vectors, train_b_vectors, train_gold, validation_data, epochs=epochs)
t2 = time.time()
print('Took %f seconds' % (t2 - t1))
if save_path is not None:
    siamese.save(model_folder=save_path)

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

print('Testing the model ...')
# Don't rely on evaluate method
#result = siamese.evaluate(test_a_vectors, test_b_vectors, test_gold, 4906)
#print(result)


y = siamese.predict(test_a_vectors, test_b_vectors)
y = [i[0] for i in y]
assert len(test_gold) == len(y)

mse = mean_squared_error(test_gold, y)
print('MSE = %.2f' % mse)

pearsonr = stats.pearsonr(test_gold, y)
print('Pearson correlation (r) = %.2f' % pearsonr[0])

spearmanr = stats.spearmanr(test_gold, y)
print('Spearman’s p = %.2f' % spearmanr.correlation)