# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:07:03 2018

@author: mwahdan
"""

from word_embeddings import WordEmbeddings
from tokenizer import Tokenizer
from vectorizer import Vectorizer
from data_reader import read_SEMEVAL_data
from utils import pad_tensor
from siamese import SiameseModel
import time
import argparse

# read command-line parameters
parser = argparse.ArgumentParser('Pretraining the model with SemEval-like data')
parser.add_argument('--word2vec', '-w', help = 'Path to word2vec .bin file with 300 dims.', type = str, required = True)
parser.add_argument('--data', '-d', help = 'Path to SemEval data used for pre-training.', type = str, required = True)
parser.add_argument('--epochs', '-e', help = 'Number of epochs.', type = int, default = 100, required = False)
parser.add_argument('--save', '-s', help = 'Weights file path to save the pretrained model.', type = str, required = True)
#print(parser.format_help())

args = parser.parse_args()
word_embeddings_file_path = args.word2vec
pretrained_weights_file_path = args.save
epochs = args.epochs
df = read_SEMEVAL_data(args.data)

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
ids, train_a_vectors, train_b_vectors, train_gold = vectorizer.vectorize_df(df)
train_max_a_length = len(max(train_a_vectors, key=len))
train_max_b_length = len(max(train_b_vectors, key=len))
print('maximum number of tokens per sentence A in training set is %d' % train_max_a_length)
print('maximum number of tokens per sentence B in training set is %d' % train_max_b_length)
max_len = max([train_max_a_length, train_max_b_length])

# padding
train_a_vectors = pad_tensor(train_a_vectors, max_len)
train_b_vectors = pad_tensor(train_b_vectors, max_len)

print('Training the model ...')
siamese = SiameseModel()
validation_data = None
t1 = time.time()
siamese.fit(train_a_vectors, train_b_vectors, train_gold, validation_data, epochs=epochs)
t2 = time.time()
print('Took %f seconds' % (t2 - t1))
siamese.save_pretrained_weights(pretrained_weights_file_path)