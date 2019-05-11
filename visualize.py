# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:34:01 2019

@author: mwahdan
"""

import argparse
import time

from word_embeddings import WordEmbeddings
from tokenizer import Tokenizer
from vectorizer import Vectorizer
from utils import pad_tensor
from siamese import SiameseModel


# read command-line parameters
parser = argparse.ArgumentParser('Testing the model with SICK-like data')
parser.add_argument('--model', '-p', help = 'Path to trained model.', type = str, required = True)
parser.add_argument('--word2vec', '-w', help = 'Path to word2vec .bin file with 300 dims.', type = str, required = True)


args = parser.parse_args()
model_path = args.model
word_embeddings_file_path = args.word2vec


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


print('Loading the model ...')
siamese = SiameseModel(False)
siamese.load(model_path)


sentences =['There is no man pointing at a car',
            'The woman is not playing the flute',
            'The man is not riding a horse',
            'A man is pointing at a slivar sedan',
            'The woman is playing the flute',
            'A man is riding a horse']
vectors = vectorizer.vectorize_sentences(sentences)
vectors = pad_tensor(vectors, None)

print('Visualizing LSTM activations ...')
siamese.visualize_activation(vectors)
siamese.visualize_specific_activation(vectors, 0)
siamese.visualize_specific_activation(vectors, 5)
siamese.visualize_specific_activation(vectors, 49)