# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:11:48 2018

@author: mwahdan
"""

from gensim.models import KeyedVectors
import numpy as np
from nltk.corpus import stopwords

class WordEmbeddings:
    
    def __init__(self, word_embeddings_file_path, unknowns_strategy='random'):
        """
        Parameters:
            word_embeddings_file_path: 
                absolute path of .bin word2vec file.
            unknowns_strategy: 
                the strategy used to handle unknown tokens.
                It may take 2 values:
                    - random: random initialization for unknowns with uniform distribution.
                    - zeros: 300 dimensions zero numpy array.
                    - none: return None
        """
        
        self.stopwords = set(stopwords.words('english'))
        # Creating the model
        self.en_model = KeyedVectors.load_word2vec_format(word_embeddings_file_path, binary=True)
        
        # Printing out number of tokens available
        print("Number of Tokens: {}".format(len(self.en_model.vocab)))

        # Printing out the dimension of a word vector 
        print("Dimension of a word vector: {}".format(len(self.en_model['car'])))
        if unknowns_strategy == 'random':
            # define the random seed to reproduce the results
            np.random.seed(7)
            self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")
        elif unknowns_strategy == 'zeros':
            self.unknowns = np.zeros(len(self.en_model['car']))
        else:
            self.unknowns = None

    
    def get_vector(self, word):
        
        if word in self.en_model.vocab:
            return self.en_model.get_vector(word)
        elif word.lower() in self.en_model.vocab:
            return self.en_model.get_vector(word.lower())
        elif word.isdigit():
            # if number, then return embedding of any number
            return self.en_model.get_vector('zero')
        elif word in self.stopwords:
            return None
        else:
            #if the word doesn't exist in word embedding dictionary returns unknown representation
            return self.unknowns
        