# -*- coding: utf-8 -*-
"""
Created on Thu May 24 23:27:45 2018

@author: mwahdan
"""

import string

class Vectorizer:
    
    def __init__(self, word_embeddings, tokenizer):
        self.word_embeddings = word_embeddings
        self.tokenizer = tokenizer
        
    def vectorize_sentence(self, sentence, threshold=-1):
        tokens = self.tokenizer.tokenize(sentence)
        if threshold > 0:
            # truncate answers to threshold tokens.
            tokens = tokens[:threshold]
        vector = []
        for token in tokens:
            if self.__valid_token(token):
                token = self.__normalize(token)
                token_vector = self.word_embeddings.get_vector(token)
                if token_vector is not None:
                    vector.append(token_vector)
        return vector
    
    def vectorize_sentences(self, sentences, threshold=-1):
        return [self.vectorize_sentence(s) for s in sentences]
    
    def vectorize_df(self, df):
        a_vectors = [self.vectorize_sentence(sentence) for sentence in df['sentence_A']]
        b_vectors = [self.vectorize_sentence(sentence) for sentence in df['sentence_B']]
        gold = df['relatedness_score'].tolist()
        ids = 0 * [len(gold)]
        if 'pair_ID' in df.columns:
            ids = df['pair_ID']
        return ids, a_vectors, b_vectors, gold
    
    def __valid_token(self, token):
        if token in string.punctuation:
            return False
        return True
    
    def __normalize(self, token):
        if token == "'s":
            token = 'is' # may be 'has' also
        elif token == "'re":
            token = 'are'
        elif token == "'t":
            token = 'not'
        elif token == "'m":
            token = 'am'
        elif token == "'d":
            token = 'would' # may be 'had' also
        
        return token