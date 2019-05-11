# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:58:47 2018

@author: mwahdan
"""

import tensorflow.python.keras.backend as K

def pearson_correlation(y_true, y_pred):
    # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
    fs_pred = y_pred - K.mean(y_pred)
    fs_true = y_true - K.mean(y_true)
    covariance = K.mean(fs_true * fs_pred)
    
    stdv_true = K.std(y_true)
    stdv_pred = K.std(y_pred)
    
    return covariance / (stdv_true * stdv_pred)

def negative_pearson_correlation(y_true, y_pred):
    return -1 * pearson_correlation(y_true, y_pred)
    