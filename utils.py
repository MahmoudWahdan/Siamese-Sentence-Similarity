# -*- coding: utf-8 -*-
"""
Created on Sat May  5 00:04:37 2018

@author: mwahdan
"""

import json
from keras.preprocessing.sequence import pad_sequences

def read_data(file_path):
    samples = []
    with open(file_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def write_data(file_path, samples):
    with open(file_path, 'w') as outfile:
        for sample in samples:
            line = json.dumps(sample)
            outfile.write(line)
            outfile.write("\n")


def pad_tensor(tensor, max_len, dtype='float32'):
    return pad_sequences(tensor, padding='post', dtype=dtype, maxlen=max_len)


if __name__ == '__main__':
    tensor = [[[1,2,3],[4,5,6],[7,8,9]],
              [[1,2,3]]]
    res = list(pad_tensor(tensor, 2))
    print(res)