# -*- coding: utf-8 -*-
"""
Created on Sat May  5 00:04:37 2018

@author: mwahdan
"""

import json
import argparse
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    tensor = [[[1,2,3],[4,5,6],[7,8,9]],
              [[1,2,3]]]
    res = list(pad_tensor(tensor, 2))
    print(res)