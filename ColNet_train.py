import torch
import numpy as np
from torch import autograd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import os
import random
import ColNet
from sklearn.linear_model import LinearRegression
import math
import argparse

torch.set_default_tensor_type(torch.DoubleTensor)
def encode(seq):
    encoded_seq = np.zeros(len(seq)*21,int)
    for j in range(len(seq)):
        if seq[j] == 'H':
            encoded_seq[j*21] = 1
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'D':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 1
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'R':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 1
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'F':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 1
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'A':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 1
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0
        elif seq[j] == 'C':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 1
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'G':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 1
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'Q':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 1
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'E':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 1
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'K':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 1
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'L':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 1
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'M':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 1
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'N':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 1
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'S':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 1
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'Y':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 1
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0
        elif seq[j] == 'T':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 1
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'I':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 1
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'W':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 1
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'P':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 1
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'V':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 1
            encoded_seq[j*21+20] = 0
        elif seq[j] == 'O':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 1
        else:
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0
    encoded_seq = encoded_seq.reshape(len(seq),21)
    return encoded_seq

def get_input(seqs,seq_lenth):
    data = np.zeros((1,seq_lenth,21))
    count = 0
    for i in seqs:
        count += 1
        if len(i) <= seq_lenth:
            i = i + (seq_lenth-len(i))*"X"
        if len(i) > seq_lenth:
            i = i[0:64]
        if count == 1:
            single_seq = encode(i)
            single_seq = np.expand_dims(single_seq, axis=0)
            data = data + single_seq
            data = np.expand_dims(data, axis=0)
        if count != 1:
            single_seq = encode(i)
            single_seq = np.expand_dims(single_seq, axis=0)
            single_seq = np.expand_dims(single_seq, axis=0)
            data = np.concatenate ((data,single_seq),axis=0)
    return data

f = open(r'all_seqs.txt')
content = f.readlines()
f.close()
seqs = []
tm = []
for i in content:
    i = i.split()
    seqs.append(i[0])
    tm.append(float(i[1]))

dataset = list (zip(seqs,tm))
amax = max(tm)
amin = min(tm)
f = open('max_min','w')
f.write('amax %s\n'%amax)
f.write('amin %s'%amin)
f.close()

def get_tm(tm):
    data = np.zeros((1,1))
    list = []
    for i in tm:
        #list.append(float(i))
        #list.append(math.log2(float(i)))
        i = np.cbrt(float(i))
        list.append(float(i))
    count = 0
    for i in list:
        count += 1
        a = float(i)
        #a = (a - amax) / (amax - amin) 
        if count == 1:
            data = data + np.array([[[float("%s"%a)]]])
            data = np.expand_dims(data, axis=0)
        if count != 1:
            data = np.concatenate((data,np.array([[[[float("%s"%a)]]]])),axis=0)
    data = data.reshape(np.shape(data)[0],1,1,1)

    return data 
    
def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
dataset = shuffle_dataset(dataset, 1234)

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2
dataset_train, dataset_test = split_dataset(dataset, 0.8)
trainset_seq = []
trainset_tm = []
for i in dataset_train:
    trainset_seq.append(i[0])
    trainset_tm.append(i[1])
testset_seq = []
testset_tm = []
for i in dataset_test:
    testset_seq.append(i[0])
    testset_tm.append(i[1])
trainseq = get_input(seqs = trainset_seq,seq_lenth=64)
testseq = get_input(seqs = testset_seq,seq_lenth=64)
traintm = get_tm(tm = trainset_tm)
testtm = get_tm(tm = testset_tm)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.len = len(x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
dataset = MyDataset(x=trainseq, y=traintm)
testset = MyDataset(x=testseq, y=testtm)
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, drop_last= True)
testloader = DataLoader(dataset=testset, batch_size=127, drop_last= True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
p = ColNet.predictor().to(device)
optimizer = torch.optim.Adam(p.parameters(),lr=1e-4,weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()
train_loss = []

f = open('epoch_loss.txt','w')
f.close()
f = open('predict.txt','w')
f.close()
f = open('predictValue.txt','w')
f.close()
model = LinearRegression()
if not os.path.exists('saved_models'):
    os.mkdir('saved_models')
    
parser = argparse.ArgumentParser()
parser.add_argument('-bz', type=int, default=8)
parser.add_argument('-test_len', type=int, default=127)
parser.add_argument('-epochs', type=int, default=200)

args = parser.parse_args()
bz = args.bz
test_len = args.test_len
epochs = args.epochs

batch_size = bz
batch_size_test = test_len
for epoch in range (epochs):
    epoch_loss = 0

    count = 0
    for i, data in enumerate (dataloader):
        count += 1
        x,y = data
        x = x.reshape(batch_size,1,21,64)

        x, y = x.to(device), y.to(device)

        output = p(x,batch_size)
        #print("output",output)

        y = y.reshape(batch_size,1)
        #print("y",y)

        p_loss = loss_fn(output,y)
        optimizer.zero_grad()

        epoch_loss += float(p_loss)
        p_loss.backward()
        optimizer.step()
    f = open('epoch_loss.txt','a+')
    loss_ave = 'epoch_loss %s\n'%(epoch_loss/count)
    f.write(loss_ave)
    f.close()
    print(epoch)
    if (epoch+1) % 3 == 0:
        state_dict = {"pred": p.state_dict(), "optim": optimizer.state_dict(),"epoch": epoch}
        torch.save(state_dict, r'saved_models/epoch_%s.pth'%(str(epoch)))

    with torch.no_grad():
        if epoch > 0:
            cof = 0
            for i, data in enumerate (testloader):
                x, y = data
                x = x.reshape(batch_size_test,1,21,64)
                x = x.to(device)
                output = p(x,batch_size_test)
                output = output.cpu().detach().numpy()
                a = output.reshape(-1,1)
                y = y.cpu().detach().numpy()
                b = y.reshape(-1,1)
                model.fit(a,b)
                score = model.score(a,b)
                cof += float(score)

                #cof = math.sqrt(cof)
                print("cof",cof)
                f = open('predict.txt','a+')
                f.write('epoch %s %s\n'%(epoch,cof))
                f.close()
                #if epoch > 50:
                #    f = open('predictValue.txt','a+')
                #    f.write('epoch %s\nout_x\n%s\nout_y\n%s'%(epoch,a,b))
                #    f.close()
