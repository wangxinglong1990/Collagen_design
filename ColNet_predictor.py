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
import sequence_encode
import argparse

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, default='seqs.txt')
    parser.add_argument('-model', type=str, default='ColNet_model.pth')
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-output', type=str, default='outputTM.txt')
    
    args = parser.parse_args()
    file = args.file
    model = args.model
    device = args.device
    output = args.output
    
    f = open(file,'r')
    content = f.readlines()
    f.close()
    seqs = []
    
    c = 0
    for i in content:
        c += 1
        if c % 2 == 0:
            i = i.split()
            seqs.append(i[0])
    
    testseq = sequence_encode.get_input(seqs = seqs,seq_lenth=64)
    model_path = model
    device = torch.device(device)
    load_model = torch.load(model_path, map_location=device)
    tm_predictor = ColNet.predictor().to(device)
    tm_predictor.load_state_dict(load_model['pred'])
    
    f = open(output,'w')
    f.close()
    f = open(output,'a+')
    for i in testseq:
        c += 1
        i = i.reshape(1,1,21,64)
        i = torch.from_numpy(i)
        i = i.to(device)
        predict_tm = tm_predictor(i,1).cpu().detach().numpy()
        predict_tm = predict_tm**3
        f.write(str(predict_tm))
        f.write('\n')
    f.close()


