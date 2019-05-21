#! /usr/bin/env python3

import numpy as np
import torchvision
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
import pickle
import os
import pandas as pd
import time
import csv



def gentxt(data,filename):
    line=[]
    for i in range(len(data)):
        for j in range(len(data[i])):
            line.append([i,j,int(data[i][j][0])])
    df_cv = pd.DataFrame(data=line).to_csv(filename,header=False,index=False)

def gentxttest(data,filename):
    line=[]
    for i in range(len(data)):
        for j in range(len(data[i])):
            i=int(self.file[index][0])
            j=int(self.file[index][1])
            line.append([i,j])
    df_cv = pd.DataFrame(data=line).to_csv(filename,header=False,index=False)

def usecsv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    print(your_list)

f = open('testyup.pckl', 'rb')
testx= pickle.load(f)
f.close()
gentxt(testx,'testy.csv')
# usecsv('vale.csv')
# ==gentxt(valy)
# usecsv('vale.csv')
