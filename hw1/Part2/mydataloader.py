
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
# f = open('valxpad.pckl', 'rb')
# valx= pickle.load(f)
# f.close()
f = open('testxup.pckl', 'rb')
testx= pickle.load(f)
f.close()
def usecsv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return(your_list)

class loader(Dataset):
    def __init__(self, data, file, k,test=False):
        self.data = data
        self.file = usecsv(file)
        self.width   = k
        self.test = test
    def __getitem__(self, index):
        one_line_content = self.file[index]
        i=int(self.file[index][0])
        j=int(self.file[index][1])
        if(self.test):loader
            return self.data[i][j:j+2*self.width+1]
        else:
            label = int(self.file[index][2])
            #print("dataset",index,i,j,label)
            return self.data[i][j:j+2*self.width+1], label
    def __len__(self):
        return len(self.file)
context_length=12
print(valx[0].shape)
val_dataset = loader(data=valx,file='valy.csv',k=context_length)
test_dataset = loader(data=testx,file='testy.csv',k=context_length)

frame, label = val_dataset.__getitem__(267)
val_dataloader=DataLoader(val_dataset,shuffle=False, batch_size=2)
for batch_idx, (data, target) in enumerate(val_dataloader):
    if batch_idx==100:
        print(data,target, data.shape,target.shape)
