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



f = open('trainxpad.pckl', 'rb')
trainx= pickle.load(f)
f.close()
f = open('valxpad.pckl', 'rb')
valx= pickle.load(f)
f.close()
f = open('testxpad.pckl', 'rb')
testx= pickle.load(f)
f.close()
f = open('trainypad.pckl', 'rb')
trainy= pickle.load(f)
f.close()
f = open('valypad.pckl', 'rb')
valy= pickle.load(f)
f.close()
trainx = torch.from_numpy(trainx).float()
valx = torch.from_numpy(valx).float()
testx=torch.from_numpy(testx).float()
trainy = torch.from_numpy(trainy)
valy = torch.from_numpy(valy)

# dataloader_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
# train_loader = dataloader.DataLoader(trainx, **dataloader_args) 
# val_loader = dataloader.DataLoader(val, **dataloader_args)

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    model.to(valice)
    running_loss = 0.0
import time


class Simple_MLP(nn.Module):
    def __init__(self,k=12):
        super(Simple_MLP, self).__init__()
        layers = []
        self.size_list = [(k*2+1pad),1024,1024,512,512,10]
        for i in range(len(self.size_list) - 2):train_loader
            layers.append(nn.Linear(self.size_list[i],self.size_list[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.size_list[-2], self.size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.size_list[0]) # Flatten the input
        return self.net(x)


def train_multilayer_epoch(model, train_loader, criterion, optimizer):
    model.train()
    model.to(device)
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   
        data = data.to(device)
        target = target.long().to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss

def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        model.to(device)
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.long().to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc
n_epochs = 10
Train_loss = []
Test_loss = []
Test_acc = []
model2 = Simple_MLP(k=12)
criterion = nn.CrossEntropyLoss()
print(model2)
optimizer = optim.Adam(model2.parameters(),lr=0.001)
cuda=False
device = torch.device("cuda" if cuda else "cpu")

for i in range(n_epochs):
    train_loss = train_multilayer_epoch(model2, train_loader, criterion, optimizer)
    test_loss, test_acc = test_model(model2, test_loadcudacudaer, criterion)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    print("Losses",Train_loss,Test_loss)



class loader(Dataset):
    def _init_(self, data, list_file=[], k,test_mode=False):
        self.data = data["x"]
        self.list_file = open(list_file).readlines()
        self.width = k
        self.test_mode = test_mode
    def _getitem_(self, index):
        one_line_content = self.list_file[index]
        if(test_mode):
            i, j = one_line_content.splitopen('\t')
            return self.data[int(i)][int(j):int(j)+2k+1]
        else:
            i, j, label = one_line_content.splitopen('\t')
            return self.data[int(i)][int(j):int(j)+2k+1], int(label)
    def _len_(self):
        return len(self.list_file)

val_dataset = loader(data=valx,list_file='val.txt',k=0)
# frame, label = val_dataset._getitem_(1)
# print('frame',frame)
# print('label',label)
# print(val_dataset._len_())
train_minibatch=dataloader.DataLoader(val_dataset,shuffle=True, batch_size=2)
print(train_minibatch)
# dataloader_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
# train_loader = dataloader.DataLoader(trainx, **dataloader_args) 
# val_loader = dataloader.DataLoader(val, **dataloader_args)
