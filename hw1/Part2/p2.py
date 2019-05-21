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
from wsj_loader import WSJ
import pickle
import os
import pandas as pd
# import matplotlib.pyplot as plt
# import time
import time


# class loader(Data.Dataset):
#     def __init__(self, data, list_file, context_length, test_mode=False):
#         self.data = data
#         self.list_file = open(list_file).readlines()
#         self.w = context_length//2
#         self.test = test_mode




# import csv
# import random
# b=[(x-1,random.randint(1,139)) for x in  range(169657)]

# np.savetxt('s2.csv', b, delimiter=',', fmt='%d')
loader = WSJ()
trainX, trainY = loader.train
valX,valY=loader.dev
test2 = np.load('data/test.npy')
print("test22",test2[1].shape)
print("test2",test2.shape)
testx=[]
for i in range(len(test2)):
    for j in range(len(test2[i])):
        testx.append(test2[i][j])  
print("listflattedned",len(testx))
testx=np.array(testx)
trainx=[]
trainy=[]
valx=[]
valy=[]

for i in range(len(trainX[1:20])):
    for j in range(len(trainX[i])):
        trainx.append(trainX[i][j])
        trainy.append(trainY[i][j])
for i in range(len(valX)):
    for j in range(len(valY[i])):
        valx.append(valX[i][j])
        valy.append(valY[i][j])
trainx=np.array(trainx)
trainy=np.array(trainy)
valx=np.array(valx)
valy=np.array(valy)


# trainx=trainX[0]
# trainy=trainY[0]
# valx=valX[0]
# valy=valY[0]

# print(trainx,trainx.shape)
trainx = torch.from_numpy(trainx).float()
valx = torch.from_numpy(valx).float()
trainy = torch.from_numpy(trainy)
valy = torch.from_numpy(valy)
testx=torch.from_numpy(testx)

# f = open('trainx.pckl', 'wb')
# pickle.dump(trainx, f)
# f.close()
# f = open('valx.pckl', 'wb')
# pickle.dump(valx, f)
# f.close()
# f = open('trainy.pckl', 'wb')
# pickle.dump(trainy, f)
# f.close()
# f = open('valy.pckl', 'wb')
# pickle.dump(valy, f)
# f.close()
# f = open('trainx.pckl', 'rb')
# trainx= pickle.load(f)
# f.close()
# f = open('valx.pckl', 'rb')
# valx= pickle.load(f)
# f.close()
# f = open('trainy.pckl', 'rb')
# trainy= pickle.load(f)
# f.close()
# f = open('valy.pckl', 'rb')
# valy= pickle.load(f)
# f.close()
# print(dim(trainx),dim(trainy))

# dataloader_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
# train_loader = dataloader.DataLoader(trainx, **dataloader_args) 
# val_loader = dataloader.DataLoader(val, **dataloader_args)


def training_routine(net,dataset,n_iters,gpu):
    # organize the data
    train_data,train_labels,val_data,val_labels = dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    
    # use the flag
    if gpu:
        train_data,train_labels = train_data.cuda(),train_labels.cuda()
        val_data,val_labels = val_data.cuda(),val_labels.cuda()
        net = net.cuda() # the network parameters also need to be on the gpu !
        print("Using GPU")
    else:
        print("Using CPU")
    for i in range(n_iters):
        # forward pass
        train_output = net(train_data)
        train_loss = criterion(train_output,train_labels)
        # backward pass and optimization
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Once every 100 iterations, print statistics
        if i%100==0:
            print("At iteration",i)
            # compute the accuracy of the prediction
            train_prediction = train_output.cpu().detach().argmax(dim=1)
            train_accuracy = (train_prediction.numpy()==train_labels.numpy()).mean() 
            # Now for the validation set
            val_output = net(val_data)
            val_loss = criterion(val_output,val_labels)
            # compute the accuracy of the prediction
            val_prediction = val_output.cpu().detach().argmax(dim=1)
            val_accuracy = (val_prediction.numpy()==val_labels.numpy()).mean() 
            print("Training loss :",train_loss.cpu().detach().numpy())
            print("Training accuracy :",train_accuracy)
            print("validation loss :",val_loss.cpu().detach().numpy())
            print("validation accuracy :",val_accuracy)
    testout=net(testx)
    testpredict=torch.argmax(testout,dim=1)
    print(testpredict[0:1000])
    # testbind=torch.cat((testpredict.index(:), testpredict), 0)
    testbind=[[i,int(testpredict[i].item())] for i in range(testpredict.size()[0])]
    #testbind.insert(0,['id','label'])
    print(type(testbind[0]))
    print(testbind[0:1000])
    pd.DataFrame(data=np.array(testbind),columns=['id','label']).to_csv('s2.csv',sep=',')
    net = net.cpu()
 
 
dataset = trainx,trainy,valx,valy
gpu = False
gpu = gpu and torch.cuda.is_available() # to know if you actually can use the GPU

def generate_single_hidden_MLP(n_hidden_neurons):
    return nn.Sequential(nn.Linear(40,n_hidden_neurons),nn.ReLU(),nn.Linear(n_hidden_neurons,138),nn.Softmax())
model1 = generate_single_hidden_MLP(300)

training_routine(model1,dataset,1000,gpu)
# class Simple_MLP(nn.Module):
#     def __init__(self, size_list,k=12):
#         super(Simple_MLP, self).__init__()
#         layers = []
#         self.size_list = [(k*2+1),1024,1024,512,512,10]
#         for i in range(len(size_list) - 2):
#             layers.append(nn.Linear(size_list[i],size_list[i+1]))
#             layers.append(nn.ReLU())
#         layers.append(nn.Linear(size_list[-2], size_list[-1]))
#         self.net = nn.Sequential(*layers)
#         self.

#     def forward(self, x):
#         x = x.view(-1, self.size_list[0]) # Flatten the input
#         return self.net(x)




# def train_multilayer_epoch(model, train_loader, criterion, optimizer):
#     model.train()
#     model.to(device)

#     running_loss = 0.0
    
#     start_time = time.time()
#     for batch_idx, (data, target) in enumerate(train_loader):   
#         optimizer.zero_grad()   
#         data = data.to(device)
#         target = target.long().to(device)

#         outputs = model(data)
#         loss = criterion(outputs, target)
#         running_loss += loss.item()

#         loss.backward()
#         optimizer.step()
    
#     end_time = time.time()
    
#     running_loss /= len(train_loader)
#     print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
#     return running_loss

# def test_model(model, test_loader, criterion):
#     with torch.no_grad():
#         model.eval()
#         model.to(device)

#         running_loss = 0.0
#         total_predictions = 0.0
#         correct_predictions = 0.0

#         for batch_idx, (data, target) in enumerate(test_loader):   
#             data = data.to(device)
#             target = target.long().to(device)

#             outputs = model(data)

#             _, predicted = torch.max(outputs.data, 1)
#             total_predictions += target.size(0)
#             correct_predictions += (predicted == target).sum().item()

#             loss = criterion(outputs, target).detach()
#             running_loss += loss.item()


#         running_loss /= len(test_loader)
#         acc = (correct_predictions/total_predictions)*100.0
#         print('Testing Loss: ', running_loss)
#         print('Testing Accuracy: ', acc, '%')
#         return running_loss, acc
# n_epochs = 10
# Train_loss = []
# Test_loss = []
# Test_acc = []
# model2 = Simple_MLP(6,k=12)
# criterion = nn.CrossEntropyLoss()
# print(model2)
# optimizer = optim.Adam(model2.parameters(),lr=0.001)
# device = torch.device("cuda" if cuda else "cpu")

# for i in range(n_epochs):
#     train_loss = train_multilayer_epoch(model2, train_loader, criterion, optimizer)
#     test_loss, test_acc = test_model(model2, test_loader, criterion)
#     Train_loss.append(train_loss)
#     Test_loss.append(test_loss)
#     Test_acc.append(test_acc)

#     print('='*20)