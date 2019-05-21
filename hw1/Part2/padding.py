import numpy as np
import torchvision
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
# from wsj_loader import WSJ
import pickle
import os
import pandas as pd
# import matplotlib.pyplot as plt
# import time
import time

def paddata(data):
	pad=np.zeros((12,40))
	for i in range(len(data)):
		data[i]=np.concatenate((pad,data[i]),axis=0)
		data[i]=np.concatenate((data[i],pad),axis=0)
	return data
def padtest(data):
	pad=np.zeros((12,1))
	for i in range(len(data)):
		data[i]=data[i].reshape((len(data[i]),1))
	return data

path="./data"
# print('./data/train.npy')
# trainX=np.load('./data/dev.npy', encoding='bytes')
# trainY =  np.load('./data/train_labels.npy', encoding='bytes')
# valX=np.load('./data/dev.npy', encoding='bytes')
# valY=np.load('./data/dev_labels.npy', encoding='bytes')
testX =np.load('data/test.npy')

print("1000000")
# trainxpad=paddata(trainX)
# trainypad=padtest(trainY)
# valxpad=paddata(valX)
# valypad=padtest(valY)
testxpad=padtest(testX)


# f = open('trainxpad.pckl', 'wb')
# pickle.dump(trainxpad, f)
# f.close()
# f = open('trainyup.pckl', 'wb')
# pickle.dump(trainypad, f)
# f.close()
# f = open('valxpad15.pckl', 'wb')
# pickle.dump(valxpad, f)
# f.close()
# f = open('valyup.pckl', 'wb')
# pickle.dump(valypad, f)
# f.close()
# f = open('trainyup.pckl', 'rb')
# valyup=pickle.load(f)
# f.close()
# f = open('valyup.pckl', 'rb')
# trainyup=pickle.load(f)
# f.close()
# print(valyup[0])
print(testxpad)
f = open('testxup.pckl', 'wb')
pickle.dump(testxpad, f)

f.close()

# # # f = open('trainxpad.pckl', 'rb')
# # trainxpad= pickle.load(f)
# # # f.close()
# f = open('valxpad.pckl15', 'rb')
# valxpad= pickle.load(f)
# f.close()
# f = open('trainypad.pckl', 'rb')
# trainypad= pickle.load(f)
# f.close()
# f = open('valypad.pckl', 'rb')
# valypad= pickle.load(f)
# f.close()
# f = open('testxpad.pckl', 'rb')
# testxpad= pickle.load(f)
# f.close()

#print( testxpad[1].shape)
# print(dim(trainxpad),dim(trainypad))

