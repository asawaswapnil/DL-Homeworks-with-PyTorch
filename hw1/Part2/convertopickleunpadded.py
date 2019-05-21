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
loader = WSJ()
trainX, trainY = loader.train
valX,valY=loader.dev
test2 = np.load('data/test.npy')
trainy=np.transpose(np.array(trainY))
valy=np.transpose(np.array(valY))

f = open('trainyup.pckl', 'wb')
pickle.dump(trainy, f)
f.close()
f = open('valyup.pckl', 'wb')
pickle.dump(valy, f)
f.close()

f = open('trainyup.pckl', 'rb')
trainy= pickle.load(f)
f.close()
print(trainy[1].shape,trainy[1])

f = open('trainypad.pckl', 'rb')
trainy= pickle.load(f)
f.close()
print(trainy[1].shape,trainy[1])
