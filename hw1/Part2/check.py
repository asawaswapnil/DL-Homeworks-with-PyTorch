import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple
from IPython.display import Image
from torch.utils.data import Dataset, DataLoader
from mydataloader import loader
import pandas as pd
modelname="./try10/"
f = open('testxpad.pckl', 'rb')
testx= pickle.load(f)
f.close()
context_length=12
test_data = loader(data=testx, file='testy.csv', k=context_length, test=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle= False, num_workers = 4, pin_memory=True)

cuda = torch.cuda.is_available()
print("cuda",cuda)
device = torch.device("cuda" if cuda else "cpu")

def testout(model, loader,filename):
	model.eval()
	model.to(device)
	final=np.empty(shape=(0,0))
	for data in loader: 
		data = data.to(device)
		X = Variable(data.view(-1, (context_length*2+1)*40))
		out = model(X.float())
		pred = out.data.max(1, keepdim=True)[1]
		final=np.append(final,pred.cpu().numpy())
	print("hi")
	pd.DataFrame(data=final).to_csv(filename,header=False,  index=True)
class speechModel(nn.Module):
    def __init__(self):
        super(speechModel, self).__init__()
        self.fc1 = nn.Linear(1000, 1024)
        self.bnorm1 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.bnorm2 = nn.BatchNorm1d(1024)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.bnorm3 = nn.BatchNorm1d(512)
        self.dp3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(512, 512)
        self.bnorm4 = nn.BatchNorm1d(512)
        self.dp4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(512, 138)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dp1(self.bnorm1(x))
        x = F.relu(self.fc2(x))
        x = self.dp2(self.bnorm2(x))
        x = F.relu(self.fc3(x))
        x = self.dp3(self.bnorm3(x))
        x = F.relu(self.fc4(x))
        x = self.dp4(self.bnorm4(x))
        x = F.log_softmax(self.fc5(x))
        return x
model = speechModel()
model.load_state_dict(torch.load(modelname+'xavier_model.pt'))
testout(model, test_loader,modelname+'test_try105.csv')
