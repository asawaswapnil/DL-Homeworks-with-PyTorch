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

np.random.seed(1000)


cuda = torch.cuda.is_available()
print("cuda",cuda)
device = torch.device("cuda" if cuda else "cpu")
Metric = namedtuple('Metric', ['loss', 'train_error', 'val_error'])


modelname="./try10/"
context_length = 12
f = open('trainxpad.pckl', 'rb')
trainx= pickle.load(f)
f.close()
f = open('valxpad.pckl', 'rb')
valx= pickle.load(f)
f.close()
f = open('testxpad.pckl', 'rb')
testx= pickle.load(f)
f.close()
train_data = loader(data=trainx, file='trainy.csv', k=context_length)
val_data = loader(data=valx, file='valy.csv', k=context_length)
test_data = loader(data=testx, file='testy.csv', k=context_length, test=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle= True, num_workers = 4, drop_last = True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle= True, num_workers = 4, drop_last = True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle= False, num_workers = 4, pin_memory=True)


def inference(model, loader):
    correct = 0
    model.eval()
    model.to(device)
    for data, label in loader:    
        data = data.to(device)
        label = label.long().to(device)
        X = Variable(data.view(-1, (context_length*2+1)*40))
        Y = Variable(label)
        out = model(X.float())
        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
        members=+len(predicted)
    correct=correct/members
    return correct.cpu().numpy() 

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
    pd.DataFrame(data=final).to_csv(filename,header=False,  index=True)

class Trainer():
    
    def __init__(self, model, optimizer, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def run(self, epochs):
        print("Start Training...")
        self.model.train()
        self.model.to(device)
        self.metrics = []
        train_size=0
        for e in range(n_epochs):
            epoch_loss = 0
            correct = 0
            for batch_idx, (data, label) in enumerate(train_loader):
                data = data.to(device)
                label = label.long().to(device)
                self.optimizer.zero_grad()
                X = Variable(data.view(-1, (context_length*2+1)*40))
                Y = Variable(label)
                out = self.model(X.float())
                pred = out.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                correct += predicted.sum()
                loss = F.nll_loss(out, Y)
                loss.backward()
                self.optimizer.step()
                if(batch_idx%100==0):
                    print(batch_idx,loss.data)
                if(batch_idx%10000==0):
                    path=modelname+str(e+5)+str(batch_idx)+".pt"
                    self.save_model(path)
                    torch.save(self.optimizer.state_dict(), path)
                epoch_loss += loss.data
                train_size+=1
            epoch_loss=epoch_loss.cpu()
            total_loss = epoch_loss.numpy()
            train_error = 1.0 - correct.cpu().numpy()/train_size
            val_error = 1.0 - inference(self.model,val_loader)
            self.metrics.append(Metric(loss=total_loss, 
                                  train_error=train_error,
                                  val_error=val_error))
            print(self.metrics)
        testout(self.model, test_loader,modelname+'test_try103.csv')

def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)
### VISUALIZATION ###
def training_plot(metrics):
    plt.figure(1)
    plt.plot([m.loss for m in metrics], 'b')
    plt.title('Training Loss')
    plt.savefig(img_file)

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

print(speechModel())
model = speechModel()
model.load_state_dict(torch.load(modelname+'4100000.pt'))
### TRAIN MODELS WITH BATCHRM AND DROPOUT ###
n_epochs = 10
print("XAVIER INIT WEIGHTS")
AdamOptimizer = torch.optim.Adam(model.parameters(), lr=0.001)
xavier_trainer = Trainer(model, AdamOptimizer)
xavier_trainer.run(n_epochs)
xavier_trainer.save_model(modelname+'xavier_model.pt')
print('')