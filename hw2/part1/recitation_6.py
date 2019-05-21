#!/usr/bin/env python
# coding: utf-8

# # Recitation - 6
# ___
# 
# * Custom Dataset & DataLoader
# * Torchvision ImageFolder Dataset
# * Residual Block
# * CNN model with Residual Block
# * Loss Fucntions (Center Loss and Triplet Loss)

# ## Imports

# In[1]:


import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# class ImageDataset(Dataset):
#   def __init__(self, file_list, target_list):
#       self.file_list = file_list
#       self.target_list = target_list
#       self.n_class = len(list(set(target_list)))

#   def __len__(self):
#       return len(self.file_list)

#   def __getitem__(self, index):
#       img = Image.open(self.file_list[index])
#       img = torchvision.transforms.ToTensor()(img)
#       label = self.target_list[index]
#       return img, label


# # #### Parse the given directory to accumulate all the images

# # In[3]:


# def parse_data(datadir):
#   img_list = []
#   ID_list = []
#   for root, directories, filenames in os.walk(datadir):
#       #print("os walking root:",root,"dir:", directories,"fn",filenames)
#       for filename in filenames:
#           if filename.endswith('.jpg'):
#               filei = os.path.join(root, filename)
#               img_list.append(filei)
#               ID_list.append(root.split('/')[-1])
#   # print("ID_list",ID_list)
#   # print()
#   # print()
#   # construct a dictionary, where key and value correspond to ID and target
#   uniqueID_list = list(set(ID_list)) # list of unique classes you have
#   class_n = len(uniqueID_list) #no of unique classes
#   target_dict = dict(zip(uniqueID_list, range(class_n))) #Key: actual class name, value:new class number( ranging from 0 to length of unique clases)
#   # print("target_dict",target_dict)
#   # print()
#   # print()
#   label_list = [target_dict[ID_key] for ID_key in ID_list] # list of new class number
#   # print("label_list",label_list)
#   # print()
#   # print()
#   print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
#   return img_list, label_list, class_n


# # In[4]:


# img_list, label_list, class_n = parse_data('medium')
# # print(img_list)

# # In[5]:


# trainset = ImageDataset(img_list, label_list)


# # In[6]:


# train_data_item, train_data_label = trainset.__getitem__(0)
# print(train_data_item,train_data_label)
# print(train_data_item.size())
# # In[5]:

# # In[7]:


# print('data item shape: {}\t data item label: {}'.format(train_data_item.shape, train_data_label))


# # In[8]:


# dataloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1, drop_last=False)


# ## Torchvision DataSet and DataLoader

# In[9]:


imageFolder_dataset = torchvision.datasets.ImageFolder(root='train_data/medium', 
                                                       transform=torchvision.transforms.ToTensor())


# # In[10]:


imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=10, shuffle=True, num_workers=4)


# In[11]:


imageFolder_dataset.__len__(), len(imageFolder_dataset.classes)


# ## Residual Block

# In[12]:


class ResBlock(nn.Module):
    def __init__(self, channel_size, stride=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=channel_size, out_channels=channel_size, 
                                             kernel_size=3, stride=stride, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=channel_size),
                                   nn.ReLU(inplace=False),
                                   nn.Conv2d(in_channels=channel_size, out_channels=channel_size, 
                                             kernel_size=3, stride=stride, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=channel_size))
        self.logit_non_linear = nn.ReLU(inplace=False)

    def forward(self, x):
        output = x
        output = self.block(output)
        output = self.logit_non_linear(output + x)
        return output


# ## CNN Model with Residual Block 

# In[13]:


class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(Network, self).__init__()
        
        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        
        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes):
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx], 
                                         out_channels=self.hidden_sizes[idx+1], 
                                         kernel_size=3, stride=2, bias=False))
            self.layers.append(nn.ReLU(inplace=False))
            self.layers.append(ResBlock(channel_size=channel_size))
            
        self.layers = nn.Sequential(*self.layers)
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)

        self.linear_label = nn.Linear(feat_dim, self.hidden_sizes[-1], bias=False)
        
        # For creating the embedding to be passed into the Center Loss criterion
        self.relu_closs = nn.ReLU(inplace=False)
    
    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
            
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        closs_output = self.linear_closs(output)
        label_output = self.linear_label(closs_output)
        label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
        
        # Create the feature embedding for the Center Loss
        closs_output = self.relu_closs(closs_output)
        return closs_output, label_output

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


# ### Training & Testing Model

# In[14]:


def train(model, data_loader, test_loader, task='Classification'):
    model.train()

    for epoch in range(numEpochs):
        avg_loss = 0.0
        print(epoch)
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            print("b",batch_num)
            optimizer.zero_grad()
            outputs = model(feats)[1]

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        path=str(epoch)+".pt"
        torch.save(model.state_dict(), path)      
        if task == 'Classification':
            val_loss, val_acc = test_classify(model, test_loader)
            train_loss, train_acc = test_classify(model, data_loader)
            #pd.DataFrame(data=model).to_csv("",header=False,  index=True)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)


def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


def test_verify(model, test_loader):
    raise NotImplementedError


# # #### Dataset, DataLoader and Constant Declarations

# # In[15]:


train_dataset = torchvision.datasets.ImageFolder(root='train_data/medium',  
                                                 transform=torchvision.transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, 
                                               shuffle=True, num_workers=4)

dev_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium', transform=torchvision.transforms.ToTensor())
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=10, shuffle=True, num_workers=4)


# # In[16]:


numEpochs = 50
num_feats = 3

learningRate = 1e-2
weightDecay = 5e-5

hidden_sizes = [32, 64]
num_classes = len(train_dataset.classes)

device = torch.device('cuda')


# # In[17]:


network = Network(num_feats, hidden_sizes, num_classes)
network.apply(init_weights)
device
# criterion = CenterLoss()
# # optimizer = torch.optim.(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
optimizer =  torch.optim.Adam(network.parameters(),lr=0.001)


# # In[18]:


# network.train()
# network.to(device)
# train(network, train_dataloader, dev_dataloader)


# # In[ ]:





# # ## Center Loss
# # ___
# # The following piece of code for Center Loss has been pulled and modified based on the code from the GitHub Repo: https://github.com/KaiyangZhou/pytorch-center-loss
# #     
# # <b>Reference:</b>
# # <i>Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.</i>

# # In[19]:


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


# # In[20]:


def train_closs(model, data_loader, test_loader, task='Classification'):
    model.train()
    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            optimizer_label.zero_grad()
            optimizer_closs.zero_grad()
            feature, outputs = model(feats)
            l_loss = criterion_label(outputs, labels.long())
            c_loss = criterion_closs(feature, labels.long())
            loss = l_loss + closs_weight * c_loss           
            loss.backward()           
            optimizer_label.step()
            for param in criterion_closs.parameters():
                param.grad.data *= (1. / closs_weight)
            optimizer_closs.step()
            avg_loss += loss.item()
            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0    
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        path=str(epoch)+".pt"
        torch.save(model.state_dict(), path)      
        if task == 'Classification':
            val_loss, val_acc = test_classify_closs(model, test_loader)
            train_loss, train_acc = test_classify_closs(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)


def test_classify_closs(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        l_loss = criterion_label(outputs, labels.long())
        c_loss = criterion_closs(feature, labels.long())
        loss = l_loss + closs_weight * c_loss
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


# In[21]:


closs_weight = 1
lr_cent = 0.5
feat_dim = 64

network = Network(num_feats, hidden_sizes, num_classes, feat_dim)
network.apply(init_weights)

criterion_label = nn.CrossEntropyLoss()
criterion_closs = CenterLoss(num_classes, feat_dim, device)
optimizer_label =  torch.optim.Adam(network.parameters(), lr=0.001)
optimizer_closs = torch.optim.Adam(network.parameters(), lr=0.001)
scheduler_label = ReduceLROnPlateau(optimizer_label, 'min')
scheduler_closs = ReduceLROnPlateau(optimizer_closs, 'min')

# optimizer =  optim.Adam(network.parameters(),lr=0.001)

# In[22]:


network.train()
network.to(device)
# train(network, train_dataloader, dev_dataloader)
def findNremove(path,pattern,maxdepth=1):
    cpath=path.count(os.sep)
    for r,d,f in os.walk(path):
        if r.count(os.sep) - cpath <maxdepth:
            for files in f:
                if files.startswith(pattern):
                    try:
                        #print "Removing %s" % (os.path.join(r,files))
                        os.remove(os.path.join(r,files))
                    except Exception as e:
                        print(e)
                    else:
                        print ("%s removed" % (os.path.join(r,files)))
path = "./train_data/medium"
findNremove(path,"._",5)
train_closs(network, train_dataloader, dev_dataloader)
path = "./validation_classification/medium"
findNremove(path,"._",5)
train_closs(network, train_dataloader, dev_dataloader)
path = "./test_classification/medium"
findNremove(path,"._",5)
train_closs(network, train_dataloader, dev_dataloader)
