from resnet50 import *

import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import shutil
import time
import csv
import pickle

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)
def usecsv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return(your_list)

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

def train(model, data_loader, dev_loader,test_loader):
    #modeld=torch.load('mnet/train0.pt')
    #model.load_state_dict(modeld)
    prevtime=int(time.time())
    model.train()
    model.to(device)
    test_classify(model, dev_loader,test_loader,"mnet/res"+str(100)+"test.csv")
    for epoch in range(numEpochs):
        avg_loss = 0.0
        #adjust_learning_rate(optimizer)
        for batch_num, (feats, labels) in enumerate(data_loader):
            # print(feats.shape)
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            # print(outputs.size())
            # print(labels.size())
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                tm=int(time.time())
                print('T/b: {}\tEpoch: {}\tBatch: {}\t#inst/batch: {}\tAvg-Loss: {:.4f}'.format(tm-prevtime,epoch+1, batch_num+1, len(labels),avg_loss/50))
                prevtime=tm
                avg_loss=0.0
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
            # if  batch_num%999==0:
        test_classify(model, dev_loader,test_loader,"mnet/res"+str(epoch)+"test.csv")
        path="mnet/v2mob"+str(epoch)+".pt"
        torch.save(model.state_dict(), path)   
        save_checkpoint({
            'epoch': epoch ,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, 1)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
def load_checkpoint(epoch,model,optimizer):
    if os.path.isfile('checkpoint.pth.tar'):
        print("=> loading checkpoint '{}'".format('checkpoint.pth.tar'))
        checkpoint = torch.load('checkpoint.pth.tar')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'],map_location=device)
        print("=> loaded checkpoint '{}' (epoch {})".format('checkpoint.pth.tar', checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    return epoch, model,optimizer
def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n

class loader(Dataset):
    def __init__(self, file):
        self.file = usecsv(file)
        print(len(self.file))

    def __getitem__(self, index):
        image_name=self.file[index]
        # print(index)
        # print(image_name)
        path="test_verification/" +image_name[0]
        # print(path)
        img = Image.open(path)
        img = torchvision.transforms.ToTensor()(img)
        return img
        
    def __len__(self):
        return len(self.file)

def test_varify(model,var_loader1, var_loader2):
    model.eval()
    model=model.to(device)
    o1=np.empty((0,0))
    o2=np.empty((0,0))
    f = open('o1.pckl', 'wb')
    pickle.dump(o1, f)
    f.close()
    f = open('o2.pckl', 'wb')
    pickle.dump(o2, f)
    f.close()
    f = open('o2.pckl', 'wb')
    pickle.dump(o2, f)
    f.close()
    for batch_num, i1 in enumerate(var_loader1):
        i1=i1.to(device)
        # print(i1)
        o=model(i1)
        o=o.cpu().detach().numpy()
        o1=np.append(o1,  o)
        print("var_loader1", batch_num)
    f = open('o1.pckl', 'wb')
    pickle.dump(o1, f)
    f.close()
    for batch_num, i2 in enumerate(var_loader2):
        i2=i2.to(device)
        # print(i1)
        o=model(i2)
        o=o.cpu().detach().numpy()
        o2=np.append(o1,  o)
        print("var_loader1", batch_num)
    f = open('o2.pckl', 'wb')
    pickle.dump(o2, f)
    f.close()
    out=F.cosine_similarity(o1, o2)
    pd.DataFrame(data=out).to_csv("out.csv",header=False,  index=True)

if __name__ =='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = resnet34()
    state_dict = torch.load('mnet/v2mob4.pt')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): 
        # print(k)
        name = k[7:]
        new_state_dict[name] = v
    network.load_state_dict(state_dict)
    network=network.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer =  torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=0.000001)
    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    # load_checkpoint(4,network,optimizer)

    # print(train_classes)
    #test_classify(network,test_dataloader,4,train_classes)
    # print("try")
    pd.DataFrame(data=[[1],[2],[3]]).to_csv("out.csv",header=False,  index=True)

    var_data1 = loader( file="test_trials_verification_student1.csv")
    var_data2 = loader( file="test_trials_verification_student2.csv")
    var_loader1 = DataLoader(var_data1, batch_size=128, shuffle= False, num_workers = 4, pin_memory=True)
    var_loader2 = DataLoader(var_data2, batch_size=128, shuffle= False, num_workers = 4, pin_memory=True)

    test_varify(network, var_loader1, var_loader2)
    # print(test_classify2(network,dev_dataloader))
