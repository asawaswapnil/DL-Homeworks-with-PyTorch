from resnet50 import *

import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import time
import csv
import pickle
torch.cuda.empty_cache()

def usecsv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return(your_list)

class loader(Dataset):
    def __init__(self, file):
        self.file = usecsv(file)
        print(len(self.file))

    def __getitem__(self, index):
        image_name=self.file[index][0]
        image_name2=self.file[index][1]
        # print(index)        
        # print(image_name,image_name2)
        path="test_verification/" +image_name
        path2="test_verification/" +image_name2
        # print(path)
        img = Image.open(path)
        img2=Image.open(path2)
        img = torchvision.transforms.ToTensor()(img)
        img2=  torchvision.transforms.ToTensor()(img2)
        return [img,img2]
        
    def __len__(self):
        return len(self.file)

def test_varify(model,var_loader):
    model.eval()
    model=model.to(device)
    print(device)
    out=np.zeros(899965)
    for batch_num, imgpair in enumerate(var_loader):
        i1=imgpair[0]
        i2=imgpair[1]
        print(imgpair[0][0])
        print(imgpair[0][1])
        i1=i1.to(device)
        i2=i2.to(device)
        o1=model(i1)
        o2=model(i2)
        sim=F.cosine_similarity(o1,  o2)
        out[batch_num*128:batch_num*128+len(o1)]=sim.detach().cpu()
        # print((out[0:batch_num*128+len(o1)]))
        # print("var_loader1", batch_num)
        del i1
        del i2
        del sim
    pd.DataFrame(data=out).to_csv("outmodel4.csv",header=False,  index=True)

if __name__ =='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = resnet34()
    state_dict = torch.load('mnet/v2mob5.pt')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): 
        name = k[7:]
        new_state_dict[name] = v
    network.load_state_dict(state_dict)
    network=network.to(device)
    var_data = loader( file="test_trials_verification_student.csv")
    var_loader = DataLoader(var_data, batch_size=128, shuffle= False, num_workers = 4, pin_memory=True,drop_last=True)
    test_varify(network, var_loader)
