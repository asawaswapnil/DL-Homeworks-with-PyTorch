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
torch.cuda.empty_cache()

def usecsv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return(your_list)


class loader(Dataset):
    def __init__(self, file):
        self.file = file
        print(len(self.file))

    def __getitem__(self, index):
        image_name=self.file[index]
        path="test_verification/" +image_name
        img = Image.open(path)
        img = torchvision.transforms.ToTensor()(img)
        return {image_name:img}
    def __len__(self):
        return len(self.file)


def test_varify(model,var_loader):
    model.eval()
    model=model.to(device)
    print(device)
    out=np.zeros(899965)
    for batch_num, imgdict in enumerate(var_loader):
        print(batch_num)
        img=imgdict.values()
        img_names=imgdict.keys()
        img=img.to(device)
        ob=model(img)
        for i in range(len(ob)): 
            f = open("mod5/"+img_names[i]+'.pckl', 'wb')
            pickle.dump(ob[i], f)
            f.close()
        del img
        del ob

def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
    return img_list

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
    img_list=parse_data("test_verification")
    var_data = loader( file=img_list)
    var_loader = DataLoader(var_data, batch_size=16, shuffle= False, num_workers = 4, pin_memory=True,drop_last=False)
    test_varify(network, var_loader)
    # print(test_classify2(network,dev_dataloader))


