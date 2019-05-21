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
def init_weights(m):
	if type(m) == nn.Conv2d or type(m) == nn.Linear:
		torch.nn.init.xavier_normal_(m.weight.data)

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


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		def conv_bn(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True)
			)
		def conv_dw(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
				nn.BatchNorm2d(inp),
				nn.ReLU(inplace=True),
				nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True)
			)
		self.model = nn.Sequential(
			conv_bn(  3,  32, 2), 
			conv_dw( 32,  64, 1),
			conv_dw( 64, 128, 2),
			conv_dw(128, 128, 1),
			conv_dw(128, 256, 2),
			conv_dw(256, 256, 1),
			conv_dw(256, 512, 2),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 1024, 2),
			conv_dw(1024, 1024, 1),
			nn.AvgPool2d(7),
		)
		self.fc1 = nn.Linear(1024, 1024)
		self.fc2 = nn.Linear(1024, 2300)

	def forward(self, x):
		x = self.model(x)
		x = x.view(-1, 1024)
		x = self.fc1(x)
		x = self.fc2(x)
		return x

def train(model, data_loader, test_loader):
	#modeld=torch.load('mnet/train0.pt')
	#model.load_state_dict(modeld)
	prevtime=int(time.time())
	model.train()
	model.to(device)
	for epoch in range(numEpochs):
		avg_loss = 0.0
		for batch_num, (feats, labels) in enumerate(data_loader):
			feats, labels = feats.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(feats)
			loss = criterion(outputs, labels.long())
			loss.backward()
			optimizer.step()
			avg_loss += loss.item()

			if batch_num % 50 == 49:
				tm=int(time.time())
				print(tm-prevtime,'Epoch: {}\tBatch: {}\t#inst/batch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, len(labels),avg_loss/50))
				prevtime=tm
				avg_loss=0.0
			torch.cuda.empty_cache()
			del feats
			del labels
			del loss
			# if  batch_num%999==0:
		test_classify(model, test_loader)
		path="mnet/trainMod1"+str(epoch)+".pt"
		torch.save(model.state_dict(), path)   
		save_checkpoint({
			'epoch': epoch ,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			}, 1)
			#loadedepoch, loadedmodel, loadedoptimizer=load_checkpoint(epoch, model,optimizer)

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
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})".format('checkpoint.pth.tar', checkpoint['epoch']))
	else:
		print("=> no checkpoint found at '{}'".format(args.resume))
	return epoch, model,optimizer

def test_classify(model, test_loader):
	model.eval()
	model.to(device)
	test_loss = []
	accuracy = 0
	total = 0
	avg_loss=0
	for batch_num, (feats, labels) in enumerate(test_loader):
		feats, labels = feats.to(device), labels.to(device)
		outputs = model(feats)
		loss = criterion(outputs, labels.long())
		pred_labels=outputs.data.max(1, keepdim=True)[1]
		accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
		total += len(labels)
		avg_loss=torch.mean(loss)/(batch_num+1) +avg_loss*batch_num/(batch_num+1)
		print("Validation loss and accuracy",avg_loss, accuracy/total)
	model.train()
	return avg_loss, accuracy/total

if __name__ =='__main__':
	#remove irrelevant files
	path = "./train_data/medium"
	findNremove(path,"._",5)	
	path = "./validation_classification/medium"
	findNremove(path,"._",5)
	path = "./test_classification/medium"
	findNremove(path,"._",5)
	#setting dataloaders
	train_dataset = torchvision.datasets.ImageFolder(root='train_data/medium', transform=torchvision.transforms.ToTensor())
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,  shuffle=True, num_workers=4)
	dev_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium', transform=torchvision.transforms.ToTensor())
	dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=64,  shuffle=True, num_workers=4)

	#setting architecture
	numEpochs = 50
	num_feats = 3
	hidden_sizes = [32, 64]
	num_classes = len(train_dataset.classes)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	network = Net()
	#network.load_state_dict(torch.load('mnet/train4.pt'))

	network.apply(init_weights)
	criterion = nn.CrossEntropyLoss()
	optimizer =  torch.optim.Adam(network.parameters(),lr=0.001)
	#scheduler = ReduceLROnPlateau(optimizer, 'min')

	train(network, train_dataloader, dev_dataloader)   