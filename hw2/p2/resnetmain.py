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


# def adjust_learning_rate( epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 4 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 4))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
def train(model, data_loader, dev_loader,test_loader,train_classes):
	#modeld=torch.load('mnet/train0.pt')
	#model.load_state_dict(modeld)
	prevtime=int(time.time())
	model.train()
	model.to(device)
	test_classify(model,test_loader,"init",train_classes)
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
			if  batch_num%3499==0:
				test_classify(model, test_loader, epoch+6+100, train_classes)
		test_classify(model, test_loader, epoch+6, train_classes)
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
		state_dict =checkpoint['state_dict']
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():	
			name = k[7:]
			new_state_dict[name] = v
		network.load_state_dict(new_state_dict)
		# model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})".format('checkpoint.pth.tar', checkpoint['epoch']))
	else:
		print("=> no checkpoint found at '{}'".format(args.resume))
	return epoch, model,optimizer

def test_classify(model, test_loader, epoch, train_classes):
	model.eval()
	test_loss = []
	accuracy = 0
	total = 0
	final=np.empty(shape=(0,0))
	filename = str(epoch)+"test_results.csv"

	for batch_num, (feats, labels) in enumerate(test_loader):
		feats, labels = feats.to(device), labels.to(device)
		outputs = model(feats)
		
		_, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
		pred_labels = pred_labels.view(-1)
		# final=np.append(final,pred_labels.cpu().numpy())
		final_labels = [train_classes[label_id] for label_id in pred_labels.cpu().numpy()]
		for i in final_labels:
			final = np.append(final,  i)
		del feats
		del labels
	pd.DataFrame(data=final).to_csv(filename,header=False,  index=True)
	model.train()
	return


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
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,  shuffle=True, num_workers=4)
	dev_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium', transform=torchvision.transforms.ToTensor())
	dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=128,  shuffle=True, num_workers=4)
	test_dataset=torchvision.datasets.ImageFolder(root='test_classification', transform=torchvision.transforms.ToTensor())
	test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=128,  shuffle=False, num_workers=4)
	train_classes=train_dataset.classes

	#setting architecture
	numEpochs = 25
	num_feats = 3
	hidden_sizes = [32, 64]
	num_classes = len(train_dataset.classes)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	network = resnet34()
	state_dict = torch.load('mnet/v2mob4.pt')
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():	
		name = k[7:]
		new_state_dict[name] = v
	network.load_state_dict(new_state_dict)
	network=network.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer =  torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=0.000001)
	#scheduler = ReduceLROnPlateau(optimizer, 'min')
	load_checkpoint(4,network,optimizer)

	train(network, train_dataloader, dev_dataloader,test_dataloader,train_classes)	