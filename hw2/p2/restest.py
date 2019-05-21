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
def test_classify2(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        
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

def test_varify(model, test_loader, epoch, train_classes):
	model.eval()
	test_loss = []
	accuracy = 0
	total = 0
	final=np.empty(shape=(0,0))
	filename = str(epoch)+"test_results.csv"

	for batch_num, (feats, labels) in enumerate(test_loader):
		print(feats.shape,labels.shape)
		# x1=leat
		# x2=
		# x1=x1.to(device)
		# x2=x2.to(device)
		# o1=model(x1)
		# o2=model(x2)

		# _, pred_labels1 = torch.max(F.softmax(o1, dim=1), 1)
		# pred_labels1 = pred_labels1.view(-1)
		# final_labels1 = [train_classes[label_id] for label_id in pred_labels1.cpu().numpy()]
		# for i in final_labels1:
		# 	final1 = np.append(final1,  i)

		# _, pred_labels2 = torch.max(F.softmax(o2, dim=1), 1)
		# pred_labels2 = pred_labels2.view(-1)
		# final_labels2 = [train_classes[label_id] for label_id in pred_labels2.cpu().numpy()]
		# for i in final_labels2:
		# 	final2 = np.append(final2,  i)	
		# sim=np.zeros((2300))
		# for i in range(len(final1)):
		# 	sim[0][i]=F.cosine_similarity(o1, o2, 2300)
		# print(sim)

		# feats, labels = feats.to(device), labels.to(device)
		# outputs = model(feats)

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
		# final=np.append(final,[train_classes[label_id] for label_id in pred_labels.cpu().numpy()])
		# print('class 1',test_classes[1])
		# exit()
		# final1 = np.append(final,test_classes[pred_labels])
		
		# loss = criterion(outputs, labels.long())
		
		# accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
		# total += len(labels)
		# test_loss.extend([loss.item()]*feats.size()[0])
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
	path = "./test_varification"
	findNremove(path,"._",5)
	train_dataset = torchvision.datasets.ImageFolder(root='train_data/medium', transform=torchvision.transforms.ToTensor())
	# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,  shuffle=True, num_workers=4)
	# dev_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium', transform=torchvision.transforms.ToTensor())
	# dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=128,  shuffle=True, num_workers=4)
	var_dataset = torchvision.datasets.ImageFolder(root='test_verification', transform=torchvision.transforms.ToTensor())
	var_dataloader = torch.utils.data.DataLoader(var_dataset, batch_size=128,  shuffle=True, num_workers=4)
	
	test_dataset=torchvision.datasets.ImageFolder(root='test_classification', transform=torchvision.transforms.ToTensor())
	test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=128,  shuffle=False, num_workers=4)
	numEpochs = 25
	num_feats = 3
	hidden_sizes = [32, 64]
	num_classes =2300
	train_classes=train_dataset.classes
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	network = resnet34()
	state_dict = torch.load('mnet/v2mob3.pt')
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():	
		name = k[7:]
		new_state_dict[name] = v
	network.load_state_dict(new_state_dict)
	network=network.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer =  torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=0.000001)
	print(train_classes)
	#test_classify(network,test_dataloader,4,train_classes)

	test_varify(network, var_dataloader,6, train_classes)
	# print(test_classify2(network,dev_dataloader))
