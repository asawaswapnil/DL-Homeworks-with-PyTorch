
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
		def NoExpansion( inp, oup, stride):
			blk= nn.Sequential(
				nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
				nn.BatchNorm2d(inp),
				nn.ReLU6(inplace=True),

				nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup)
				)
			return blk
		def bottleneck( inp, oup, stride):
			expansion=6
			middle=inp*expansion
			blk= nn.Sequential(
				nn.Conv2d(inp, middle, 1, 1, 0, bias=False),
				nn.BatchNorm2d(middle),
				nn.ReLU6(inplace=True),

				nn.Conv2d(middle, middle, 3, stride, 1, groups=middle, bias=False),
				nn.BatchNorm2d(middle),
				nn.ReLU6(inplace=True),

				nn.Conv2d(middle, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup)
				)
			return blk
		def convNormal(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU6(inplace=True)
			)

		def conv_ptw(inp, oup):
			return nn.Sequential(
				nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU6(inplace=True)
			)
		self.model3 = [
			convNormal  (3,  32, stride=1), 
			NoExpansion (32, 16, stride=1),

			bottleneck  (16, 24, stride=1),
			bottleneck  (24, 24, stride=1),  

			bottleneck  (24, 32, stride=2),
			bottleneck  (32, 32, stride=1),
			bottleneck  (32, 32, stride=1),

			bottleneck  (64, 96, stride=2),
			bottleneck  (96, 96, stride=1),
			bottleneck  (96, 96, stride=1),

			bottleneck  (96,  160, stride=2),
			bottleneck  (160,  160, stride=1),

			bottleneck  (160, 320, stride=1),
			
			]


		self.model2 = [
			# convNormal  (3,  64, stride=1), 

			bottleneck  (3, 32, stride=2),
			bottleneck  (32, 32, stride=1),
			bottleneck  (32, 64, stride=1),

			bottleneck  (64, 128, stride=2),
			bottleneck  (128, 128, stride=1),
			bottleneck  (128, 256, stride=1),

			bottleneck  (256, 312, stride=2),
			bottleneck  (312, 512, stride=1),

			]
		self.model = [
			convNormal  (3,  32, stride=1), 
			NoExpansion (32, 16, stride=1),

			bottleneck  (16, 24, stride=2),
			bottleneck  (24, 24, stride=1),  

			bottleneck  (24, 32, stride=2),
			bottleneck  (32, 32, stride=1),
			bottleneck  (32, 32, stride=1),

			bottleneck  (32, 64, stride=2),
			bottleneck  (64, 64, stride=1),
			bottleneck  (64, 64, stride=1),
			bottleneck  (64, 64, stride=1),

			bottleneck  (64,  256, stride=1),
			#bottleneck  (256, 386, stride=1),

			# bottleneck  (64, 96, stride=2),
			# bottleneck  (96, 96, stride=1),
			# bottleneck  (96, 96, stride=1),

			# bottleneck  (96,  160, stride=1),
			# bottleneck  (160, 160, stride=1),
			# bottleneck  (160, 160, stride=1),

			# bottleneck  (160, 320, stride=1),

			#conv_ptw    (256, )
			]
		self.modelx = [
			convNormal  (3,  32, stride=2), 
			NoExpansion (32, 16, stride=1),

			bottleneck  (16, 24, stride=2),
			bottleneck  (24, 24, stride=1),  

			bottleneck  (24, 32, stride=2),
			bottleneck  (32, 32, stride=1),
			bottleneck  (32, 32, stride=1),

			bottleneck  (32, 64, stride=2),
			bottleneck  (64, 64, stride=1),
			bottleneck  (64, 64, stride=1),
			bottleneck  (64, 64, stride=1),

			bottleneck  (64, 96, stride=2),
			bottleneck  (96, 96, stride=1),
			bottleneck  (96, 96, stride=1),

			bottleneck  (96,  160, stride=1),
			bottleneck  (160, 160, stride=1),
			bottleneck  (160, 160, stride=1),

			bottleneck  (160, 320, stride=1),

			conv_ptw    (320, 1280)
			]
		
		for i in range(len(self.model)):
			self.model[i]=self.model[i].to(device)

		self.classifier = nn.Sequential(
			#nn.Dropout(0.5),
			nn.Linear(256*16, 2300)
			)


	def forward(self, x):
		#x = self.model(x) 
		#self.model[0].to(device)
		x1=x
		for block in self.model:
			x=block(x)
			if(x1.shape==x.shape):
				x=x+x1
			x1=x
		x = x.view(-1,256*16)
		x = self.classifier(x)
		return x
		# #x2=x
		# for block in self.model:
		# 	x=block(x)
		# 	# if(x2.shape==x.shape):
		# 	# 	x2=x2+x
		# 	# x=x2
		# # x = self.model(x) 
		# x = x.view(-1, 8192)
		# x = self.classifier(x)
		# return x


def train(model, data_loader, dev_loader,test_loader):
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
				print('T/b: {}\tEpoch: {}\tBatch: {}\t#inst/batch: {}\tAvg-Loss: {:.4f}'.format(tm-prevtime,epoch+1, batch_num+1, len(labels),avg_loss/50))
				prevtime=tm
				avg_loss=0.0
			torch.cuda.empty_cache()
			del feats
			del labels
			del loss
			# if  batch_num%999==0:
		test_classify(model, dev_loader,test_loader,"mnet/v2mobArch2_"+str(epoch)+"test")
		path="mnet/v2mob"+str(epoch)+".pt"
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

def test_classify(model, dev_loader,test_loader,filename):
	model.eval()
	model.to(device)
	test_loss = []
	accuracy = 0
	total = 0
	avg_loss=0
	for batch_num, (feats, labels) in enumerate(dev_loader):
		feats, labels = feats.to(device), labels.to(device)
		outputs = model(feats)
		loss = criterion(outputs, labels.long())
		pred_labels=outputs.data.max(1, keepdim=True)[1]
		accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
		total += len(labels)
		avg_loss=torch.mean(loss)/(batch_num+1) +avg_loss*batch_num/(batch_num+1)
		print("Validation loss and accuracy",avg_loss, accuracy/total)
	for batch_num, (feats, labels) in enumerate(test_loader):
		outputs = model(feats)
		pred_labels=outputs.data.max(1, keepdim=True)[1]
		pd.DataFrame(data=pred_labels).to_csv(filename,header=False,  index=True)
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
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,  shuffle=True, num_workers=4)
	dev_dataset = torchvision.datasets.ImageFolder(root='validation_classification/medium', transform=torchvision.transforms.ToTensor())
	dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=128,  shuffle=True, num_workers=4)
	test_dataset=torchvision.datasets.ImageFolder(root='test_classification', transform=torchvision.transforms.ToTensor())
	test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=128,  shuffle=False, num_workers=4)

	#setting architecture
	numEpochs = 50
	num_feats = 3
	hidden_sizes = [32, 64]
	num_classes = len(train_dataset.classes)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	network = Net()
	network.cuda()
	network = torch.nn.DataParallel(network, device_ids=range(torch.cuda.device_count()))
	#network.load_state_dict(torch.load('mnet/train4.pt'))

	network.apply(init_weights)
	network=network.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer =  torch.optim.RMSprop(network.parameters(),lr=0.001, weight_decay=0.99, momentum=0.9)
	#scheduler = ReduceLROnPlateau(optimizer, 'min')

	train(network, train_dataloader, dev_dataloader,test_dataloader)   