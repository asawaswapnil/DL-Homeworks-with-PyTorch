#todo: 
#2 types of noise
#output with beam search
# implimnet beam search.
#graphs
import shutil

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
from wsj_loader import WSJ
import ctc_model
from phoneme_list import * 
import argparse
import csv
import os
from model import Seq2Seq 
import itertools
import sys
import torch.nn.functional as F
#import Levenshtein as L
#from ctcdecode import CTCBeamDecoder
from torch.autograd import Variable
import torch.optim as optim
import pdb
import pandas as pd
import time
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

def init_weights(m):
	if type(m) == nn.Conv2d or type(m) == nn.Linear:
		torch.nn.init.xavier_normal_(m.weight.data)

class loader(Dataset):
	def __init__(self,inp,expOut=None, test=False):
		self.inp=inp
		self.test=test
		print("selfout",self.inp.shape)

		if (test==True):
			self.out=torch.IntTensor(np.zeros((self.inp.shape[0],1)))
		else:
			self.out=expOut
			#print(self.out)

	def __getitem__(self,i):
		# print("inside dataloader")
		# print(np.float32(self.inp[i]))
		return torch.from_numpy(np.float32(self.inp[i])),torch.from_numpy(np.array(self.out[i])) #MBS*US*40, MBS,#P
	def __len__(self):
		return len(self.out)

def collate_lines(seq_list):
	inputs,targets = zip(*seq_list)
	lens = [len(seq) for seq in inputs]
	maxlen=max(lens)
	seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True) # bs* utterncesize*40
		
	inputs = [inputs[i] for i in seq_order]
	#inputs=
	targets_sorted = [targets[i].tolist() for i in seq_order]
	targets_concat=[]
	for i in range(len(targets_sorted)):
		targets_sorted[i].insert(0, 32) 
		targets_sorted[i].append(32)
		targets_concat.append(targets_sorted[i])
		targets_sorted[i]=torch.tensor(targets_sorted[i])
	pad_targets_sorted_lens=[len(i) for i in targets_sorted] # output lengths concatinated 
	pad_targets_sorted=	rnn.pad_sequence(targets_sorted, batch_first=False, padding_value=33)# max len uterence, bs, 40 
	sorted_lens=[lens[i] for i in seq_order] 
	#pdb.set_trace()
	#print(pad_targets_sorted)
	return inputs, sorted_lens , torch.tensor(pad_targets_sorted), pad_targets_sorted_lens, sum(targets_concat,[])


def train_epoch_packed(model, optimizer, train_loader,test_loader,list_of_char):#, val_loader,test_loader):
	# train_loader=val_loader
	numEpochs=20
	epoch=0
	Loss=nn.CrossEntropyLoss(reduction='none')
	Loss=Loss.to(device)
	#test_classify(model, test_loader,epoch,list_of_char)
	for epoch in range(numEpochs):
		print("epoch",epoch)
		model.train()
		avg_loss = 0.0
		batch_id=0
		#before = time.time()
		add_loss=0
		path=str(epoch)+".pt"
		start=time.time()
		for inputs,sorted_lens, targets_sorted, targets_sorted_lens,targets_concat in train_loader: # lists, presorted, preloaded on GPU
			
			#prev=start
			#start=time.time()
			#print(batch_id,"time",start-prev)
			batch_id+=1		
			#print((targets_sorted))	
			#inputs=inputs.to(device)
			#print(inputs)
			targets_sorted=targets_sorted.to(device)
			maxlen=targets_sorted_lens[0]
			optimizer.zero_grad()
			print("sorted",targets_sorted)
			logits,words = model(inputs,targets_sorted[0:-1,:],TF=1-epoch/20) # output dims=bs, longestlenofinput,vocab_size
			#probs=F.softmax(logits,dim=2) #S,N,V
			#break;
			# output_submask=[[1]*targets_sorted_lens[i] for i in range(len(targets_sorted_lens))]
			# output_submask=torch.Tensor(output_subinputsmask)
			# output_submask=rnn.pad_sequence(output_submask, batch_first=True,padding_value=0)
			#outmask=[output_submask[i]+output_submask[i] for i in range(len(targets_sorted_lens))]
			#inputs,sorted_lens=inputs.to(DEVICE),torch.IntTensor(sorted_lens).to(DEVICE)
			#targets_concat=targets_concat.to(DEVICE)
			#targets_sorted_lens = torch.IntTensor(targets_sorted_lens).to(DEVICE)			
			output_submask=targets_sorted.clone().detach()
			output_submask=output_submask.transpose(0,1)
			output_submask[output_submask==33]=-1.0
			output_submask[output_submask>=0]=1.0
			output_submask[output_submask<0]=0.0
			output_submask=output_submask.transpose(0,1)
			output_submask=output_submask[:-1] #TODO: eos is by mistake taken care by this, don't mask eos
			output_submask=output_submask.view(-1)
			output_submask=output_submask.float()
			total_valid_chars=sum(output_submask).detach()
			#print("total_valid_chars",total_valid_chars)
			logits_for_loss=logits.view(logits.shape[0]*logits.shape[1],logits.shape[2]) #S*N,V
			targets_for_loss=targets_sorted[1:,:].view(logits.shape[0]*logits.shape[1])#S*N
			# print("output_submask",output_submask.shape)
			# print("targets_for_loss",targets_for_loss.shape)
			
			loss=Loss(logits_for_loss,targets_for_loss) #S*N
			loss.type(torch.FloatTensor)
			# print("loss",loss.shape)

			loss=(loss)*(output_submask.detach())
			# print("loss",loss.shape)

			loss=sum(loss)/total_valid_chars
			loss.backward()
			optimizer.step()
			bloss=loss.detach().cpu().numpy()
			add_loss+=bloss
			#print(loss)
			# out_predicted, scores, timesteps, predicted_seq_len = ctc_decoder.decode(probs=probs, seq_lens=sorted_lens)		

			if( batch_id % 25 == 0):
				print("time for batch",time.time()-start)
				print("add_loss",epoch,batch_id/50,add_loss)
				add_loss=0
				save_checkpoint({
				'epoch': epoch ,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				}, 1)
				start=time.time()
			del inputs,sorted_lens, targets_sorted, targets_sorted_lens,targets_concat
			del output_submask,logits_for_loss, targets_for_loss,logits,maxlen
		torch.save(model.state_dict(), path)   
		test_classify(model, test_loader,epoch,list_of_char)


def test_classify(model, test_loader, epoch,list_of_char):
	model.eval()
	filename = str(epoch)+"test_results.csv"
	final=np.empty(shape=(0,0))
	batch_id=0
	start=time.time()
	print("test")
	final=[]
	#with torch.no_grad():
	for inputs,sorted_lens, _, _,_ in test_loader: # lists, presorted, preloaded on GPU
		prev=start
		start=time.time()
		print(batch_id,"time",start-prev)
		batch_id+=1		
		targets_sorted_test=torch.Tensor(np.ones((100,1))*32)
		#print((targets_sorted))	
		#inputs=inputs.to(device)
		#print(inputs)100
		targets_sorted_test=targets_sorted_test.to(device)
		#maxlen=targets_sorted_lens[0]
		# optimizer.zero_grad()
		# print(inputs,sorted_lens, targets_sorted_test)
		logits,words = model(inputs,targets_sorted_test[0:-1,:],TF=0) # output dims=bs, longestlenofinput,vocab_size
		
		chars="".join(list_of_char[i] for i in words)
		print(chars)
		final.append(chars)

		# probs=F.softmax(logits,dim=2) #S,N,V # output dims=bs, longestlenofinput,vocab_size
		# print(logits)
		# probs=F.softmax(logits,dim=2) #S,N,V
		# print(probs.shape)
		# probs=torch.transpose(probs, 0,1)
		# probs=torch.transpose(probs, 1,2)

		# print(probs.shape)

		# sorted_lens=torch.FloatTensor(sorted_lens).detach()#.to(device)
		# pdb.set_trace()
		# out_predicted, scores, timesteps, predicted_seq_len = ctc_decoder.decode(probs=probs, seq_lens=torch.Tensor(sorted_lens))		
		# pos = 0

		# for i in range(out_predicted.size(0)): # for all utterances
		# 	pred = "".join([o-1] for o in out_predicted[i, 0, :predicted_seq_len[i, 0]])
		# 	print(pred)
		# 	final = np.append(final,  pred)
		# del inputs,sorted_lens, targets_sorted_test, logits,probs,out_predicted, scores, timesteps 

	pd.DataFrame(data=final).to_csv(filename,header=False,  index=True)
	model.train()
	return

if __name__ == "__main__":
	device = "cuda" if torch.cuda.is_available() else "cpu"
	device
	ld = WSJ()
	trainX, trainY = ld.train
	valX,valY=ld.dev
	testX = np.load('../data/test.npy',encoding='bytes')
	print("trainlens",len(trainY))
	print("data_loaded")

	# small_test=np.array([a for a in testX[1:10]])
	# small_train_X=np.array([a for a in trainX[1:100]])
	# small_train_Y=np.array([y for y in trainY[1:100]])
	# small_dev_X=np.array([a for a in valX[1:10]])
	# small_dev_Y=np.array([y for y in valY[1:10]])

	# np.save("small_test",small_test)
	# np.save("small_dev_X",small_dev_X)
	# np.save("small_dev_Y",small_dev_Y)
	# np.save("small_train_X",small_train_X)
	# np.save("small_train_Y",small_train_Y)

	# small_test=np.load("small_test.npy")
	# small_dev_X=np.load("small_dev_X.npy")
	# small_dev_Y=np.load("small_dev_Y.npy")
	# small_train_X=np.load("small_train_X.npy")
	# small_train_Y=np.load("small_train_Y.npy")

	small_test=testX
	small_dev_X=valX
	small_dev_Y=valY
	small_train_X=trainX 
	small_train_Y=trainY
	for i in range(small_train_Y.shape[0]):
		small_train_Y[i]= [y.decode('utf-8') for y in small_train_Y[i]]
		small_train_Y[i]=	list(' '.join(small_train_Y[i]))
	list_of_char=list(set(sum(small_train_Y, [])))
	list_of_char.sort()
	list_of_char.append('!')
	list_of_char.append('@')
	ind=[i for i in range(len(list_of_char))]
	ind=np.array(ind)
	np.save("ind", ind)
	list_of_char=np.array(list_of_char)
	np.save("list_of_char", list_of_char)
	ind=np.load("ind.npy")
	vocab_size=len(ind)

	list_of_char=np.load("list_of_char.npy")
	print("list_of_char",list_of_char)
	dic=dict(zip(list_of_char.tolist(),range(len(list_of_char))))
	for i in range(small_train_Y.shape[0]):
		small_train_Y[i]=np.array([ dic[y] for y in small_train_Y[i]])
	
	small_train_Y=np.array(small_train_Y)
	np.save("train_Y", small_train_Y)
	small_train_Y=np.load("train_Y.npy")
	print(small_train_Y[1][1])

	#dic=dict(zip(list_of_char,range(len(list_of_char))))
	#print("list of char",dic)
	#print(list_of_char)
	#small_train_Y[i]=	[ord(y) for y in small_train_Y[i]]
	# print(small_train_X)
	# print("trainY",small_train_Y)
	# list_of_char=(sum(small_train_Y, []))
	# print(*small_train_Y)
	# print("set",list_of_char)
	

	for i in range(small_dev_Y.shape[0]):
		small_dev_Y[i]= [y.decode('utf-8') for y in small_dev_Y[i]]
		small_dev_Y[i]=	list(' '.join(small_dev_Y[i]))
		small_dev_Y[i]=	[ord(y) for y in small_dev_Y[i]]
	# print(small_dev_X)
	# print(small_dev_Y)
	# print(small_train_X[0].shape)
	# print(testX[0].shape	)
	print("train shape")
	train_dataset = loader(small_train_X, small_train_Y)
	print("test shape")
	test_dataset=loader(testX,test=True)
	#val_dataset = loader(valX,valY)	
	#val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32, collate_fn = collate_lines, drop_last=True)
	train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn = collate_lines)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn = collate_lines)

	model = Seq2Seq(base=64, out_dim=34,device=device)
	
	LogSoftmax=nn.LogSoftmax(dim=2)
	model = model.to(device)
	optimizer= torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
	

	# # state_dict=torch.load("19.pt")
	# print(model)
	# # print(state_dict)
	# #test_classify(model, test_loader,00)
	train_epoch_packed(model, optimizer, train_loader,test_loader, list_of_char)#, val_loader,test_loader)