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
import sys
import torch.nn.functional as F
import Levenshtein as L
from ctcdecode import CTCBeamDecoder
from torch.autograd import Variable
import torch.optim as optim
import pdb;
import pandas as pd
import time
from torchsummary import summary



class loader(Dataset):
	def __init__(self,inp,expOut=None, test=False):
		self.inp=inp
		self.test=test
		print("data length", inp.shape)
		if (test==True):
			self.out=np.zeros((inp.shape[0],1))
		else:
			self.out=expOut

	def __getitem__(self,i):
		# print(self.inp[i])
		return torch.from_numpy(np.float32(self.inp[i])),torch.from_numpy(self.out[i]) #MBS*US*40, MBS,#P
	def __len__(self):
		return len(self.out)

def collate_lines(seq_list):
	inputs,targets = zip(*seq_list)
	lens = [len(seq) for seq in inputs]
	seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True) # bs* utterncesize*40
	inputs = [inputs[i] for i in seq_order]
	targets_sorted = [targets[i] for i in seq_order]
	
	targets_sorted_lens=[len(i) for i in targets_sorted] # output lengths concatinated 
	sorted_lens=[lens[i] for i in seq_order] 
	inputs=rnn.pad_sequence(inputs, batch_first=False, padding_value=0)# max len uterence, bs, 40 
	packedv=rnn.pack_padded_sequence(inputs, sorted_lens)
	# if (self.test==False):
	return packedv, sorted_lens, targets_sorted, targets_sorted_lens
	# if (self.test==True):
	# 	return packedv, sorted_lens

class CharLanguageModel(nn.Module):

	def __init__(self,vocab_size=47,embed_size=256,hidden_size=256, nlayers=4):
		super(CharLanguageModel,self).__init__()
		self.vocab_size=vocab_size
		self.hidden_size = hidden_size
		self.nlayers=nlayers
		self.rnn = nn.LSTM(input_size = 40,hidden_size=hidden_size,num_layers=nlayers) 
		self.scoring = nn.Linear(hidden_size,vocab_size) 
		
	def forward(self,x):
		batch_size = 2
		#hidden = None
		c=None
		output_lstm,_ = self.rnn(x) #output_lstm dim=max_seq_len,bs,hidden_size #hidden dims= number of layers, batch, hidden_size
		output_lstm_unpacked, input_sizes =rnn.pad_packed_sequence(output_lstm, batch_first=True)
		output = self.scoring(output_lstm_unpacked) # output dims=max_seq_len*bs,vocab_size
		return output # output dims=max_seq_len,bs,vocab_size


def train_epoch_packed(model, optimizer, train_loader, val_loader,test_loader):
	# train_loader=val_loader
	numEpochs=10
	epoch=0
	# test_classify(model, test_loader,epoch)

	for epoch in range(numEpochs):
		model.train()
		avg_loss = 0.0
		batch_id=0
		before = time.time()
		add_loss=0
		path=str(epoch)+".pt"
		start=time.time()
		print(start)
		for inputs,sorted_lens, targets_sorted, targets_sorted_lens in train_loader: # lists, presorted, preloaded on GPU
			targets_concat=torch.cat(targets_sorted)# output values concatinated
			# print(batch_id)
			torch.save(model.state_dict(), path)   
			batch_id+=1
			optimizer.zero_grad()
			inputs,sorted_lens=inputs.to(DEVICE),torch.IntTensor(sorted_lens).to(DEVICE)
			targets_concat=targets_concat.to(DEVICE)
			# targets_sorted=torch.LongTensor(targets_sorted).to(DEVICE)
			targets_sorted_lens = torch.IntTensor(targets_sorted_lens).to(DEVICE)
			logits = model(inputs) # output dims=bs, longestlenofinput,vocab_size
			# probs = F.softmax(logits, dim=2)
			logprobs=LogSoftmax(logits)
			# out_predicted, scores, timesteps, predicted_seq_len = ctc_decoder.decode(probs=probs, seq_lens=sorted_lens)
			loss = ctc_loss.forward(torch.transpose(logprobs,0,1),targets_concat,sorted_lens, targets_sorted_lens)
			loss.backward()
			optimizer.step()
			bloss=loss.detach().cpu().numpy()
			# print("bloss",epoch,bloss)
			add_loss+=bloss

			if( batch_id % 50 == 49):
				print("time for 50 batch",time.time()-start)
				print("add_loss",epoch,batch_id/50,add_loss)
				add_loss=0
			del inputs,sorted_lens, targets_sorted, targets_sorted_lens,targets_concat,logits,logprobs
		torch.save(model.state_dict(), path)   
		test_classify(model, test_loader,epoch)


def test_classify(model, test_loader, epoch):
	model.eval()
	filename = str(epoch)+"test_results.csv"
	final=np.empty(shape=(0,0))
	batch_id=0
	with torch.no_grad():
		for inputs,sorted_lens , targets_sorted, targets_sorted_lens in test_loader: # lists, presorted, preloaded on GPU
			batch_id+=1
			inputs,sorted_lens=inputs.to(DEVICE),torch.IntTensor(sorted_lens).to(DEVICE)
			# print(len(inputs))
			# print(inputs)
			logits = model(inputs) # output dims=max_seq_len,bs,vocab_sizetargets_sorted
			probs = F.softmax(logits, dim=2)
			out_predicted, scores, timesteps, predicted_seq_len = ctc_decoder.decode(probs=probs, seq_lens=sorted_lens)		
			pos = 0
			# print(out_predicted.size(0))
			# print(out_predicted.size())

			for i in range(out_predicted.size(0)): # for all utterances
				pred = "".join(PHONEME_MAP[o-1] for o in out_predicted[i, 0, :predicted_seq_len[i, 0]])
				print(pred)
				final = np.append(final,  pred)
			del inputs,sorted_lens, targets_sorted, targets_sorted_lens,logits,probs,out_predicted, scores, timesteps 
	
	pd.DataFrame(data=final).to_csv(filename,header=False,  index=True)
	model.train()
	return

if __name__ == "__main__":
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	DEVICE
	ld = WSJ()
	trainX, trainY = ld.train
	valX,valY=ld.dev
	testX = np.load('transformed_test_data.npy',encoding='bytes')
	print("trainlens",len(trainY))
	print("data_loaded")
	small_test=np.array([np.float32(a) for a in testX[1:10]])
	small_dev_X=np.array([np.float32(a) for a in valX[1:10]])
	small_dev_Y=np.array([np.float32(a) for a in valY[1:10]])
	np.save("small_test",small_test)
	np.save("small_dev_X",small_dev_X)
	np.save("small_dev_Y",small_dev_Y)

	train_dataset = loader(trainX, trainY)
	val_dataset = loader(valX,valY)
	test_dataset=loader(testX,test=True)

	train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn = collate_lines)
	val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32, collate_fn = collate_lines, drop_last=True)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn = collate_lines)
	label_list=[" "]+PHONEME_MAP#print("PHONEME_LIST", [" "]+PHONEME_LIST)
	# print(len(label_list))
	ctc_decoder = ctc_model.CTCBeamDecoder(labels=label_list, blank_id=0)	#magic box returns 100 beas with thier sccores
	model = CharLanguageModel()
	LogSoftmax=nn.LogSoftmax(dim=2)
	ctc_loss=nn.CTCLoss(blank=0, reduction='mean') # ‘mean’: the output losses will be divided by the target lengths and then the mean over the batch is taken. Default: ‘mean’
	# model = model.to(DEVICE)
	optimizer = torch.optim.Adam(model.parameters())
	# optimizer.to(DEVICE)
	summary(model)
	#train_epoch_packed(model, optimizer, train_loader, val_loader,test_loader)
	state_dict = torch.load('4.pt')
	print(state_dict)
	model=model.load_state_dict(state_dict)
	print(model)
	# test_classify(model, test_loader,00)
