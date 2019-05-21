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

class loader(Dataset):
	def __init__(self,inp,expOut=None, test=False):
		self.inp=inp#
		if (test==True):
			self.out=np.zeros((inp.shape[0],1))
		else:
			self.out=expOut

	def __getitem__(self,i):
		return torch.from_numpy(self.inp[i]),torch.from_numpy(self.out[i]) #MBS*US*40, MBS,#P
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
	return packedv, sorted_lens, targets_sorted, targets_sorted_lens

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


def train_epoch_packed(model, optimizer, train_loader, val_loader):
	train_loader=val_loader
	numEpochs=20
	epoch=0
	# test_classify(model, test_loader,epoch)

	for epoch in range(numEpochs):
		avg_loss = 0.0
		batch_id=0
		before = time.time()
		add_loss=0
		path=str(epoch)+".pt"

		for inputs,sorted_lens, targets_sorted, targets_sorted_lens in train_loader: # lists, presorted, preloaded on GPU
			targets_concat=torch.cat(targets_sorted)# output values concatinated
			print(batch_id)
			batch_id+=1
			optimizer.zero_grad()
			inputs,sorted_lens=inputs.to(DEVICE),torch.IntTensor(sorted_lens).to(DEVICE)
			targets_concat=targets_concat.to(DEVICE)
			# targets_sorted=torch.LongTensor(targets_sorted).to(DEVICE)
			targets_sorted_lens = torch.IntTensor(targets_sorted_lens).to(DEVICE)
			logits = model(inputs) # output dims=max_seq_len,bs,vocab_sizetargets_sorted
			probs = F.softmax(logits, dim=2)
			logprobs=LogSoftmax(probs)
			out_predicted, scores, timesteps, predicted_seq_len = ctc_decoder.decode(probs=probs, seq_lens=sorted_lens)
			loss = ctc_loss.forward(torch.transpose(logprobs,0,1),targets_concat,sorted_lens, targets_sorted_lens)
			loss.backward()
			optimizer.step()
			bloss=loss.detach().cpu().numpy()
			print(bloss)
			add_loss+=bloss
			if( batch_id % 50 == 49):
				print(add_loss)
				add_loss=0
				torch.save(model.state_dict(), path)   
				save_checkpoint({
					'epoch': epoch ,
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict(),
					}, 1)
		torch.save(model.state_dict(), path)   
		# test_classify(model, test_loader,epoch)
		del inputs,sorted_lens, targets_sorted, targets_sorted_lens,targets_concat,logits,probs,out_predicted, scores, timesteps 


def test_classify(model, test_loader, epoch):
	model.eval()
	filename = str(epoch)+"test_results.csv"
	final=np.empty(shape=(0,0))
	batch_id=0
	with torch.no_grad():
		for inputs,sorted_lens, targets_sorted, targets_sorted_lens in test_loader: # lists, presorted, preloaded on GPU
			batch_id+=1
			inputs,sorted_lens=inputs.to(DEVICE),torch.IntTensor(sorted_lens).to(DEVICE)
			logits = model(inputs) # output dims=max_seq_len,bs,vocab_sizetargets_sorted
			probs = F.softmax(logits, dim=2)
			out_predicted, scores, timesteps, predicted_seq_len = ctc_decoder.decode(probs=probs, seq_lens=sorted_lens)		
			pos = 0
			for i in range(out_predicted.size(0)): # for all utterances
				pred = "".join(PHONEME_MAP[o] for o in out_predicted[i, 0, :predicted_seq_len[i, 0]])
				final = np.append(final,  pred)
			del inputs,sorted_lens, targets_sorted, targets_sorted_lens,targets_concat,logits,probs,out_predicted, scores, timesteps 
	pd.DataFrame(data=final).to_csv(filename,header=False,  index=True)
	model.train()
	return

if __name__ == "__main__":
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	DEVICE
	ld = WSJ()
	trainX, trainY = ld.train
	valX,valY=ld.dev
	testX = np.load('transformed_test_data.npy')
	print("data_loaded")

	train_dataset = loader(trainX, trainY)
	val_dataset = loader(valX,valY)
	test_dataset=loader(testX,test=True)
	train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn = collate_lines)
	val_loader = DataLoader(val_dataset, shuffle=False, batch_size=32, collate_fn = collate_lines, drop_last=True)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn = collate_lines)
	label_list=[" "]+PHONEME_MAP#print("PHONEME_LIST", [" "]+PHONEME_LIST)
	print(len(label_list))
	ctc_decoder = ctc_model.CTCBeamDecoder(labels=label_list, blank_id=0)
	# print(ctc_decoder.vo)
	model = CharLanguageModel()
	LogSoftmax=nn.LogSoftmax(dim=2)
	ctc_loss=nn.CTCLoss(blank=0, reduction='mean')
	model = model.to(DEVICE)
	optimizer = torch.optim.Adam(model.parameters())


	train_epoch_packed(model, optimizer, train_loader, val_loader)

























# stop_character = charmap['\n']
# space_character = charmap[" "]
# lines = np.split(shakespeare_array, np.where(shakespeare_array == stop_character)[0]+1) # split the data in lines
# shakespeare_lines = []
# for s in lines:
#     s_trimmed = np.trim_zeros(s-space_character)+space_character # remove space-only lines
#     if len(s_trimmed)>1:
#         shakespeare_lines.append(s)
# for i in range(10):
#     print(sh.to_text(shakespeare_lines[i],chars))
# print(len(shakespeare_lines))


# idxs = np.random.choice(20000, 15)
# data_batch, label_batch = torch.Tensor(data[idxs]), torch.Tensor(labels[idxs]).long()

# logits, out_lengths = model(data_batch.unsqueeze(1))
# label_lengths = torch.zeros((15,)).fill_(10)ininin
# print(output.size())decode
# for i in range(output.size(0)):
#     chrs = [label_map[o.item()] for o in output[i, 0, :out_seq_len[i, 0]]]
#     image = data_batch[i].numpy()
#     plt.figure()
#     imshow(image, cmap='binary')
#     txt_top = "Prediction: {}".format("".join(chrs))
#     txt_bottom = "Labelling:  {}".format("".join(label_batch[i].numpy().astype(str)))
#     plt.figtext(0.5, 0.10, txt_top, wrap=True, horizontalalignment='center', fontsize=16)
#     plt.figtext(0.5, 0.01, txt_bottom, wrap=True, horizontalalignment='center', fontsize=16)