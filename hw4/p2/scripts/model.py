#All code for your nn.Module/s (encoder, decoder, etc) in a python file, referred to as model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, pack_padded_sequence
from lockeddrop import LockedDrop
from weightdrop import WeightDrop
import numpy as np
import pdb
Device = "cuda" if torch.cuda.is_available() else "cpu"
tensortype = 'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'
tensortype2 = 'torch.cuda.LongTensor' if torch.cuda.is_available() else 'torch.LongTensor'

class Encoder(nn.Module):
	def __init__(self, base=64):
		super(Encoder, self).__init__()
		self.lstm1 = nn.LSTM(40, base, bidirectional = True)
		self.lstm2 = self.__make_layer__(base*4, base)
		self.lstm3 = self.__make_layer__(base*4, base)
		self.lstm4 = self.__make_layer__(base*4, base)

		self.fc1 = nn.Linear(base*2, base*2)
		self.fc2 = nn.Linear(base*2, base*2)
		self.act = nn.SELU(True)

		self.drop = LockedDrop(.05)

	def _stride2(self, x):
		x = x[:x.size(0)//2*2]
		x = self.drop(x)
		x = x.permute(1, 0, 2) # seq, batch, feature -> batch, seq, feature
		x = x.reshape(x.size(0), x.size(1)//2, x.size(2)*2)
		x = x.permute(1, 0, 2) # batch, seq, feature -> seq, batch, feature
		return x

	def __make_layer__(self, in_dim, out_dim):
		lstm = nn.LSTM(input_size=in_dim, hidden_size=out_dim, bidirectional=True)
		return WeightDrop(lstm, ['weight_hh_l0','weight_hh_l0_reverse'],dropout=0.1, variational=True)
		return lstm

	def forward(self, x):
		x = pack_sequence(x)                      # seq, batch, 40
		x=x.to(Device)
		x, _ = self.lstm1(x)                      # seq, batch, base*2
		
		x, seq_len = pad_packed_sequence(x)
		x = self._stride2(x)                      # seq//2, batch, base*4

		x = pack_padded_sequence(x, seq_len//2)
		x, _ = self.lstm2(x)                      # seq//2, batch, base*2
		x, _ = pad_packed_sequence(x)
		x = self._stride2(x)                      # seq//4, batch, base*4

		x = pack_padded_sequence(x, seq_len//4)
		x, _ = self.lstm3(x)                      # seq//4, batch, base*2
		x, _ = pad_packed_sequence(x)
		x = self._stride2(x)                      # seq//8, batch, base*4

		x = pack_padded_sequence(x, seq_len//8)
		x, (hidden, _) = self.lstm4(x)            # seq//8, batch, base*2
		x, _ = pad_packed_sequence(x)

		key = self.act(self.fc1(x))
		value = self.act(self.fc2(x))
		hidden = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=1)
		return seq_len//8, key, value, hidden


class Attention(nn.Module):
	def __init__(self):
		super(Attention, self).__init__()

	def forward(self, hidden2, key, value, mask):
		#print(hidden2.shape,key.shape,value.shape,mask.shape)

		# key: seq//8, batch, base*2 --> batch base*2, seq//8
		# hidden2: batch, base*2     --> batch 1 base*2
		# key: seq//8, batch, base*2 --> batch base*2, seq//8
		# hidden2: batch, base*2     --> batch 1 base*2
		# value: seq//8, batch, base*2 --> batch base*2, seq//8
		#shape of context: batch base*2, 1

		key=key.permute(1,2,0) #batch base*2, seq//8
		value=value.permute(1,2,0)#batch base*2, seq//8
		hidden2=torch.unsqueeze(hidden2,1) #batch 1 base*2
		energy= torch.bmm(hidden2,key) #[ batch 1 base*2] * [batch base*2, seq//8]= batch,1,seq//8
		soft=nn.Softmax(dim=2) 
		attention_wts=soft(energy) #batch,1,seq//8
		attention_wts=attention_wts.squeeze(1) #batch,seq//8
		masked_attention=attention_wts*mask.type(tensortype) 		# batch seq//8
		attention_wts=torch.nn.functional.normalize(masked_attention, p=1, dim=1, eps=1e-12) #L1 norm
		attention_wts=torch.unsqueeze(attention_wts,2)# batch seq//8 1
		context=torch.bmm(value,attention_wts)#[batch base*2, seq//8]*[ batch seq//8 1]= batch, base*2, 1
		#return context.squeeze(2), energy.cpu().squeeze(2).data.numpy()
		return context.squeeze(2), attention_wts.cpu().squeeze(2).data.numpy()#return  batch, base*2

class Decoder(nn.Module):
	def __init__(self, out_dim, lstm_dim):
		super(Decoder, self).__init__()
		self.embed = nn.Embedding(out_dim, lstm_dim)
		self.lstm1 = nn.LSTMCell(lstm_dim*2, lstm_dim)
		self.lstm2 = nn.LSTMCell(lstm_dim, lstm_dim)
		self.drop = nn.Dropout(0.05)
		self.fc = nn.Linear(lstm_dim, out_dim)
		self.fc.weight = self.embed.weight

	def forward(self, x, context, hidden1, cell1, hidden2, cell2, first_step):
		#print(x.shape,x	 )
		#pdb.set_trace()
		x = self.embed(x.type(tensortype2))
		x = torch.cat([x, context], dim=1)
		if first_step:
			hidden1, cell1 = self.lstm1(x)
			hidden2, cell2 = self.lstm2(hidden1)
		else:
			hidden1, cell1 = self.lstm1(x, (hidden1, cell1))
			hidden2, cell2 = self.lstm2(hidden1, (hidden2, cell2))
		x = self.drop(hidden2)
		x = self.fc(x)
		x = self.fc(hidden2)
		#this x is called logits
		return x, hidden1, cell1, hidden2, cell2



class Seq2Seq(nn.Module):
	def __init__(self, base, out_dim, device):
		super().__init__()
		self.encoder = Encoder(base)
		self.decoder = Decoder(out_dim=out_dim, lstm_dim=base*2)
		self.attention = Attention()
		self.out_dim = out_dim
		self.device = device

		for name, param in self.named_parameters():
			if 'weight' in name:
				nn.init.orthogonal_(param.data)
			else:
				nn.init.constant_(param.data, 0)


	def forward(self, inputs, words, TF=0.7):
		max_len, batch_size = words.shape[0], words.shape[1]
		prediction = torch.zeros(max_len, batch_size, self.out_dim).to(self.device)

		word, hidden1, cell1, hidden2, cell2 = words[0,:], None, None, None, None

		lens, key, value, hidden2 = self.encoder(inputs)
		mask = torch.arange(lens.max()).unsqueeze(0) < lens.unsqueeze(1)
		mask = mask.to(Device)
		figure = []
		wordspred=[]
		for t in range(1, max_len):
			context, attention  = self.attention(hidden2, key, value, mask)
			word_vec, hidden1, cell1, hidden2, cell2 = self.decoder(word, context, hidden1, cell1, hidden2, cell2, first_step=(t==1))
			prediction[t] = word_vec
			teacher_force = torch.rand(1) < TF
			if teacher_force:
				word = words[t]
			else:
				word = word_vec.max(1)[1]
			wordspred.append(int(word_vec.max(1)[1].cpu().detach().data[0]))	
			figure.append(attention)
		np.save('attention', np.stack(figure))
		return prediction,wordspred
