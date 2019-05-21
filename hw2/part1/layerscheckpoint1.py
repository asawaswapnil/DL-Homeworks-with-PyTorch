import numpy as np
import math
import torch
import torch.nn as nn
from  torch.autograd import Variable
import multiprocessing as mtp
import traceback


class Linear():
	# DO NOT DELETE
	def __init__(self, in_feature, out_feature):
		self.in_feature = in_feature
		self.out_feature = out_feature

		self.W = np.random.randn(out_feature, in_feature)
		self.b = np.zeros(out_feature)
		
		self.dW = np.zeros(self.W.shape)
		self.db = np.zeros(self.b.shape)

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		self.x = x
		self.out = x.dot(self.W.T) + self.b
		return self.out

	def backward(self, delta):
		self.db = delta
		self.dW = np.dot(self.x.T, delta)
		dx = np.dot(delta, self.W.T)
		return dx

		

class Conv1D():
	def __init__(self, in_channel, out_channel, 
				 kernel_size, stride):

		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.stride = stride

		self.W = np.random.randn(out_channel, in_channel, kernel_size)
		self.b = np.zeros(out_channel)

		self.dW = np.zeros(self.W.shape)
		self.db = np.zeros(self.b.shape)

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		self.batch, __ , self.width = x.shape
		assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)
		#out=np.empty((0,0))
		width=len(x[0][0])
		in_channel=len(x[0])
		batch_size=len(x)
		#print(x.shape,self.W.shape,self.stride)
		#x shape is batch size, in channel, width

		# width=len(x[0])
		# for s2 in range(self.out_channel):
		# 	for s1 in range(batch_size):
		# 		for i in range(height-self.kernel_size+1):
		# 			out[][s2][i]=np.add(self.W[s2],x[batch_size][s1][][i:i+self.kernel_size])
		# 			print (s2,i)
		out=np.zeros((batch_size,self.out_channel,((width-self.kernel_size)//self.stride)+1))
		wshaped=self.W.reshape(self.out_channel,-1 )
	
		count=0
		for i in range(0,width-self.kernel_size+1,self.stride):
			xslice=x[:,:,i:i+self.kernel_size]
			xslice=xslice.reshape(batch_size,-1)
			out[:,:,count]=np.dot(xslice,np.transpose(wshaped))+self.b[None,:]
			#print (i,xslice.shape,wshaped.shape, out.shape)
			count+=1
		# out=out+self.b[None,:,None]
		# print(self.b[None, :,None].shape)
		return out

	def backward(self, delta):
		
		## Your codes here
		#self.db = 
		# self.dW = ???
		# return dx




class Flatten():
	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		## Your codes here
		raise NotImplemented

	def backward(self, x):
		# Your codes here
		raise NotImplemented




class ReLU():
	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		self.dy = (x>=0).astype(x.dtype)
		return x * self.dy

	def backward(self, delta):
		return self.dy * delta