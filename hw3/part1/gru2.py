import torch
import torch.nn as nn
import numpy as np
import itertools

class Sigmoid:
	"""docstring for Sigmoid"""
	def __init__(self):
		pass
	def forward(self, x):
		self.res = 1/(1+np.exp(-x))
		return self.res
	def backward(self):
		return self.res * (1-self.res)
	def __call__(self, x):
		return self.forward(x)


class Tanh:
	def __init__(self):
		pass
	def forward(self, x):
		self.res = np.tanh(x)
		return self.res
	def backward(self):
		return 1 - (self.res**2)
	def __call__(self, x):
		return self.forward(x)


class GRU_Cell:
	"""docstring for GRU_Cell"""
	def __init__(self, in_dim, hidden_dim):
		self.d = in_dim
		self.h = hidden_dim
		h = self.h
		d = self.d

		self.Wzh = np.random.randn(h,h)
		self.Wrh = np.random.randn(h,h)
		self.Wh  = np.random.randn(h,h)
		# converting to hidden dimension
		self.Wzx = np.random.randn(h,d)
		self.Wrx = np.random.randn(h,d)
		self.Wx  = np.random.randn(h,d)



		self.dWzh = np.zeros((h,h))
		self.dWrh = np.zeros((h,h))
		self.dWh  = np.zeros((h,h))

		self.dWzx = np.zeros((h,d))
		self.dWrx = np.zeros((h,d))
		self.dWx  = np.zeros((h,d))

		self.z_act = Sigmoid()
		self.r_act = Sigmoid()
		self.h_act = Tanh()

		
	def forward(self, x, h):
		# input:
		# 	- x: shape(input dim),  observation at current time-step
		# 	- h: shapTanhe(hidden dim), hidden-state at previous time-step
		# 
		# output:
		# 	- h_t: hidden state at current time-step
		self.Zt=self.z_act(np.dot(self.Wzh,h)+np.dot(self.Wzx,x))
		self.Rt=self.r_act(np.dot(self.Wrh,h)+np.dot(self.Wrx,x))
		self.Hcapt=self.h_act(np.dot(self.Wh,self.Rt*h)+np.dot(self.Wx,x))
		self.Ht=(1-self.Zt)*h+self.Zt*self.Hcapt
		return self.Ht
	def backward(self, delta):
		# input:
		# 	- delta: 	shape(hidden dim), summation of derivative wrt loss from next layer at 
		# 			same time-step and derivative wrt loss from same layer at
		# 			next time-step
		#
		# output:
		# 	- dx: 	Derivative of loss wrt the input x
		# 	- dh: 	Derivative of loss wrt the input hidden h
			dZtbydh=self.Zt*(1-self.Zt)*  +np.dot(self.Wzx,x)
			dHhcapt=(1-self.Hcapt*self.Hcapt)*(np.dot(dWh,)
			dh=1-self.Zt+self.Zt*dHhcapt









if __name__ == '__main__':
	test()









