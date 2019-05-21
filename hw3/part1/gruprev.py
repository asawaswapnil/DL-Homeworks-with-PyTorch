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

		self.dWzx = np.zeros((d,h))
		self.dWrx = np.zeros((d,h))
		self.dWx  = np.zeros((d,h))

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
		self.x=x
		self.h=h
		self.dx=np.zeros((1,len(x)))
		self.dh=np.zeros((1,len(h)))

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
		z=[[]for i in len(18)]
		eqns=[
		[dz[1],		self.Wzh,	"dot",	self.h],
		[dz[2],		self.Wzx,	"dot",	self.x],
		[dz[3],		z[1],		"+",	z[2]],
		[dz[4],		z[3], 		"sig", 	0],

		[dz[5],self.Wrh,	"dot",	self.h],
		[dz[6],self.Wrx,	"dot",	self.x],
		[dz[7],z[5],	"+",	z[6]],
		[dz[8],z[7], 	"sig", 	0]		,
			
		[dz[9],z[8], "*",	self.h],
		[dz[10],self.Wh,	"dot",	z[9]],
		[dz[11],self.Wx,	"dot",	self.x],
		[dz[12],z[10],"+",	z[11]],
		[dz[13],z[12],"tanh", 0],
		
		[dz[14],1,'-',z[4]]
		[dz[15],z[14],"*", 	self.h],
		[dz[16],z[4], "*",	z[13]],
		[dz[17], z[15],"+",	z[16]]
		]
		for i in range(1,len(eqns)):
			z[i]=forw(eqns[i])
		
		self.dWzh.fill(0)
		self.dWrh.fill(0)
		self.dWh.fill(0)
		self.dWzx.fill(0)
		self.dWrx.fill(0)
		self.dWx.fill(0)
		self.dx.fill(0)
		self.dh.fill(0)
		dz=dz.fill(0)
		
		dz[17]=delta
		t=len(eqns)-1

		dz[15],dz[16]		=grad(eqns[t])
		t=t-1
		dz[4],dz[13]		=grad(eqns[t])
		t=t-1
		dz[14],self.dh 		=grad(eqns[t])
		t=t-1
		z[4]				=grad(eqns[t])
		t=t-1
		dz[12]				=grad(eqns[t])	
		t=t-1
		dz[10],		dz[11]	=grad(eqns[t])
		t=t-1
		self.dWx, self.dx 	=grad(eqns[t])
		t=t-1
		self.dWh,	dz[9] 	=grad(eqns[t])
		t=t-1
		dz[8],	self.dh 	=grad(eqns[t])
		t=t-1
		dz[7]				=grad(eqns[t])
		t=t-1
		dz[5],		dz[6] 	=grad(eqns[t])
		t=t-1
		self.dWrx,	self.dx =grad(eqns[t])
		t=t-1
		self.dWrh,	self.dh =grad(eqns[t])
		t=t-1
		dz[3]				=grad(eqns[t])
		t=t-1
		z[1],		z[2] 	=grad(eqns[t])
		t=t-1
		self.Wzx,	self.dx =grad(eqns[t])
		t=t-1
		self.dWzh,	self.dh =grad(eqns[t])

		return self.dx,self.dh

def forw(eqn):
	sig = Sigmoid()
	tanh=Tanh()
	if (op=='*'):
		return eqn[0]*eqn[1]
	if(op=='dot'):
		return np.dot(eqn[0],eqn[1])
	if(op=='+'):
		return eqn[0]+eqn[1]
	if(op=="tanh"):
		return tanh(eqn[0])
	if(op=="sig"):
		return sig(eqn[0])
	if (op=='-'):
		return eqn[0]-eqn[1]

def grad(eqn):
	sig = Sigmoid()
	tanh=Tanh()
	dz,x,op,y=eqn
	if (op=='*'):
		return dz*(x.T),  dz*(y.T)
	if(op=='dot'):
		return np.dot(y,dz), np.dot(dz,x)
	if(op=='+'):
		return dz, dz
	if(op=="tanh"):
		return dz*((1-tanh(x)*tanh(x)).T)
	if(op=="sig"):
		return dz*sig(x).T*(1-sig(x)).T
	if (op=='-'):
		return  eqn[0]-eqn[1]


if __name__ == '__main__':
	test()









