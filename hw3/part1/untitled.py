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
		self.z=[[]for i in range(25)]
		self.z[0]=self.x
		self.z[1]=self.h
		self.z[2]=self.Wzh
		self.z[3]=self.Wrh
		self.z[4]=self.Wh
		self.z[5]=self.Wzx
		self.z[6]=self.Wrx
		self.z[7]=self.Wx
		eqns=[
		[2,	"dot",	1],
		[5,	"dot",	0],
		[1+7,		"+",	2+7],
		[3+7, 		"sig", 	-1],

		[3,	"dot",	1],
		[6,	"dot",	0],
		[5+7,	"+",	6+7],
		[7+7, 	"sig", 	-1]		,
			
		[8+7, "*",	1],
		[4,	"dot",	9+7],
		[7,	"dot",	0],
		[10+7,"+",	11+7],
		[12+7,"tanh", -1],
		
		[-1,'-',4+7]
		[14+7,"*", 	1],
		[4+7, "*",	13+7],
		[15+7,"+",	16+7]
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
		self.dz=[z[i].T for i in range(25)]
		self.dz.fill(0)
		self.dz[24]=delta
		for i in range(len(dz)-1,-1,-1)
			grad(i+7, eqns[i])
		# dz=dz.fill(0)

		return self.dx,self.dh

	def forw(eqn):
		sig = Sigmoid()
		tanh=Tanh()
		if (op=='*'):
			return self.z[eqn[0]]*self.z[eqn[2]]
		if(op=='dot'):
			return np.dot(self.z[eqn[0]],self.z[eqn[2]])
		if(op=='+'):
			return self.z[eqn[0]]+self.z[eqn[2]]
		if(op=="tanh"):
			return tanh(self.z[eqn[0]])
		if(op=="sig"):
			return sig(self.z[eqn[0]])
		if (op=='-'):
			return 1-self.z[eqn[2]]

	def grad(dzi,eqn):
		sig = Sigmoid()
		tanh=Tanh()
		xi,op,yi=eqn
		if (op=='*'):
			self.dz[xi]= dz*(x.T)
			self.dz[yi]= dz*(y.T)
		if(op=='dot'):
			self.dz[xi]=np.dot(y,dz)
			self.dz[yi]=np.dot(dz,x)
		if(op=='+'):
			self.dz[xi]=dz
			self.dz[yi]= dz
		if(op=="tanh"):
			self.dz[xi]=dz*((1-tanh(x)*tanh(x)).T)
		if(op=="sig"):
			self.dz[xi]=dz*sig(x).T*(1-sig(x)).T
		if (op=='-'):
			self.dz[yi]= -dz


if __name__ == '__main__':
	test()









