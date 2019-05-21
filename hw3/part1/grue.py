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
		# self.d = in_dim
		# self.h = hidden_dim
		h = hidden_dim
		d = in_dim

		self.Wzh = np.ones((h,h),dtype=np.float64)
		self.Wrh = np.ones((h,h),dtype=np.float64)
		self.Wh  = np.ones((h,h),dtype=np.float64)
		# converting to hidden dimension
		self.Wzx = np.ones((h,d),dtype=np.float64)
		self.Wrx = np.ones((h,d),dtype=np.float64)
		self.Wx  = np.ones((h,d),dtype=np.float64)



		self.dWzh = np.zeros((h,h),dtype=np.float64)
		self.dWrh = np.zeros((h,h),dtype=np.float64)
		self.dWh  = np.zeros((h,h),dtype=np.float64)

		self.dWzx = np.zeros((d,h),dtype=np.float64)
		self.dWrx = np.zeros((d,h),dtype=np.float64)
		self.dWx  = np.zeros((d,h),dtype=np.float64)

		self.z_act = Sigmoid()
		self.r_act = Sigmoid()
		self.h_act = Tanh()

		
	def forward(self, x, h):
		print("forward called")
		# input:
		# 	- x: shape(input dim),  observation at current time-step
		# 	- h: shapTanhe(hidden dim), hidden-state at previous time-step
		# 
		# output:
		# 	- h_t: hidden state at current time-step
		self.x=np.array(x)
		self.h=np.array(h)
		# print("x,hshape",x,h)
		self.dx=np.zeros((1,len(x)))
		self.dh=np.zeros((1,len(h)))
		# # print("in forward",self.Wzh.shape,h.shape )
		# self.Zt=self.z_act(np.dot(self.Wzh,h)+np.dot(self.Wzx,x))
		# self.Rt=self.r_act(np.dot(self.Wrh,h)+np.dot(self.Wrx,x))
		# self.Hcapt=self.h_act(np.dot(self.Wh,self.Rt*h)+np.dot(self.Wx,x))
		# self.Ht=(1-self.Zt)*h+self.Zt*self.Hcapt
		# return self.Ht
		self.z=[[]for i in range(25)]
		self.z[0]=self.x
		self.z[1]=self.h
		self.z[2]=self.Wzh
		self.z[3]=self.Wrh
		self.z[4]=self.Wh
		self.z[5]=self.Wzx
		self.z[6]=self.Wrx
		self.z[7]=self.Wx
		# print(self.z)
		# print("z0", self.z[0])

		self.eqns=[
		[2,	"dot",	1],
		[5,	"dot",	0],
		[1+7,		"+",	2+7],
		[3+7, 		"sig", 	-1],

		[3,	"dot",	1],
		[6,	"dot",	0],
		[5+7,	"+",	6+7],
		[7+7, 	"sig", 	-1]	,
			
		[8+7, "*",	1],
		[4,	"dot",	9+7],
		[7,	"dot",	0],
		[10+7,"+",	11+7],
		[12+7,"tanh", -1],
		
		[-1,'-',4+7],
		[14+7,"*", 	1],
		[4+7, "*",	13+7],
		[15+7,"+",	16+7]
		]
		# print("no of eqns", len(self.eqns))
		for i in range(len(self.eqns)):

			self.z[i+8]=self.forw(self.eqns[i])
		for i in range(len(self.z)):
			self.z[i]=np.array(self.z[i])
		self.Zt=self.z[11]
		self.Rt=self.z[15]
		self.Hcapt=self.z[20]
		self.Ht=self.z[24]
		for i in range(8):
			print(i,"zi",self.z[i])
		
		return self.Ht

	def backward(self, delta):
		# print("del",delta)
		print("backward called")
		print("z")
		# for i in range(len(self.z)):
		# 	print(i,self.z[i])
		print("Zt",self.Zt)
		print("Rt",self.Rt)
		print("Hcapt",self.Hcapt)
		print("Zt",self.Ht)
		print("x",self.x)
		print("h",self.h)
		# input:
		# 	- delta: 	shape(hidden dim), summatiostn of derivative wrt loss from next layer at 
		# 			same time-step and derivative wrt loss from same layer at
		# 			next time-step
		#
		# output:
		# 	- dx: 	Derivative of loss wrt the input x
		# 	- dh: 	Derivative of loss wrt the input hidden h
		# print("x,hshapeb",self.x.shape,self.h.shape, self.Wzh.shape, self.Wrh.shape, self.Wh.shape, self.Wzx.shape, self.Wrx.shape, self.Wx.shape)
		

		self.dWzh.fill(0)
		self.dWrh.fill(0)
		self.dWh.fill(0)
		self.dWzx.fill(0)
		self.dWrx.fill(0)
		self.dWx.fill(0)
		self.dx.fill(0)
		self.dh.fill(0)
		self.dz=[np.array(self.z[i]) for i in range(25)]
		for i in range(len(self.dz)):
			if(len(self.dz[i].shape))==1:
				self.dz[i]=self.dz[i].reshape(len(self.dz[i]),-1)
				self.z[i]=self.z[i].reshape(len(self.z[i]),-1)
			self.dz[i]=np.transpose(self.dz[i])
			self.dz[i].fill(0)
			# print(type(self.dz[i]),self.dz[i].shape)
		# print(self.dz)
		self.dz[24]=np.array(delta)
		# print("del",delta)
		for i in range(len(self.eqns)-1,-1,-1):
			print(i)
			self.grad(i+8, self.eqns[i])
		print(self.dz)
		# dz=dz.fill(0)phpphp
		self.dx=self.dz[0]
		self.dh=self.dz[1]
		self.dWzh=self.dz[2]
		self.dWrh=self.dz[3]
		self.dWh=self.dz[4]
		self.dWzx=self.dz[5]
		self.dWrx=self.dz[6]
		self.dWx=self.dz[7]
		return self.dx,self.dh

	def forw(self,eqn):
		sig = Sigmoid()
		tanh=Tanh()
		op=eqn[1]
		# print("eqn",eqn)
		# print("shapes of eqn values",self.z[eqn[0]].shape,self.z[eqn[2]].shape)
		# print("values of eqn",self.z[eqn[0]],self.z[eqn[2]])
		if (op=='*'):
			return self.z[eqn[0]]*self.z[eqn[2]]
		elif(op=='dot'):
			return np.dot(self.z[eqn[0]],self.z[eqn[2]])
		elif(op=='+'):
			return self.z[eqn[0]]+self.z[eqn[2]]
		elif(op=="tanh"):
			return tanh(self.z[eqn[0]])
		elif(op=="sig"):
			return self.z_act (self.z[eqn[0]])
		elif (op=='-'):
			return 1-self.z[eqn[2]]

	def grad(self,dzi,eqn):
		# print(eqn)
		# print("dzi, eqn",dzi,eqn)
		# print(self.dz[dzi])
		# print(type(self.dz[dzi]))
		# print("shapes of eqn values",self.z[eqn[0]].shape,self.z[eqn[2]].shape)
		# print("values of eqn",self.z[eqn[0]],self.z[eqn[2]])
		# print("values of eqn",self.dz[eqn[0]],self.dz[eqn[2]])
		sig = Sigmoid()
		tanh=Tanh()
		xi,op,yi=eqn
		x=self.z[xi]
		y=self.z[yi]
		# print("dx",self.dz[xi].shape)
		# print("dy",self.dz[xi].shape)

		# print("dz",self.dz[dzi])
		# print("x",x)
		# print("y",y)

		if (op=='*'):
			self.dz[xi]+= self.dz[dzi]*np.transpose(x)
			self.dz[yi]+= self.dz[dzi]*np.transpose(y)
		if(op=='dot'):
			self.dz[xi]+=np.dot(y,self.dz[dzi])
			self.dz[yi]+=np.dot(self.dz[dzi],x)
		if(op=='+'):
			self.dz[xi]+=self.dz[dzi]
			self.dz[yi]+= self.dz[dzi]
		if(op=="tanh"):
			self.dz[xi]+=self.dz[dzi]*(np.transpose(1-tanh(x)*tanh(x)))
		if(op=="sig"):
			self.dz[xi]+=self.dz[dzi]*np.transpose(sig(x))*np.transpose(1-sig(x))
		if (op=='-'):
			self.dz[yi]+= -self.dz[dzi]
		print("dz operated",self.dz[xi],self.dz[yi])


if __name__ == '__main__':
	# test()
	# h=[-0.15323616,-2.43250851,0.50798434,-0.32403233,-1.51107661,-0.87142207,-0.86482994,0.60874908,0.5616381,1.51475038,0.64792481,-1.35164939,-1.40920928,1.13072535,1.5666862,-0.2377481,0.55880299,-1.50489128,-1.94392176,-1.17402368,-0.35718753,-0.52137639,-0.23011406,-0.49101443,0.67930114,1.42754695,0.03619746,2.02999749,-0.63440471,-0.52510339,0.38773466,-0.35479876,1.17705226,-0.64110782,1.32269399,0.19417502,2.56545278,-0.46411491,-0.20269391,0.14565182,-2.18102797,0.60226513,0.48084611,0.10931836,-1.54439578,-1.54656104,0.58661852,1.17517869,1.59446463,-0.89544152,-1.03079803,-0.2719388,-1.97573014,-0.58893118,0.85178964,1.6346025,0.27915545,1.64055365,0.41087294,0.19136392,-0.17144119,0.18693705,-0.25485295,-0.14091075,-0.66189183,0.2590319,0.01444842,-1.47958003,-0.2407005,-0.85567139,-2.04820046,0.48388365,1.55868825,2.36973019,1.56241953,-0.87080155,1.17524499,1.119899,-1.98782953,0.86128852,0.62717704,0.16280825,0.28861672,0.05830738,1.63193585,-0.40178883,-0.19993939,0.00738898,0.27566408,-1.7632498,1.38797381,0.22619976,0.5691246,0.19731599,-0.18644127,-0.35524151,0.09611414,0.15205234,1.15526176,0.34605775,-0.13348867,1.98656511,-1.27942616,-1.34020918,0.35460205,-0.21237329,-1.77459599,-0.31222966,-0.71065577,1.1311286,-0.62125177,1.05061465,0.4597817,-0.20633091,0.02117183,0.42865874,-2.30803851,0.32706841,-0.37911961,1.79791937,-0.69126896,1.14256392,-2.51492462,0.81462501,0.27610275,-0.24701649,-0.12088931,-0.26056059,0.42300321,-0.13424856,-1.78773771,-0.18581086,2.23472174,0.0468462,0.29078795,-0.43805451,0.17405447,0.17794556,-0.26120192,0.8632634,-0.92307796,-0.13019521,0.50505375,-0.26700418,-1.22387965,0.55826422,-0.98216096,-0.44730816,-0.82814759,-0.11072841]
	# x=[1.62434536,-0.61175641,-0.52817175,-1.07296862,0.86540763,-2.3015387,1.74481176,-0.7612069,0.3190391,-0.24937038,1.46210794,-2.06014071,-0.3224172,-0.38405435,1.13376944,-1.09989127,-0.17242821,-0.87785842,0.04221375,0.58281521,-1.10061918,1.14472371,0.90159072,0.50249434,0.90085595,-0.68372786,-0.12289023,-0.93576943,-0.26788808,0.53035547,-0.69166075,-0.39675353,-0.6871727,-0.84520564,-0.67124613,-0.0126646,-1.11731035,0.2344157,1.65980218,0.74204416,-0.19183555,-0.88762896,-0.74715829,1.6924546,0.05080775,-0.63699565,0.19091548,2.10025514,0.12015895,0.61720311,0.30017032,-0.35224985,-1.1425182,-0.34934272,-0.20889423,0.58662319,0.83898341,0.93110208,0.28558733,0.88514116,-0.75439794,1.25286816,0.51292982,-0.29809284,0.48851815,-0.07557171,1.13162939,1.51981682,2.18557541,-1.39649634,-1.44411381,-0.50446586,0.16003707,0.87616892,0.31563495,-2.02220122,-0.30620401,0.82797464,0.23009474,0.76201118,-0.22232814,-0.20075807,0.18656139,0.41005165,0.19829972,0.11900865,-0.67066229,0.37756379,0.12182127,1.12948391,1.19891788,0.18515642,-0.37528495,-0.63873041,0.42349435,0.07734007,-0.34385368,0.04359686,-0.62000084,0.69803203]
	
	# x=[ 1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763]
	# h=[ 0.30017032, -0.35224985]
	# delta=[[ 0.52057634, -1.14434139]]
	# Zt= [ 0.57024792,  0.30974427]
	# Rt= [ 0.75880065,  0.80351915]
	# Hcapt= [-0.21899591,  0.78356533]
	# Zt= [ 0.00411685, -0.0004376 ]
	x=[ 1.0,1.0,1.0,2.0,2.0]
	h=[ 1.0 ,2.0 ]
	delta=[[1.0,2.0]]
	Zt=[1.0,2.0]
	Rt=[1.0,2.0]
	Hcapt= [1.0,2.0]
	gru=GRU_Cell(5,2)
	gru.forward(x,h)
	gru.backward(delta)

	#