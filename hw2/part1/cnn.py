from layers import *

class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        #in_ch,out_ch, k_s, stride
        self.layers = [Conv1D(24,8,8,4),ReLU(),Conv1D(8,16,1,1),ReLU(),Conv1D(16,4,1,1),Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        w=weights
        self.layers[0].W = w[0].T.reshape(8,8,24).transpose(0,2,1)
        self.layers[2].W = w[1].T.reshape(16,8,1)
        self.layers[4].W = w[2].T.reshape(4,16,1)
        

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta




class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        self.dim=[[24,2,2,2],[2,8,2,2],[8,4,2,1]]
        self.layers = [Conv1D(*(self.dim[0])),ReLU(),Conv1D(*(self.dim[1])),ReLU(),Conv1D(*(self.dim[2])),Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        w=weights
        # for i in w:
        #     for j in range(len(i)):
        #         print(j,i[j])
        #     print(i.shape)
        # print(w[0],w[0].shape)
        #               0,i*kw .T
        for i in range(len(w)):
            #chop for output depth
            #traspose
            # print("init",w[i])
            # print(w[i].shape) 
            w[i]=w[i].T
            # print("Transposed",w[i])
            w[i]=w[i][:][0:self.dim[i][1]]
            # print(w[i].shape)
            # print("c1",w[i])

            #reshape for input size
            #print(type(w[i]),type(len(w[i][0])),type(self.dim[i][0]))
            w[i]=w[i].reshape(len(w[i]),-1,self.dim[i][0])

            # print("r",w[i].shape)#o,k',i
            # print("c1",w[i])

            #transpose
            w[i]=w[i].transpose(1,0,2)
            # print("t",w[i].shape)#k',o,i
            # print("t2",w[i])

            #chop for kernal size
            # w[i]=w[i].T
            w[i]=w[i][0:self.dim[i][2]]#k.o.i
            w[i]=w[i].transpose(1,2,0)
            # print("c2shape",w[i].shape)
            # print("wend",w[i])
        # w0=(w[0].T)[0:2] #,2,192
        # w1=(w[1].T)[0:8] #,8,8
        # w2=w[2].T         #,4,8
        # print(w0.shape)
        #                 #o,,i
        # w0=w0.reshape(2,8,24)
        # w1=w1.reshape(8,4,2)
        # w2=w2.reshape(4,2,8)
        #               o,k,i  
        # w0=w0[:][0:2][:]#2,2,24
        # w1=w1[:][0:2][:]#8,2,2
        # w2=w2[:][:][:]  #4,2,8
        #                               o,i,k
        # print("heheha")
        # w0= w0.transpose(1,0,2)
        # w0=w0[0:2] 
        # print((w0)[:][:][0])
        self.layers[0].W =w[0]
        self.layers[2].W =w[1]
        self.layers[4].W =w[2]

        # print(self.layers[0].W.shape,self.layers[0].W)
        # print(self.layers[2].W.shape,self.layers[2].W)
        # print(self.layers[4].W.shape,self.layers[4].W)

        # self.layers[0].W = w[0].T.reshape(8,8,24).transpose(0,2,1)
        # self.layers[2].W = w[1].T.reshape(16,8,1)
        # self.layers[4].W = w[2].T.reshape(4,16,1)

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
