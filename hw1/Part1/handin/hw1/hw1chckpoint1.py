"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
    	self.state=x  
    	self.exp=np.exp(x)
    	self.state=self.exp/(self.exp+1)
    	return self.state
        # Might we need to store something before returning?
        

    def derivative(self):
    	
        return self.state-self.state**2
        # Maybe something we need later in here...



class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):   
        self.state= (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return self.state

    def derivative(self):
        return 1-self.state*self.state


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state=x
        self.state[self.state<0]=0
        return self.state
    def derivative(self):
        return (self.state>0).astype(float)


# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        M=np.amax(x, axis=1,keepdims=True)
        self.x1=x-M
        self.exp=np.exp(self.x1)
        self.sm=np.sum(self.exp,axis=1,keepdims=True)
        self.sm=np.array(self.sm)
        self.sig=self.exp/self.sm
        self.logsig=np.log(self.sig)
        self.ce=np.sum(y*self.logsig, axis=1)
        self.ce=-self.ce
        return self.ce


    def derivative(self):
        return( self.sig-self.labels)

        


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        # if eval:
        #    # ???

        self.x = x

        # self.mean = # ???
        # self.var = # ???
        # self.norm = # ???
        # self.out = # ???

        # update running batch statistics
        # self.running_mean = # ???
        # self.running_var = # ???

        # ...

        raise NotImplemented

    def backward(self, delta):

        raise NotImplemented


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    raise NotImplemented


def zeros_bias_init(d):
    raise NotImplemented

# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.randn(d0,d1)

def zeros_bias_init(d):
    return np.zeros((1,d))


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.ih=hiddens
        self.ih.insert(0,input_size)
        self.ih.append(output_size)
        #print("hi",self.ih)
        self.W = np.array([weight_init_fn(self.ih[i],self.ih[i+1]) for i in range(len(self.ih)-1)])
        self.dW = np.array([np.zeros((self.ih[i],self.ih[i+1])) for i in range(len(self.ih)-1)])
        self.b = np.array([zeros_bias_init(self.ih[i]) for i in range(1,len(self.ih))])
        self.db = np.array([np.zeros(self.ih[i]) for i in range(1,len(self.ih))])
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        self.x=x
        self.batch_size=len(x)
        self.Z1=np.dot(x,self.W[0])+self.b[0]
        self.Z1=self.activations[0].forward(self.Z1)
        return self.Z1

    def zero_grads(self):
        self.dW.fill(0)
        self.db.fill(0)

    def step(self):
        self.W[0]=self.W[0]-self.lr*self.dW[0]
        self.b[0]=self.b[0]-self.lr*self.db[0]

    def backward(self, labels):
        self.criterion.forward(self.Z1, labels)
        self.dLbyDy=self.criterion.derivative()
        #print("t1",t1.shape)
        self.theX=self.x.T 
        #print(t2.shape)
        #=(t2@t1)
        #print(t3.shape)
        #print(self.dW)
        self.dW[0]=np.dot(self.theX,self.dLbyDy)/self.batch_size
        self.db[0]=np.sum(self.dLbyDy,axis=0)/20
        #print(self.db[0])
        

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainX, trainY = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainX))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainX), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # val ...

        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):

        pass  # Remove this line when you start implementing this
        # Test ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented


sce=SoftmaxCrossEntropy()

#sce.forward([[0,2,3,4],[3,1,2,3]],[[0,0,0,1],[1,0,0,0]])