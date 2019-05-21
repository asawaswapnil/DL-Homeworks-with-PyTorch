import numpy as np
import os
import time
import torch    
    


def sumproducts(x, y):
    """
    x is a 1-dimensional int numpy array.
    y is a 1-dimensional int numpy array.
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    result = 0
    for i in range(len(x)):
        for j in range(len(y)):
            result += x[i] * y[j]
    return result




def vectorize_sumproducts(x, y):
    """
    x is a 1-dimensional int numpy array. Shape of x is (N, ).
    y is a 1-dimensional int numpy array. Shape of y is (N, ).
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> vectorize_sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    # Write the vecotrized version here
    return (x*y).sum()
    pass


def Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """	
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
    return result

def vectorize_Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    result=np.copy(x)
    result[result<0]=0
    return result
    pass


def ReluPrime(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
            else:
                result[i][j] = 1
    return result


def vectorize_PrimeRelu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    result=np.copy(x)
    result=np.where(x<0,0,1)
    return result
    pass  


def slice_fixed_point(x, l, start_point):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should have.
    start_point is an integer representing the point at which the final utterance should start in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)
    
    """
    result=[instance[start_point:start_point+l] for instance in x]
    return np.array(result)
    pass


def slice_last_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    result=[instance[len(instance)-l:len(instance)] for instance in x]
    return np.array(result)
    pass


def slice_random_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    result=[[] for i in range(len(x))]
    for i in range(len(x)):
        r=np.random.randint(0,high=len(x[i])-l+1)
        result[i]=x[i][r:r+l] 
    return np.array(result)
    pass


def pad_pattern_end(x):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    result=[[] for i in range(len(x))]
    maxlen=0
    for instance in x:
        l=len(instance)
        maxlen=l if l>maxlen else maxlen
    for instanceN in range(len(x)):
        l=len(x[instanceN])
        shap=np.shape(x[instanceN])
        pad_with=[(0,0) for i in range(len(shap))]
        pad_with=[(0,maxlen-l),(0,0)]
        result[instanceN]=np.pad(x[instanceN],pad_with,'symmetric')
        result2=np.array(result)
    return(result2)
    pass
    


def pad_constant_central(x, c_):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.
    
    # """
    result=[[] for i in range(len(x))]
    maxlen=0
    for instance in x:
        l=len(instance)
        maxlen=l if l>maxlen else maxlen
    for instanceN in range(len(x)):
        l=len(x[instanceN])
        shap=np.shape(x[instanceN])
        diff=maxlen-l
        pad_with=[(0,0) for i in range(len(shap))]
        pad_with[0]=(int(diff/2),diff-int(diff/2))
        result[instanceN]=np.pad(x[instanceN],pad_with,'constant',constant_values=(c_,c_))
    return np.array(result)
    pass
    # result=np.copy(x)
    # maxlen=0
    # for instance in x:
    #     l=len(instance)
    #     maxlen=l if l>maxlen else maxlen
    # for instanceNumber in range(len(result)):
    #     l=len(result[instanceNumber])
    #     diff=maxlen-l
    #     #print(int((maxlen-l)/2),maxlen-l-int((maxlen-l)/2))
    #     result[instanceNumber]=np.pad(result[instanceNumber],(int(diff/2),diff-int(diff/2)),'constant',constant_values=(c_,c_))
    # return np.array(result)
    #pass



def numpy2tensor(x):
    """
    x is an numpy nd-array. 

    Return a pytorch Tensor of the same shape containing the same data.
    """
    x2=torch.LongTensor(x)
    return x2
    pass

def tensor2numpy(x):
    """
    x is a pytorch Tensor. 

    Return a numpy nd-array of the same shape containing the same data.
    """
    return x.numpy()
    pass

def tensor_sumproducts(x,y):
    """
    x is an n-dimensional pytorch Tensor.
    y is an n-dimensional pytorch Tensor.

    Return the sum of the element-wise product of the two tensors.
    """
    result=(x*y).sum()
    return (result)    
    pass

def tensor_ReLU(x):
    """
    x is a pytorch Tensor. 
    For every element i in x, apply the ReLU function: 
    RELU(i) = 0 if i < 0 else i

    Return a pytorch Tensor of the same shape as x containing RELU(x)
    """
    result=x.clone()
    #result=torch.nn.functional.relu(result)
    result[result<0]=0
    return (result)
    pass        

def tensor_ReLU_prime(x):
    """
    x is a pytorch Tensor. 
    For every element i in x, apply the RELU_PRIME function: 
    RELU_PRIME(i) = 0 if i < 0 else 1

    Return a pytorch Tensor of the same shape as x containing RELU_PRIME(x)
    """
    result=x.clone()
    result[result>=0]=1
    result[result<0]=0
    return (result)
    pass


# if __name__ == '__main__':
#     test = (np.array([np.array([[11, 12, 13],
#         [21, 22, 23],
#         [31, 32, 33],
#         [41, 42, 43],
#         [51, 52, 53],
#         [61, 62, 63],
#         [71, 72, 73],
#         [81, 82, 83]]),
#         np.array([[111, 112, 113],
#         [121, 122, 123],
#         [131, 132, 133],
#         [131, 132, 133],
#         [141, 142, 143]]),
#         np.array([[151, 152, 153],
#         [161, 162, 163],
#         [171, 172, 173],
#         [181, 182, 183]]),
#         np.array([[191, 192, 193],
#         [101, 102, 103],
# [201, 202, 203]])], dtype=object))

#     #test2 =  np.array([[0,1],[0,1,3],[1,2,3,4]])
#     print(pad_pattern_end(test))
#     #print(pad_pattern_end(test2))
