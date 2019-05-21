import numpy as np
import os

class WSJ():
    """ Load the WSJ speech dataset
        
        Ensure WSJ_PATH is path to directory containing 
        all data files (.npy) provided on Kaggle.
        
        Example usage:
            loader = WSJ()
            trainX, trainY = loader.train
            assert(trainX.shape[0] == 24590)
            
    """
    os.environ['WSJ_PATH']="../data/"
    def __init__(self):
        self.val_set = None
        self.train_set = None
        self.test_set = None
  
    @property
    def dev(self):
        if self.val_set is None:
            self.val_set = load_raw(os.environ['WSJ_PATH'], 'dev')
        return self.val_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(os.environ['WSJ_PATH'], 'train')
        return self.train_set
  
    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['test'], 'test.npy'), encoding='bytes'), None)
        return self.test_set
    
def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'), 
        np.load(os.path.join(path, '{}_transcripts.npy'.format(name)))
    )