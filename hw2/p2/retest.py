import os
path = 'test_classification/medium/'
for filename in os.listdir(path):
    new_name = '0'*(8-len(filename)) + filename
    os.rename(path+filename, path+new_name)