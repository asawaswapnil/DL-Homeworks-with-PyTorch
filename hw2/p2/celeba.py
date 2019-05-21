import numpy as np
data = np.genfromtxt("list_attr_celeba.csv", delimiter=',', names=True)
print(data[0:5])