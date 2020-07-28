#Loading the datasets

#from sklearn.datasets import datasets
#iris = load_iris()

#'BUNCH' Attributes
#print(type(iris))

#print(iris['target_names'])
#print(iris['data'])
#print(iris['target'])
#print(iris['DESCR'])
#print(iris['feature_names'])

#Dataset Object
#print(iris)

#Import
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

fig = plt.figure()
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)

iris = load_iris()
data = np.array(iris['data'])
target = np.array(iris['target'])

cd = {0:'r',1:'b',2:'g'}
cols = np.array([cd[target] for target in target])

ax1.scatter(data[:,2], data[:,1], c=cols)
ax2.scatter(data[:,2], data[:,0], c=cols)

plt.show()
