import numpy as np 
import matplotlib.pyplot as plt

from GaussianProcess import *
from genData import *

f = genRandomFunction()
X, y = genDataFromFunction(f)

params = {'sigma_n':.0001, 'sigma_s':1, 'kernel_width':1}
kernel = genGaussianKernel(params['sigma_s'],params['kernel_width'])
mean = lambda x: 0
GP = GaussianProcess(mean, kernel, **params)
GP.train(X, y)

x = np.linspace(np.min(X),np.max(X))
x = np.reshape(x,(1,-1))

y_expect = [GP.eval_mean(x[:,i]) for i in range(x.shape[1])]

plt.plot(x[0,:].flatten(),f(x))
plt.plot(X[0,:],y,'.')

plt.plot(x[0,:],y_expect,'--')
plt.show()