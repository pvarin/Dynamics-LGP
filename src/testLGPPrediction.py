import numpy as np 
import matplotlib.pyplot as plt

from LocalGaussianProcess import LGPCollection
from kernels import genGaussianKernel
from genData import *

# 1 dimension
f = genRandomFunction()
X, y = genDataFromFunction(f, N=1000)

params = {'sigma_n':.0001, 'sigma_s':1, 'width':100}
LGP = LGPCollection(.98, 100, max_models=3, **params)
LGP.train(X, y)

x = np.linspace(np.min(X),np.max(X))
x = np.reshape(x,(1,-1))

y_expect = [LGP.eval_mean(x[:,i]) for i in range(x.shape[1])]

y_sub_model = [[m.eval_mean(x[:,i]) for i in range(x.shape[1])] for m in LGP.models]
y_sub_model = np.array(y_sub_model).T

plt.figure()
plt.plot(x[0,:].flatten(),f(x)) # true function
plt.plot(X[0,:],y,'.')          # noisy data
for m in LGP.models:
    plt.plot(m.X[0,:],m.y,'o',mfc='none')
plt.plot(x[0,:],y_sub_model)
plt.plot(x[0,:],y_expect,'k--')  # estimated from the GP
plt.show()


# # two dimensions
# dim = 2
# f = genRandomFunction(dim)
# X, y = genDataFromFunction(f, dim=dim, N=1000)

# x_coord, y_coord = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
# x_eval = np.vstack([np.reshape(x_coord,(1,-1)), np.reshape(y_coord,(1,-1))])
# f_eval = np.reshape(f(x_eval), x_coord.shape)

# LGP = LGPCollection(.98,100)
# LGP.train(X,y)
# y_expect = [LGP.eval_mean(x_eval[:,i]) for i in range(x_eval.shape[1])]
# y_expect = np.reshape(y_expect,x_coord.shape)
# v_max = max(np.max(y), -np.min(y))

# # plot
# color_options = {'cmap':'RdBu', 'vmin':-v_max, 'vmax':v_max}

# plt.figure()
# plt.subplot(4,1,1)
# plt.pcolor(x_coord, y_coord, f_eval, **color_options)
# plt.title('Original Function')

# plt.subplot(4,1,2)
# plt.scatter(X[0,:], X[1,:], c=y, **color_options)
# plt.title('Data')

# plt.subplot(4,1,3)
# plt.title('LGP Estimation')
# plt.pcolor(x_coord, y_coord, y_expect, **color_options)

# plt.subplot(4,1,4)
# plt.pcolor(x_coord, y_coord, -f_eval+y_expect, **color_options)
# plt.title('Residual')

# plt.show()