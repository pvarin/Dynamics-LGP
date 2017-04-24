import numpy as np 
import matplotlib.pyplot as plt

from GaussianProcess import GaussianProcess
from kernels import GaussianKernel
from genData import *

def test_1d():
    # 1 dimension
    f = genRandomFunction()
    X, y = genDataFromFunction(f, N=100)

    params = {'sigma_n':.01, 'sigma_s':1.0, 'width':10.0}
    kernel = GaussianKernel([params['sigma_s'],params['width']])
    mean = lambda x: 0
    GP = GaussianProcess(mean, kernel, sigma_n=params['sigma_n'])
    GP.train(X=X, y=y)
    GP.optimize_hyperparameters_grid_search()

    x = np.linspace(np.min(X),np.max(X))
    x = np.reshape(x,(1,-1))

    y_expect = np.array([GP.eval_mean(x[:,i]) for i in range(x.shape[1])])
    y_var = np.array([GP.eval_var(x[:,i]) for i in range(x.shape[1])])
    y_std = np.sqrt(y_var)

    plt.figure()
    true, = plt.plot(x[0,:].flatten(), f(x), color='black',label='True Function') # true function
    data, = plt.plot(X[0,:], y, '.',color='orange',label='Data') # noisy data
    mean, = plt.plot(x[0,:], y_expect, '--',color='blue',label='Estimated Mean') # estimated from the GP
    plt.fill_between(x[0,:], y_expect + 2*y_std, y_expect - 2*y_std, color='gray', linewidth=0.0, alpha=0.5)
    plt.legend(handles = [true, data, mean])

def test_2d():
    # two dimensions
    dim = 2
    f = genRandomFunction(dim)
    X, y = genDataFromFunction(f, dim=dim, N=1000)

    x_coord, y_coord = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
    x_eval = np.vstack([np.reshape(x_coord,(1,-1)), np.reshape(y_coord,(1,-1))])
    f_eval = np.reshape(f(x_eval), x_coord.shape)

    mean = lambda x: 0
    params = {'sigma_n':.01, 'sigma_s':1.0, 'width':10.0}
    kernel = GaussianKernel([params['sigma_s'],params['width']])
    GP = GaussianProcess(mean, kernel)
    GP.train(X,y)
    GP.optimize_hyperparameters_random_search()

    # evaluate
    y_expect = [GP.eval_mean(x_eval[:,i]) for i in range(x_eval.shape[1])]
    y_expect = np.reshape(y_expect,x_coord.shape)
    v_max = max(np.max(y), -np.min(y))

    # plot
    color_options = {'cmap':'RdBu', 'vmin':-v_max, 'vmax':v_max}

    plt.figure()
    plt.subplot(4,1,1)
    plt.pcolor(x_coord, y_coord, f_eval, **color_options)
    plt.title('Original Function')

    plt.subplot(4,1,2)
    plt.scatter(X[0,:], X[1,:], c=y, **color_options)
    plt.title('Data')

    plt.subplot(4,1,3)
    plt.title('GP Estimation')
    plt.pcolor(x_coord, y_coord, y_expect, **color_options)

    plt.subplot(4,1,4)
    plt.pcolor(x_coord, y_coord, -f_eval+y_expect, **color_options)
    plt.title('Residual')

if __name__ == '__main__':
    test_1d()
    # test_2d()
    plt.show()