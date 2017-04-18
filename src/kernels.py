import numpy as np

def genGaussianKernel(sigma_s, width):
    def kernel(x_p,x_q):
        return sigma_s*np.exp(-.5*(x_p-x_q).T.dot(width).dot(x_p-x_q))
    
    return kernel