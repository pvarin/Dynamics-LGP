import numpy as np

class GaussianKernel(object):
    def __init__(self, params):
        self.params_map = ['sigma_s', 'width']
        self.set_params(params)

    def set_params(self, params):
        self.params = params
        self.sigma_s = params[0]
        self.width = params[1]

    def get_params_map(self, params):
        return ['sigma_s',
                'width']

    def eval(self, x_p, x_q):
        return self.sigma_s*np.exp(-.5*(x_p-x_q).T.dot(self.width).dot(x_p-x_q))

    def eval_grad(self, x_p, x_q):
        g = [np.exp(-.5*(x_p-x_q).T.dot(self.width).dot(x_p-x_q)), # partial wrt sigma_s
             self.sigma_s*np.exp(-.5*(x_p-x_q).T.dot(self.width).dot(x_p-x_q))*(-.5*(x_p-x_q).T.dot(x_p-x_q))]
        return g

def genGaussianKernel(sigma_s, width):
        def kernel(x_p,x_q):
                return sigma_s*np.exp(-.5*(x_p-x_q).T.dot(width).dot(x_p-x_q))
        
        return kernel