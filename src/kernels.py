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
    
    def eval_batch(self, X_p, X_q):
        Np = X_p.shape[1]
        Nq = X_q.shape[1]

        K = np.zeros((Np, Nq))
        for i in range(Np):
            for j in range(Nq):
                K[i,j] = self.eval(X_p[:,i], X_q[:,j])
        return K

    def eval_batch_symm(self, X):
        N = X.shape[1]

        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                k = self.eval(X[:,i], X[:,j])
                K[i,j] = k
                K[j,i] = k

        return K

    def eval_grad(self, x_p, x_q):
        g = [np.exp(-.5*(x_p-x_q).T.dot(self.width).dot(x_p-x_q)), # partial wrt sigma_s
             self.sigma_s*np.exp(-.5*(x_p-x_q).T.dot(self.width).dot(x_p-x_q))*(-.5*(x_p-x_q).T.dot(x_p-x_q))]
        return g

    def eval_grad_batch_symm(self, X):
        N = X.shape[1]
        N_params = len(self.params)

        dK = [np.zeros((N,N)) for _ in range(N_params)]
        for i in range(N):
            for j in range(i,N):
                for k, dp in enumerate(self.eval_grad(X[:,i],X[:,j])):
                    dK[k][i,j] = dp
                    dK[k][j,i] = dp
        return dK
