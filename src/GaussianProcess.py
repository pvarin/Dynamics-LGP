import numpy as np
from kernels import GaussianKernel

class GaussianProcess(object):
    def __init__(self, mean=None, kernel=None, sigma_n=0.001):
        
        # set defaults
        self.sigma_n = sigma_n
        self.kernel = kernel or GaussianKernel([1.0,1.0])
        self.mean = mean or (lambda x: 0)
        
        # indicate that the model hasn't been trained yet
        self._clean = False

    def train(self, X=None, y=None):
        if self._clean:
            return 

        if X is not None:
            self.X = X
        if y is not None:
            self.y = y

        N = self.X.shape[1]
        self.K = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                self.K[i,j] = self.kernel.eval(self.X[:,i],self.X[:,j])
                self.K[j,i] = self.K[i,j]

        self.inv_K = np.linalg.inv(self.K)

        self.alpha = np.linalg.solve(self.K + self.sigma_n*np.eye(self.K.shape[0]),self.y)
        self._clean = True

    def update(self, x, y):
        self.X = np.hstack([self.X,x[...,np.newaxis]])
        self.y = np.hstack([self.y,y])
        self._clean = False

    def eval_mean(self, x, k=None):
        if not self._clean:
            self.train()
        
        if k is None:
            k = self.get_k(x)

        return k.dot(self.alpha)

    def get_k(self,x):
        k = np.array([self.kernel.eval(self.X[:,i],x) for i in range(self.X.shape[1])])
        return k

    def eval_var(self, x, k=None):
        if not self._clean:
            self.train()

        if k is None:
            k = self.get_k(x)

        A = np.linalg.solve(self.K + self.sigma_n*np.eye(self.K.shape[0]),k)
        var = self.kernel.eval(x,x) - k.dot(A)
        return var

    def eval(self, x):
        k = self.get_k(x)
        return self.eval_mean(x,k=k), self.eval_var(x,k=k)

    def drop_data(self):
        # TODO: drop a data point to minimize information loss
        N = self.X.shape[1]
        idx = np.random.randint(0,N)
        self.X = np.delete(self.X,idx,1)
        self.y = np.delete(self.y,idx)

    def get_loglikelihood(self):
        # make sure the model is trained
        if not self._clean:
            self.train()

        # compute the log-likelihood

        sign, logdet = np.linalg.slogdet((self.K + self.sigma_n*np.eye(self.K.shape[0])))
        if sign < 0:
            raise RuntimeError('Covariance has negative determinant')

        ll = -.5*self.y.T.dot(self.alpha)
        ll += -.5*logdet
        ll += -.5*self.y.shape[0]*np.log(2*np.pi)

        return ll

    def get_cov_deriv(self):
        N = self.y.shape[0]
        N_params = len(self.kernel.params)
        dK = [np.zeros((N,N)) for _ in range(N_params)]
        for i in range(N):
            for j in range(i,N):
                for k, dp in enumerate(self.kernel.eval_grad(self.X[:,i],self.X[:,j])):
                    dK[k][i,j] = dp
                    dK[k][j,i] = dp
        return dK

    def get_param_grad(self):
        dK = self.get_cov_deriv()
        grad = np.zeros(len(dK))

        for i in range(len(dK)):
            grad[i] = .5*np.trace((self.alpha[...,np.newaxis].dot(self.alpha[...,np.newaxis].T) - self.inv_K).dot(dK[i]))

        return grad

    def optimize_hyperparameters_gradient(self):
        # use gradient descent to optimize the hyperparameters
        params = self.kernel.params
        ll = self.get_loglikelihood()
        grad = np.inf
        while (np.linalg.norm(grad) > 0.001):
            grad = self.get_param_grad()
            stepsize = min(1.0, 10.0/np.linalg.norm(grad)) #don't move more than 10 units away in any step
            while True:
                # choose the stepsize
                new_params = params + stepsize*grad

                if (new_params < 0).any(): # don't let any of the parameters go below zero
                    stepsize *= 0.1
                    continue

                self.kernel.set_params(new_params)
                self._clean = False
                self.train()
                try:
                    new_ll = self.get_loglikelihood()
                except:
                    stepsize *= 0.1
                    continue

                if new_ll > ll:
                    params = new_params
                    ll = new_ll
                    break
                else:
                    if stepsize < 1e-20:
                        return
                    stepsize *= 0.1
