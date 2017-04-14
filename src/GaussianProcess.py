import numpy as np

def genGaussianKernel(sigma_s, width):
    def kernel(x_p,x_q):
        return sigma_s*np.exp(-.5*(x_p-x_q).T.dot(width).dot(x_p-x_q))
    
    return kernel

class GaussianProcess:
    def __init__(self, mean=None, kernel=None, **kwargs):
        
        # handle no input params and set defaults
        if kwargs is None:
            kwargs = dict()
        self.sigma_n = kwargs.get('sigma_n',1)
        self.sigma_s = kwargs.get('sigma_s',1)
        self.width = kwargs.get('width',1)

        # set default kernel to be the gaussian kernel
        self.kernel = kernel or genGaussianKernel(self.sigma_s, self.width)

        # set default mean to be the zero function
        self.mean = mean or (lambda x: 0)
        
        # indicate that the model hasn't been trained yet
        self._clean = False

    def train(self, X=None, y=None):
        N = X.shape[1]
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y

        N = X.shape[1]
        self.K = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                self.K[i,j] = self.kernel(X[:,i],X[:,j])
                self.K[j,i] = self.K[i,j]

        self.alpha = np.linalg.solve(self.K + self.sigma_n*np.eye(self.K.shape[0]),y)
        self._clean = True

    def update(self, x, y):
        self.X = np.hstack([self.X,x])
        self.y = np.hstack([self.y,y])
        self._clean = False

    def eval_mean(self, x, k=None):
        if not self._clean:
            self.train()
        
        if k is None:
            k = self.get_k(x)

        return k.dot(self.alpha)

    def get_k(self,x):
        k = np.array([self.kernel(self.X[:,i],x) for i in range(self.X.shape[1])])
        return k

    def eval_var(self, x, k=None):
        if not self._clean:
            self.train()

        if k is None:
            k = self.get_k(x)

        A = np.linalg.solve(self.K + self.params['sigma_n']*np.eye(self.K.shape),k)
        var = self.kernel(x,x) - k.dot(A)
        return var

    def eval(self, x):
        k = self.get_k(x)
        return self.eval_mean(x,k=k), self.eval_var(x,k=k)