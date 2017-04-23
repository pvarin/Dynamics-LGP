import numpy as np
from kernels import GaussianKernel

class GaussianProcess(object):
    def __init__(self, mean=None, kernel=None, sigma_n=0.001):
        
        # set defaults
        self.sigma_n = sigma_n
        self.kernel = kernel or GaussianKernel([1.0,1.0])
        self.mean = mean or (lambda x: 0)

    #################################
    ## Training and update methods ##
    #################################

    def train(self, X=None, y=None):
        '''
        This should only be called to overwrite all of the the data or the hyperparameters have changed,
        otherwise use the update method
        '''
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y

        # compute K
        self.K = self.kernel.eval_batch_symm(self.X)
        K_y = self.K + self.sigma_n*np.eye(self.K.shape[0])
        self.inv_K = np.linalg.inv(K_y)
        self.alpha = np.linalg.solve(K_y,self.y)

    def update(self, x, y):
        '''
        Efficiently updates the model with a single datapoint
        '''
        k = self.get_k(x) 
        self.X = np.hstack([self.X,x[...,np.newaxis]])
        self.y = np.hstack([self.y,y])
        self.update_K(x,k=k)
    def update_K(self, x,k=None):
        '''
        Appends a new row and column to K when adding a single datapoint and updates all of the downstream values
        '''
        if k is None:
            k = self.get_k(x)

        k = k[...,np.newaxis]

        k_ = self.kernel.eval(x,x)
        self.K = np.vstack([np.hstack([self.K, k]), np.hstack([k.T, [[k_]]])])

        K_y = self.K + self.sigma_n*np.eye(self.K.shape[0])
        self.inv_K = np.linalg.inv(K_y) # TODO, we could make this more efficient with a rank-one update
        self.alpha = np.linalg.solve(K_y,self.y)


    def get_k(self,x):
        '''
        Helper method to get the covariance vector between a single data point and the
        training set.

        Returns a column vector (Nx1)
        '''
        k = self.kernel.eval_batch(self.X,x[...,np.newaxis])[:,0]
        return k

    #########################
    ## Evalutation methods ##
    #########################

    def eval_mean(self, x, k=None):
        '''
        Evaluate the predicted mean at the single point x. Reuses k if it is precomputed
        '''
        if k is None:
            k = self.get_k(x)

        return k.dot(self.alpha)

    def eval_var(self, x, k=None):
        '''
        Evaluates the variance at the single point x. Reuses k if it is precomputed
        '''
        if k is None:
            k = self.get_k(x)

        A = np.linalg.solve(self.K + self.sigma_n*np.eye(self.K.shape[0]),k)
        var = self.kernel.eval(x,x) - k.dot(A)
        return var

    def eval(self, x):
        '''
        Evaluates the mean and variance at the single point k
        '''
        k = self.get_k(x)
        return self.eval_mean(x,k=k), self.eval_var(x,k=k)

    ###########################
    ## Miscellaneous Methods ##
    ###########################

    def drop_data(self):
        '''
        Removes a random data point from the dataset

        #TODO: change this to drop a datapoint that will minimize information loss
        '''
        N = self.X.shape[1]
        idx = np.random.randint(0,N)
        self.X = np.delete(self.X,idx,1)
        self.y = np.delete(self.y,idx)
        self.train()

    ##########################
    ## Optimization Methods ##
    ##########################

    @staticmethod
    def loglikelihood(y, K, sigma_n, alpha=None):
        '''
        Static method.

        Compute the log-likelihood of a given set of hyperparameters.
        '''
        K_y = K + sigma_n*np.eye(K.shape[0])
        if alpha is None:
            alpha = np.linalg.solve(K_y,y)
        sign, logdet = np.linalg.slogdet(K_y)
        
        if sign < 0:
            raise RuntimeError('Covariance has negative determinant')

        ll = -.5*y.T.dot(alpha)
        ll += -.5*logdet
        ll += -.5*y.shape[0]*np.log(2*np.pi)
        return ll

    @staticmethod
    def loglikelihood_grad(alpha, inv_K, dK):
        '''
        Static method.

        Compute the gradient of the log-likelihood with respect to each of the parameters.
        '''
        N_params = len(dK)
        grad = np.zeros(N_params)

        for i in range(N_params):
            grad[i] = .5*np.trace((alpha[...,np.newaxis].dot(alpha[...,np.newaxis].T) - inv_K).dot(dK[i]))

        return grad

    def get_loglikelihood(self):
        '''
        Compute the log-likelihood for the current model, wrapper for the static method
        '''
        return self.loglikelihood(self.y, self.K, self.sigma_n, alpha=self.alpha)

    def get_loglikelihood_grad(self):
        '''
        Compute the gradient of the log-likelihood with respect to each of the parameters, wrapper for the static method
        '''
        dK = self.kernel.eval_grad_batch_symm(self.X)        
        return self.loglikelihood_grad(self.alpha, self.inv_K, dK)

    def optimize_hyperparameters_gradient(self):
        '''
        Uses gradient descent to optimize the hyperparameters
        '''
        params = self.kernel.params

        ll = self.get_loglikelihood()
        grad = np.inf
        stepsize = 1
        while (np.linalg.norm(grad) > 0.1 and stepsize > 1e-10):
            print 'computing gradient'
            grad = self.get_loglikelihood_grad()
            print 'Gradient magnitude: %s' % np.linalg.norm(grad)
            new_params, stepsize = self.hyperparameter_linesearch(params, grad, max_step=10.0)
            

    def optimize_hyperparameters_grid_search(self, lb=[0.1, 0.1], ub=[10,100]):
        '''
        Uses a grid search to optimize the hyperparameters
        '''
        params = np.meshgrid(np.linspace(lb[0], ub[0], 10), np.linspace(lb[1],ub[1],10))
        params = np.reshape(params, (2,-1))
        ll = np.zeros(params.shape[1])

        for i in range(params.shape[1]):
            self.kernel.set_params(params[:,i])
            self.train()
            ll[i] = self.get_loglikelihood()

        self.kernel.set_params(params[:, np.argmax(ll)])
        self.train()

    def optimize_hyperparameters_random_search(self, lb=[0.1, 0.1], ub=[10,100]):
        '''
        Uses a grid search to optimize the hyperparameters
        '''

        # generate some random hyperparameter samples, include the current parameters
        params = np.vstack([[self.kernel.params], np.random.uniform(lb, ub, (10, len(lb)))])
        params = params.T
        # params = np.hstack([[self.kernel.params], params])
        ll = np.zeros(params.shape[1])

        for i in range(params.shape[1]):
            self.kernel.set_params(params[:,i])
            self.train()
            ll[i] = self.get_loglikelihood()

        self.kernel.set_params(params[:, np.argmax(ll)])
        self.train()

    def hyperparameter_linesearch(self, params, direction, max_step=np.inf):
        '''
        Perform an Armijo-style line search to find a stepsize that works
        '''
        
        ll = self.get_loglikelihood()
        stepsize = min(1.0, max_step/np.linalg.norm(direction))
        step_factor = 0.1
        
        while True:
            new_params = params + stepsize*direction
            if (new_params <= 0).any():
                stepsize *= step_factor
                continue
            self.kernel.set_params(new_params)
            self.train()
            try:
                new_ll = self.get_loglikelihood()
            except:
                stepsize *= step_factor
                continue

            if new_ll > ll:
                return params, stepsize
            else:
                if stepsize < 1e-20:
                    self.kernel.set_params(params)
                    print 'stepsize too small'
                    return params, stepsize
                stepsize *= step_factor