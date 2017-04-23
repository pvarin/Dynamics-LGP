import numpy as np

from GaussianProcess import GaussianProcess
from kernels import GaussianKernel

class LocalGaussianProcess(GaussianProcess):
    def __init__(self, kernel, max_data=100, **kwargs):
        super(LocalGaussianProcess, self).__init__(kernel=kernel, **kwargs)
        self.center = None
        self.max_data = max_data

    def update_center(self):
        self.center = np.mean(self.X,1)

    def train(self, **kwargs):
        super(LocalGaussianProcess, self).train(**kwargs)
        self.update_center()

    def update(self, x, y):
        super(LocalGaussianProcess, self).update(x, y)
        if self.X.shape[1] > self.max_data:
            self.drop_data()
        self.update_center()

class LGPCollection:
    def __init__(self, kernel, distance_threshold, max_local_data, max_models=10, X=None, y=None, sigma_n = 0.001):
        
        self.kernel = kernel
        self.sigma_n = sigma_n
        self.max_models = max_models
        self.models = set()
        self.distance_threshold = distance_threshold
        self.max_local_data = max_local_data

        if (X is not None) and (y is not None):
            self.initialize(X,y)

    def compute_distance(self, x1, x2):
        return  self.kernel.eval(x1, x2)

    def train(self, X=None, y=None, optimize=True):
        if (X is None) and (y is None):
            for m in self.models:
                m.train()
        else:
            if optimize:
                optimizing_iter = range(0,X.shape[1],10)
            for i in range(X.shape[1]):
                print i
                self.update(X[:,i],y[i,np.newaxis])
                if optimize and i in optimizing_iter:
                    self.optimize_hyperparameters_random_search()

    def update(self, x, y):
        m, d = self.get_closest_model(x)
        if (m is None) or (d < self.distance_threshold):
            self.add_model(x[..., np.newaxis],y)
        else:
            m.update(x, y)

    def add_model(self, X, y):
        m = LocalGaussianProcess(self.kernel, sigma_n=self.sigma_n, max_data=self.max_local_data)
        m.train(X=X,y=y)
        self.models.add(m)

    def get_closest_model(self, x):
        nearest_model = None
        max_distance = -np.inf

        for m in self.models:
            d = self.compute_distance(m.center, x)
            if d > max_distance:
                nearest_model = m
                max_distance = d

        return nearest_model, max_distance

    def get_relevant_models(self, x):
        models = []
        distances = []

        for m in self.models:
            d = self.compute_distance(m.center, x)
            if d > self.distance_threshold:
                models.append(m)
                distances.append(d)

        return models, distances

    def get_nearest_models(self, x, M):
        models = [(self.compute_distance(m.center, x), m) for m in self.models]
        models.sort()
        return models[-M:]

    def eval_mean(self, x):
        models_distances = self.get_nearest_models(x, self.max_models)
        return sum([d*m.eval_mean(x) for d, m in models_distances])/sum([d for d, _ in models_distances])

    def get_loglikelihood(self):
        ll = 0.0
        for m in self.models:
            ll += m.get_loglikelihood()

        return ll

    def optimize_hyperparameters_random_search(self, lb=[0.1, 0.1], ub=[10.0,100.0], N=10):
        # generate some random hyperparameter samples, include the current parameters
        params = np.vstack([[self.kernel.params], np.random.uniform(lb, ub, (N, len(lb)))])
        params = params.T
        # params = np.hstack([[self.kernel.params], params])
        ll = np.zeros(params.shape[1])

        for i in range(params.shape[1]):
            self.kernel.set_params(params[:,i])
            self.train()
            ll[i] = self.get_loglikelihood()
        i = np.argmax(ll)
        self.kernel.set_params(params[:, i])
        self.train()