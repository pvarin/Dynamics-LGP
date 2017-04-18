import numpy as np

from GaussianProcess import GaussianProcess
from kernels import genGaussianKernel

class LocalGaussianProcess(GaussianProcess):
    def __init__(self, max_data=100, **kwargs):
        super(LocalGaussianProcess, self).__init__(
                    kernel = genGaussianKernel(kwargs['sigma_s'],
                                               kwargs['width']))
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
    def __init__(self, distance_threshold, max_local_data, X=None, y=None, **kwargs):
        # handle no input params and set defaults
        if kwargs is None:
            kwargs = dict()
        self.sigma_n = kwargs.get('sigma_n',1)
        self.sigma_s = kwargs.get('sigma_s',1)
        self.width = kwargs.get('width',1)

        self.models = set()
        self.distance_threshold = distance_threshold
        self.max_local_data = max_local_data

        if (X is not None) and (y is not None):
            self.initialize(X,y)

    def compute_distance(self, x1, x2):
        return  np.exp(-.5*(x1-x2).T.dot(self.width).dot(x1-x2))

    def initialize(self, X, y):
        for i in range(X.shape[1]):
            self.update(X[:,i],y[i,np.newaxis])

    def update(self, x, y):
        m, d = self.get_closest_model(x)
        if (m is None) or (d < self.distance_threshold):
            self.add_model(x[..., np.newaxis],y)
        else:
            m.update(x, y)

    def add_model(self, X, y):
        GP_params = {'sigma_n' : self.sigma_n,
                     'sigma_s' : self.sigma_s,
                     'width'   : self.width}
        m = LocalGaussianProcess(max_data = self.max_local_data, **GP_params)
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

    def eval_mean(self, x):
        models, distances = self.get_relevant_models(x)
        return sum([d*m.eval_mean(x) for m, d in zip(models, distances)])/np.sum(distances)
