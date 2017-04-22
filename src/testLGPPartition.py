import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from LocalGaussianProcess import LGPCollection


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    '''
    plots an ellipse corresponding to the 2 standard deviation
    level set of the data around the mean
    '''
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def get_sequential_data(N):
    '''
    Generate noisy data that is ordered as if it was generated from a
    particle moving through space
    '''
    x_mean = np.linspace(0,1,N)
    y_mean = np.linspace(0,1,N)
    X = np.vstack([x_mean, y_mean]) + np.random.normal(0,.1,[2,N])
    y = np.random.random(N)
    return X, y

def get_random_data(N):
    '''
    Generate data in a random order
    '''
    X = np.random.random((2,N))
    y = np.random.random(N)
    return X, y

def step_through_data(X, y, model):
    '''
    Watch as one data point at a time is added to the LGP model
    used for debugging
    '''
    plt.figure()
    plt.ion()
    plt.show()

    for i in range(X.shape[1]):
        model.update(X[:,i],y[i,np.newaxis])
        plt.clf()
        for j, m in enumerate(model.models):
            mean = m.center
            cov = (m.X-m.center[...,np.newaxis]).dot((m.X-m.center[...,np.newaxis]).T)/m.X.shape[1]
            plot_cov_ellipse(cov, mean, fill=False, zorder=j)
            plt.scatter(m.X[0,:],m.X[1,:],zorder=j)

        plt.draw()
        raw_input('Press to update model')
        print 'There are %s current models' % len(model.models)

def plot_local_models(X, y, model):
    '''
    Plots the data points for each of the local GPs, grouped by
    color and outlined by an ellipse according to the dataset covariance
    '''
    model.train(X,y)
    plt.figure()
    for m in model.models:
        mean = m.center
        cov = (m.X-m.center[...,np.newaxis]).dot((m.X-m.center[...,np.newaxis]).T)/m.X.shape[1]
        plot_cov_ellipse(cov, mean, fill=False)
        plt.scatter(m.X[0,:],m.X[1,:])

def plot_model_weights(X, y, center, model):
    '''
    Same as plot_local_model, but color codes the models by their
    distance to the variable 'center'
    '''
    model.train(X,y)
    plt.figure()
    for m in model.models:
        mean = m.center
        cov = (m.X-m.center[...,np.newaxis]).dot((m.X-m.center[...,np.newaxis]).T)/m.X.shape[1]
        plot_cov_ellipse(cov, mean, fill=False)
        color = model.compute_distance(m, m.center, center)*np.ones(m.X[1,:].shape)
        plt.scatter(m.X[0,:], m.X[1,:], c=color, vmax=1.0, vmin=min(np.min(color),0.8))
    plt.colorbar()

if __name__ == '__main__':
    N = 1000
    init_params = [1.0,1.0]

    # test with sequential data
    X, y = get_sequential_data(N)
    seq_model = LGPCollection(.98, 100, init_params=init_params)
    plot_local_models(X,y,seq_model)

    # test with randomly ordered data
    X, y = get_random_data(N)
    unordered_model = LGPCollection(.98, 100, init_params=init_params)
    plot_local_models(X,y,unordered_model)

    # test the model weights
    plot_model_weights(X, y, np.mean(X,1), unordered_model)
    
    plt.show()
