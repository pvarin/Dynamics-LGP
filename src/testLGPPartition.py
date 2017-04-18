import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from LocalGaussianProcess import LGPCollection


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
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
    model.initialize(X,y)
    plt.figure()
    for j, m in enumerate(model.models):
        mean = m.center
        cov = (m.X-m.center[...,np.newaxis]).dot((m.X-m.center[...,np.newaxis]).T)/m.X.shape[1]
        plot_cov_ellipse(cov, mean, fill=False, zorder=j)
        plt.scatter(m.X[0,:],m.X[1,:],zorder=j)

if __name__ == '__main__':
    N = 1000

    # test with sequential data
    X, y = get_sequential_data(N)
    seq_model = LGPCollection(.98,100)
    plot_local_models(X,y,seq_model)

    # test with randomly ordered data
    X, y = get_random_data(N)
    unordered_model = LGPCollection(.98,100)
    plot_local_models(X,y,unordered_model)
    plt.show()

