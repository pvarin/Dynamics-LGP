import numpy as np

def genRandomFunction(dim=1):
  '''
  Generates a random multi-dimensional fourier sine series on the periodic domain [0,1)
  parameterized by random coefficients that fall off exponentially with the degree of
  the sinusoid
  '''
  depth = 5
  m = np.meshgrid(*(range(1,depth+1),)*dim)
  m = np.stack(m,dim)
  coeff = np.random.random((depth,)*dim)**np.prod(m,dim)
  m = m*np.pi

  def f(x):
    y = np.sin(m[...,np.newaxis]*x)
    y = np.prod(y,dim)
    y = coeff[...,np.newaxis]*y
    y = np.sum(y,tuple(range(dim)))
    return y
  return f

def genDataFromFunction(f, dim=1, domain=None, sigma=None, N=100):
  '''
  Samples uniformly at random over a rectangular domain and adds Gaussian noise
  '''
  if domain is None:
    domain = ((0,1),)*dim
  x = [np.random.uniform(domain[i][0],domain[i][1],N) for i in range(dim)]
  x = np.stack(x,0)
  y = f(x)
  if sigma is None:
    sigma = 0.1*np.sqrt(np.mean((y-np.mean(y))**2))

  y += np.random.normal(0,sigma,y.shape)
  return x, y

if __name__ == "__main__":
  import matplotlib.pyplot as plt

  # test one-dimensional
  dim = 1
  x = np.linspace(0,1)
  f = genRandomFunction(dim)
  x_data, y_data = genDataFromFunction(f,dim)
  plt.plot(x,f(x[np.newaxis,:]))
  plt.plot(x_data[0,:], y_data,'.')

  # test two dimensional
  dim = 2
  domain = ((0,1),)*dim
  x, y = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
  x_eval = np.vstack([np.reshape(x,(1,-1)), np.reshape(y,(1,-1))])
  f = genRandomFunction(dim)
  f_eval = np.reshape(f(x_eval), x.shape)
  x_data, y_data = genDataFromFunction(f,dim,N=10000)

  plt.figure()
  plt.subplot(2,1,1) # plot the real function
  plt.pcolor(x,y,f_eval,cmap='RdBu')
  plt.title('True Function')
  plt.subplot(2,1,2) # plot the data
  plt.scatter(x_data[0,:],x_data[1,:],c=y_data,cmap='RdBu')
  plt.title('Sampled Data')

  plt.show()