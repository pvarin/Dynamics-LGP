import numpy as np

def genRandomFunction():
  coeff = np.random.random(10)
  coeff = np.cumprod(coeff)
  coeff = np.reshape(coeff,(1,-1))
  def f(x):
    return coeff.dot(np.vstack([np.sin(i*x) for i in range(coeff.size)])).flatten()
  # f = lambda x : coeff.dot(np.hstack([np.sin(i*x) for i in range(len(coeff))]))
  return f

def genDataFromFunction(f, domain=(0,np.pi), sigma=.01, N=100):
  x = np.array([np.random.uniform(domain[0],domain[1]) for _ in range(N)])
  x = np.reshape(x,(-1,N))
  y = f(x)
  y += np.random.normal(0,sigma,y.shape)
  return x, y

if __name__ == "__main__":
  import matplotlib.pyplot as plt

  x = np.linspace(0,np.pi)
  x = np.reshape(x,(1,len(x)))
  f = genRandomFunction()

  plt.plot(x[0,:],f(x))
  x, y = genDataFromFunction(f,sigma=0.05**2)
  plt.plot(x[0,:],y,'.')
  plt.show()