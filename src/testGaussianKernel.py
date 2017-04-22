import numpy as np
import matplotlib.pyplot as plt
from kernels import GaussianKernel

# generate the center of the gaussian and the grid
x_q = np.random.uniform(-1,1,2)
x_p = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
x_p_0 = np.reshape(x_p[0],(-1,1))
x_p_1 = np.reshape(x_p[1],(-1,1))

# generate a random 2x2 positive definite matrix with close to orthogonal eigenvectors
a = np.random.random(2)
a = a/np.linalg.norm(a)
b = np.random.random(2)
b = b/np.linalg.norm(b)
b = b - .8*b.dot(a)*a # make b almost proportional to a
b = b/np.linalg.norm(b)
W = np.outer(a,a) + np.outer(b,b)

# compute the kernel value over the entire grid
kernel = GaussianKernel([1,W])
K = np.array([kernel.eval(x_q,np.hstack([x_0,x_1])) for x_0, x_1 in zip(x_p_0, x_p_1)])
K = np.reshape(K,x_p[0].shape)

# plot the gaussian
plt.pcolor(x_p[0],x_p[1],K)
plt.plot(x_q[0],x_q[1],'*')
plt.show()