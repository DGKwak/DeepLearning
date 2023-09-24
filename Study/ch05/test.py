import numpy as np

t = np.array([2, 7, 0, 1])

y = np.array([[0,0,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0],
              [1,0,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0]])

dx = y.copy()

b = t.shape[0]

print(dx[np.arange(b), t])

dx[np.arange(b), t] -= 1

print(dx[np.arange(b), t])

x = np.array([[0,1,2,3,4,5,6,7,8,9],
              [0,0,0,0,0,0,0,1,8,0],
              [1,0,0,0,0,0,2,0,0,0],
              [0,1,0,0,0,0,33,0,0,0]])

permutation = np.random.permutation(x.shape[0])
x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]

print(permutation)