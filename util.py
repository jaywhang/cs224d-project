import numpy as np

def orthogonal_initializer(shape, dtype):
  u, _, v = np.linalg.svd(np.random.random(shape), full_matrices=False)
  q = u if u.shape == shape else v
  q = q.reshape(shape)
  return q
