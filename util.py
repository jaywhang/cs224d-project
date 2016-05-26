import numpy as np
import tensorflow as tf

def variable_summaries(var, name=""):
  if not name:
    name = var.name
  tf.scalar_summary('max/' + name, tf.reduce_max(var))
  tf.scalar_summary('min/' + name, tf.reduce_min(var))
  tf.histogram_summary(name, var)

def orthogonal_initializer(shape, dtype):
  # taken from https://github.com/cooijmanstim/recurrent-batch-normalization/blob/master/penntreebank.py
  # ...which was taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v  # pick the one with the correct shape
  q = q.reshape(shape)
  return q[:shape[0], :shape[1]]
