from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn

def linear(x, output_size):
  with tf.variable_scope("Linear"):
    W = tf.get_variable("W", [x.get_shape().as_list()[1], output_size])
    return tf.matmul(x, W)

def batch_norm(x, scale=True, offset=False):
  with tf.variable_scope("BatchNorm"):
    mean, variance = tf.nn.moments(x, axes=[0])
    if scale:
      gamma = tf.get_variable("gamma",
          initializer=tf.ones([x.get_shape().as_list()[1]])/10)
    else:
      gamma = None
    if offset:
      beta = gamma = tf.get_variable("gamma",
          initializer=tf.zeros([x.get_shape().as_list()[1]]))
    else:
      beta = None
    return tf.nn.batch_normalization(x, mean, variance,
              offset=beta, scale=gamma, variance_epsilon=1e-3)


class BNLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = tf.split(1, 2, state)
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate

      with tf.variable_scope("x"):
        xconcat = batch_norm(linear(inputs, 4*self._num_units))
      with tf.variable_scope("h"):
        hconcat = batch_norm(linear(h, 4*self._num_units))

      i, j, f, o = [a+b for a, b in zip(tf.split(1, 4, xconcat), tf.split(1, 4, hconcat))]

      new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
      new_h = tf.tanh(batch_norm(new_c)) * tf.sigmoid(o)

      return new_h, tf.concat(1, [new_c, new_h])
