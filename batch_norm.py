from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn

def bn_linear(x, h, output_size):
  with tf.variable_scope("Linear"):
    Wh = tf.get_variable("Wh", [h.get_shape().as_list()[1], output_size])
    Wx = tf.get_variable("Wx", [x.get_shape().as_list()[1], output_size])
    b = tf.get_variable("b", [output_size],
                        initializer=tf.constant_initializer(0.0))
  with tf.variable_scope("BN_h"):
    hgamma = tf.get_variable("gamma", initializer=tf.ones([output_size])/10)
    yh = tf.matmul(h, Wh)
    mh, vh = tf.nn.moments(yh, axes=[0])
    bn_h = tf.nn.batch_normalization(yh, mh, vh,
              offset=None, scale=hgamma, variance_epsilon=1e-3)
  with tf.variable_scope("BN_x"):
    xgamma = tf.get_variable("gamma", initializer=tf.ones([output_size])/10)
    yx = tf.matmul(x, Wx)
    mx, vx = tf.nn.moments(yx, axes=[0])
    bn_x = tf.nn.batch_normalization(yx, mx, vx,
              offset=None, scale=xgamma, variance_epsilon=1e-3)
  return bn_h + bn_x + b

class BNLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = tf.split(1, 2, state)
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      with tf.variable_scope("IGate"):
        i = bn_linear(inputs, h, self._num_units)
      with tf.variable_scope("fGate"):
        f = bn_linear(inputs, h, self._num_units)
      with tf.variable_scope("oGate"):
        o = bn_linear(inputs, h, self._num_units)
      with tf.variable_scope("new_state"):
        j = bn_linear(inputs, h, self._num_units)

      new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
      mc, vc = tf.nn.moments(new_c, axes=[0])
      gamma = tf.get_variable("gamma", initializer=tf.ones([self._num_units])/10)
      beta = tf.get_variable("beta", initializer=tf.zeros([self._num_units]))
      bn_new_c = tf.nn.batch_normalization(new_c, mc, vc, offset=beta, scale=gamma, variance_epsilon=1e-3)
      new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

      return new_h, tf.concat(1, [new_c, new_h])
