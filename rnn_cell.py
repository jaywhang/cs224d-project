"""
Our own module for RNN cells, similar to tf.nn.rnn_cell.*.
The main difference is that we control whether or not a variable is resused
*within* the cells rather than relying on an external call to reuse_variables
to make this happen. The general pattern to follow is that any shared should be
created in __init__, and retreived with reuse=True in __call__, while
variables not shared between time steps should be created in __call__.

This allowed more fine-grained control of which variables in the cell are shared
and which are not, eg. share standard LSTM variables but not means/avg in
batch normalization across time steps.

Most of this is copied and pasted from here, slightly refactored and
some validations of function inputs that does not relate to our use case
removed.

https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/python/ops/rnn_cell.py
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import moving_averages

from util import *


def _batch_norm(is_training, x, moving_mean, moving_var, gamma=None, beta=None,
                variance_epsilon=1e-5):
  control_inputs = []
  if is_training:
    mean, variance = tf.nn.moments(x, axes=[0])
    update_moving_mean = moving_averages.assign_moving_average(
        moving_mean, mean, 0.95)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_var, variance, 0.95)
    control_inputs = [update_moving_mean, update_moving_variance]
  else:
    mean = moving_mean
    variance = moving_var
  with tf.control_dependencies(control_inputs):
    return tf.nn.batch_normalization(x, mean, variance,
              scale=gamma, offset=beta, variance_epsilon=variance_epsilon)


class RNNCell(object):
  """Abstract class for an RNN cell"""
  def __call__(self, inputs, state):
    """Run this RNN cell on inputs, starting from the given state."""
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """Integer or tuple of integers: size(s) of state(s) used by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    zeros = tf.zeros(tf.pack([batch_size, self.state_size]), dtype=dtype)
    zeros.set_shape([None, self.state_size])
    return zeros


class BasicLSTMCell(RNNCell):

  def __init__(self, is_training, num_units, forget_bias=1.0, input_size=None):
    self._num_units = num_units
    self._input_size = input_size or num_units
    self._forget_bias = forget_bias
    self._is_training = is_training

    # h' = hH + xW + b
    with tf.variable_scope("BasicLSTMCell"):
      H = tf.get_variable("H", [self._num_units, 4*self._num_units],
          initializer=orthogonal_initializer)
      W = tf.get_variable("W", [self._input_size, 4*self._num_units],
          initializer=orthogonal_initializer)
      b = tf.get_variable("b", [4*self._num_units])

  def __call__(self, inputs, state, time_step):
    with tf.variable_scope("BasicLSTMCell", reuse=True):
      H = tf.get_variable("H")
      W = tf.get_variable("W")
      b = tf.get_variable("b")

    c, h = tf.split(1, 2, state)
    concat = tf.matmul(h, H) + tf.matmul(inputs, W) + b
    i, j, f, o = tf.split(1, 4, concat)

    new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
    new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, tf.concat(1, [new_c, new_h])

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return 2 * self._num_units


class BNLSTMCell(BasicLSTMCell):

  def __init__(self, is_training, num_units, forget_bias=1.0, input_size=None):
    super(BNLSTMCell, self).__init__(is_training, num_units,
                                     forget_bias, input_size)
    with tf.variable_scope("LSTMBatchNorm"):
      tf.get_variable("xgamma", initializer=tf.ones([4*num_units])/10)
      tf.get_variable("hgamma", initializer=tf.ones([4*num_units])/10)
      tf.get_variable("cgamma", initializer=tf.ones([num_units])/10)
      tf.get_variable("cbeta", initializer=tf.zeros([num_units]))

  def __call__(self, inputs, state, time_step):
    with tf.variable_scope("BasicLSTMCell", reuse=True):
      H = tf.get_variable("H")
      W = tf.get_variable("W")
      b = tf.get_variable("b")

    with tf.variable_scope("LSTMBatchNorm", reuse=True):
      xgamma = tf.get_variable("xgamma")
      hgamma = tf.get_variable("hgamma")
      cgamma = tf.get_variable("cgamma")
      cbeta = tf.get_variable("cbeta")

    with tf.variable_scope("BNLSTM-Stats-T%s" % time_step):
      xmean = tf.get_variable("xmean",
          initializer=tf.zeros([4*self._num_units]), trainable=False)
      xvar = tf.get_variable("xvar",
          initializer=tf.ones([4*self._num_units]), trainable=False)
      hmean = tf.get_variable("hmean",
          initializer=tf.zeros([4*self._num_units]), trainable=False)
      hvar = tf.get_variable("hvar",
          initializer=tf.ones([4*self._num_units]), trainable=False)
      cmean = tf.get_variable("cmean",
          initializer=tf.zeros([self._num_units]), trainable=False)
      cvar = tf.get_variable("cvar",
          initializer=tf.ones([self._num_units]), trainable=False)

    c, h = tf.split(1, 2, state)

    # i, j, f, o = BN(hH) + BN(xW) + b
    # these have no betas, we let the single bias b take care of this
    xconcat = _batch_norm(self._is_training, tf.matmul(inputs, W),
                          xmean, xvar, xgamma)
    hconcat = _batch_norm(self._is_training, tf.matmul(h, H),
                          hmean, hvar, hgamma)
    concat = xconcat + hconcat + b
    i, j, f, o = tf.split(1, 4, concat)

    new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
    new_c_bn = _batch_norm(self._is_training, new_c,
                           cmean, cvar, cgamma, cbeta)
    new_h = tf.tanh(new_c_bn) * tf.sigmoid(o)

    return new_h, tf.concat(1, [new_c, new_h])


class GRUCell(RNNCell):
  """Gated Recurrent Unit cell implementation, mostly copied from TensorFlow."""
  def __init__(self, is_training, num_units, input_size=None, bias=1.0):
    self._num_units = num_units
    self._input_size = input_size or num_units
    self._is_training = is_training
    self._bias = bias

    with tf.variable_scope("GRUCell"):
      # Gate weights
      W = tf.get_variable("W", [self._input_size, 2*self._num_units])
      H = tf.get_variable("H", [self._num_units, 2*self._num_units])
      B = tf.get_variable("B", [2*self._num_units])
      # Memory weights
      Wh = tf.get_variable("Wh", [self._input_size, self._num_units])
      Hh = tf.get_variable("Hh", [self._num_units, self._num_units])

  @property
  def state_size(self):
    return self._num_units

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, time_step):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope("GRUCell", reuse=True):
      W = tf.get_variable("W")
      H = tf.get_variable("H")
      B = tf.get_variable("B")
      Wh = tf.get_variable("Wh")
      Hh = tf.get_variable("Hh")

    # [r, u] = sigmoid(x*W + h*H + B)
    # h_tilde = x*Wh + r o (h * Hh)
    # h_new = u o h + (1-u) o tanh(h_tilde)
    concat = tf.matmul(inputs, W) + tf.matmul(state, H) + B + self._bias
    r, u = tf.split(1, 2, tf.sigmoid(concat))
    h_tilde = tf.matmul(inputs, Wh) + r * tf.matmul(state, Hh)
    new_h = u * state + (1 - u) * tf.tanh(h_tilde)

    return new_h, new_h


class BNGRUCell(GRUCell):
  """Batch normalized Gated Recurrent Unit cell implementation."""

  def __init__(self, is_training, num_units,
               input_size=None, bias=1.0, full_bn=True):
    super(BNGRUCell, self).__init__(is_training, num_units,
                                    input_size=input_size, bias=bias)
    self._full_bn = full_bn
    with tf.variable_scope("GRUBatchNorm"):
      # Means and variances for gate BN.
      tf.get_variable("xgamma", initializer=tf.ones([2*num_units])/10)
      tf.get_variable("hgamma", initializer=tf.ones([2*num_units])/10)
      # Means and variances for actual memory content BN.
      tf.get_variable("mgamma", initializer=tf.ones([num_units])/10)
      tf.get_variable("mbeta", initializer=tf.zeros([num_units]))

      # We batch normalize more terms if full_bn is set.
      if full_bn:
        # Means and variances for h_tilde BN.
        tf.get_variable("hx_gamma", initializer=tf.ones([num_units])/10)
        tf.get_variable("hx_beta", initializer=tf.zeros([num_units]))
        tf.get_variable("hh_gamma", initializer=tf.ones([num_units])/10)
        tf.get_variable("hh_beta", initializer=tf.zeros([num_units]))

  @property
  def state_size(self):
    return self._num_units

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, time_step):
    with tf.variable_scope("GRUBatchNorm", reuse=True):
      xgamma = tf.get_variable("xgamma")
      hgamma = tf.get_variable("hgamma")
      mgamma = tf.get_variable("mgamma")
      mbeta = tf.get_variable("mbeta")

      if self._full_bn:
        hx_gamma = tf.get_variable("hx_gamma")
        hx_beta = tf.get_variable("hx_beta")
        hh_gamma = tf.get_variable("hh_gamma")
        hh_beta = tf.get_variable("hh_beta")

    with tf.variable_scope("GRUCell", reuse=True):
      W = tf.get_variable("W")
      H = tf.get_variable("H")
      B = tf.get_variable("B")
      Wh = tf.get_variable("Wh")
      Hh = tf.get_variable("Hh")

    # Means and variables for each time_step.
    with tf.variable_scope("BNGRU-Stats-T%s" % time_step):
      xmean = tf.get_variable("xmean",
          initializer=tf.zeros([2*self._num_units]), trainable=False)
      xvar = tf.get_variable("xvar",
          initializer=tf.ones([2*self._num_units]), trainable=False)
      hmean = tf.get_variable("hmean",
          initializer=tf.zeros([2*self._num_units]), trainable=False)
      hvar = tf.get_variable("hvar",
          initializer=tf.ones([2*self._num_units]), trainable=False)
      mmean = tf.get_variable("mmean",
          initializer=tf.zeros([self._num_units]), trainable=False)
      mvar = tf.get_variable("mvar",
          initializer=tf.ones([self._num_units]), trainable=False)

      if self._full_bn:
        hx_mean = tf.get_variable("hx_mean",
            initializer=tf.zeros([self._num_units]), trainable=False)
        hx_var = tf.get_variable("hx_var",
            initializer=tf.ones([self._num_units]), trainable=False)
        hh_mean = tf.get_variable("hh_mean",
            initializer=tf.zeros([self._num_units]), trainable=False)
        hh_var = tf.get_variable("hh_var",
            initializer=tf.ones([self._num_units]), trainable=False)

    # "Full" BN:
    #     [r, u] = sigmoid(BN(x*W) + BN(h*H) + B)
    #     h_tilde = BN(x*Wh) + r o BN(h*Hh)
    #     h_new = u * h_old + (1-u) * tanh(BN(h_tilde))
    #
    # "Simple" BN:
    #     [r, u] = sigmoid(BN(x*W) + BN(h*H) + B)
    #     h_tilde = x*Wh + r o h*Hh
    #     h_new = u * h_old + (1-u) * tanh(BN(h_tilde))

    bn_x = _batch_norm(self._is_training, tf.matmul(inputs, W),
                       xmean, xvar, xgamma)
    bn_h = _batch_norm(self._is_training, tf.matmul(state, H),
                       hmean, hvar, hgamma)
    concat = bn_x + bn_h + B + self._bias
    r, u = tf.split(1, 2, tf.sigmoid(concat))
    if self._full_bn:
      bn_Wh = _batch_norm(self._is_training, tf.matmul(inputs, Wh),
                          hx_mean, hx_var, hx_gamma, hx_beta)
      bn_Hh = _batch_norm(self._is_training, tf.matmul(state, Hh),
                          hh_mean, hh_var, hh_gamma, hh_beta)
      h_tilde = bn_Wh + r * bn_Hh
    else:
      h_tilde = tf.matmul(inputs, Wh) + r * tf.matmul(state, Hh)
    bn_h_tilde = _batch_norm(self._is_training, h_tilde,
                             mmean, mvar, mgamma, mbeta)
    new_h = u * state + (1 - u) * tf.tanh(bn_h_tilde)

    return new_h, new_h


class DropoutWrapper(RNNCell):
  """Operator adding dropout to inputs and outputs of the given cell."""
  def __init__(self, cell, input_keep_prob=1.0,
      output_keep_prob=1.0, seed=None):
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    self._seed = seed

  def __call__(self, inputs, state, time_step, **kwargs):
    """Run the cell with the declared dropouts."""
    if (self._input_keep_prob < 1):
      inputs = tf.nn.dropout(inputs, self._input_keep_prob, seed=self._seed)
    output, new_state = self._cell(inputs, state, time_step, **kwargs)
    if (self._output_keep_prob < 1):
      output = tf.nn.dropout(output, self._output_keep_prob, seed=self._seed)
    return output, new_state

  @property
  def input_size(self):
    return self._cell.input_size

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size
