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
    self._input_size = num_units if input_size is None else input_size
    self._forget_bias = forget_bias
    self.is_training = is_training

    # h' = hH + xW + b
    with tf.variable_scope("BasicLSTMCell"):
      H = tf.get_variable("H", [self._num_units, 4*self._num_units],
          initializer=orthogonal_initializer)
      W = tf.get_variable("W", [self._input_size, 4*self._num_units],
          initializer=orthogonal_initializer)
      b = tf.get_variable("b", [4*self._num_units])
      if not self.is_training:
        tf.scalar_summary('W_norm', tf.global_norm([W]))
        tf.scalar_summary('H_norm', tf.global_norm([H]))

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

    if not self.is_training and time_step in [0, 49, 99]:
      variable_summaries(new_c, "new_c/%s" % time_step)
      variable_summaries(new_h, "new_h/%s" % time_step)
      variable_summaries(i, "i/%s" % time_step)
      variable_summaries(j, "j/%s" % time_step)
      variable_summaries(f, "f/%s" % time_step)
      variable_summaries(o, "o/%s" % time_step)

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
    super(BNLSTMCell, self).__init__(is_training, num_units, forget_bias, input_size)
    with tf.variable_scope("BatchNorm"):
      tf.get_variable("xgamma", initializer=tf.ones([4*num_units])/10)
      tf.get_variable("hgamma", initializer=tf.ones([4*num_units])/10)
      tf.get_variable("cgamma", initializer=tf.ones([num_units])/10)
      tf.get_variable("cbeta", initializer=tf.zeros([num_units]))

  def _batch_norm(self, x, xmean, xvar, gamma=None, beta=None,
                  variance_epsilon=1e-5):
    control_inputs = []
    if self.is_training:
      mean, variance = tf.nn.moments(x, axes=[0])
      update_moving_mean = moving_averages.assign_moving_average(
          xmean, mean, 0.95)
      update_moving_variance = moving_averages.assign_moving_average(
          xvar, variance, 0.95)
      control_inputs = [update_moving_mean, update_moving_variance]
    else:
      mean = xmean
      variance = xvar
    with tf.control_dependencies(control_inputs):
      return tf.nn.batch_normalization(x, mean, variance,
                scale=gamma, offset=beta, variance_epsilon=variance_epsilon)

  def __call__(self, inputs, state, time_step):
    with tf.variable_scope("BasicLSTMCell", reuse=True):
      H = tf.get_variable("H")
      W = tf.get_variable("W")
      b = tf.get_variable("b")

    with tf.variable_scope("BatchNorm", reuse=True):
      xgamma = tf.get_variable("xgamma")
      hgamma = tf.get_variable("hgamma")
      cgamma = tf.get_variable("cgamma")
      cbeta = tf.get_variable("cbeta")

    with tf.variable_scope("BN-Stats-T%s" % time_step):
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
    xconcat = self._batch_norm(tf.matmul(inputs, W), xmean, xvar, xgamma)
    hconcat = self._batch_norm(tf.matmul(h, H), hmean, hvar, hgamma)
    concat = xconcat + hconcat + b
    i, j, f, o = tf.split(1, 4, concat)

    new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
    new_c_bn = self._batch_norm(new_c, cmean, cvar, cgamma, cbeta)
    new_h = tf.tanh(new_c_bn) * tf.sigmoid(o)

    if not self.is_training and time_step in [0, 49, 99]:
      variable_summaries(new_c, "new_c/%s" % time_step)
      variable_summaries(new_h, "new_h/%s" % time_step)
      variable_summaries(i, "i/%s" % time_step)
      variable_summaries(j, "j/%s" % time_step)
      variable_summaries(f, "f/%s" % time_step)
      variable_summaries(o, "o/%s" % time_step)
      variable_summaries(xmean, "xmean/%s" % time_step)
      variable_summaries(hmean, "hmean/%s" % time_step)
      variable_summaries(cmean, "cmean/%s" % time_step)
      variable_summaries(xvar, "xvar/%s" % time_step)
      variable_summaries(hvar, "hvar/%s" % time_step)
      variable_summaries(cvar, "cvar/%s" % time_step)
      variable_summaries(xgamma, "xgamma/%s" % time_step)
      variable_summaries(hgamma, "hgamma/%s" % time_step)
      variable_summaries(cgamma, "cgamma/%s" % time_step)
      variable_summaries(cbeta, "cbeta/%s" % time_step)

    return new_h, tf.concat(1, [new_c, new_h])

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
