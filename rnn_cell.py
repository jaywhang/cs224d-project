"""
Our own module for RNN cells, similar to tf.nn.rnn_cell.*.
The main difference is that we control whether or not a variable is resused
*within* the cells rather than relying on an external call to reuse_variables
to make this happen.

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

class RNNCell(object):
  """Abstract class for an RNN cell"""
  def __call__(self, inputs, state, scope=None, **kwargs):
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
  def __init__(self, num_units, forget_bias=1.0, input_size=None):
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size
    self._forget_bias = forget_bias

    # h' = hH + xW + b
    with tf.variable_scope("BasicLSTMCell"):
      tf.get_variable("H", [self._num_units, 4*self._num_units])
      tf.get_variable("W", [self._input_size, 4*self._num_units])
      tf.get_variable("b", [4*self._num_units])

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope("BasicLSTMCell", reuse=True):
      H = tf.get_variable("H", [self._num_units, 4*self._num_units])
      W = tf.get_variable("W", [self._input_size, 4*self._num_units])
      b = tf.get_variable("b", [4*self._num_units])

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


class DropoutWrapper(RNNCell):
  """Operator adding dropout to inputs and outputs of the given cell."""
  def __init__(self, cell, input_keep_prob=1.0,
      output_keep_prob=1.0, seed=None):
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    self._seed = seed

  def __call__(self, inputs, state, scope=None):
    """Run the cell with the declared dropouts."""
    if (self._input_keep_prob < 1):
      inputs = tf.nn.dropout(inputs, self._input_keep_prob, seed=self._seed)
    output, new_state = self._cell(inputs, state)
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
