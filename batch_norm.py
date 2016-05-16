from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.python.training import moving_averages

def linear(x, output_size):
  with tf.variable_scope("Linear"):
    W = tf.get_variable("W", [x.get_shape().as_list()[1], output_size])
    return tf.matmul(x, W)

def batch_norm(x, xmean, xvar, is_training, scale=True, offset=False):
  with tf.variable_scope("BatchNorm"):
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
  control_inputs = []
  if is_training:
    mean, variance = tf.nn.moments(x, axes=[0])
    update_moving_mean = moving_averages.assign_moving_average(
        xmean, mean, 0.99)
    update_moving_variance = moving_averages.assign_moving_average(
        xvar, variance, 0.99)
    control_inputs = [update_moving_mean, update_moving_variance]
  else:
    mean = xmean
    variance = xvar
  with tf.control_dependencies(control_inputs):
    return tf.nn.batch_normalization(x, mean, variance,
              offset=beta, scale=gamma, variance_epsilon=1e-3)


class BNLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
  def __call__(self, inputs, state, is_training,
                    xmean, xvar, hmean, hvar, cmean, cvar):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(type(self).__name__):
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = tf.split(1, 2, state)
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate

      with tf.variable_scope("x"):
        xconcat = batch_norm(linear(inputs, 4*self._num_units), xmean, xvar, is_training)
      with tf.variable_scope("h"):
        hconcat = batch_norm(linear(h, 4*self._num_units), hmean, hvar, is_training)

      b = tf.get_variable("b", initializer=tf.zeros([4*self._num_units]))
      concat = xconcat + hconcat + b

      i, j, f, o = tf.split(1, 4, concat)

      new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
      new_h = tf.tanh(batch_norm(new_c, cmean, cvar, is_training)) * tf.sigmoid(o)

      return new_h, tf.concat(1, [new_c, new_h])

class DropoutWrapper(tf.nn.rnn_cell.RNNCell):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               seed=None):
    """Create a cell with added input and/or output dropout.
    Dropout is never used on the state.
    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is float and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is float and 1, no output dropout will be added.
      seed: (optional) integer, the randomness seed.
    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if keep_prob is not between 0 and 1.
    """
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    self._seed = seed

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, *moreargs):
    """Run the cell with the declared dropouts."""
    if (not isinstance(self._input_keep_prob, float) or
        self._input_keep_prob < 1):
      inputs = nn_ops.dropout(inputs, self._input_keep_prob, seed=self._seed)
    output, new_state = self._cell(inputs, state, *moreargs)
    if (not isinstance(self._output_keep_prob, float) or
        self._output_keep_prob < 1):
      output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
    return output, new_state
