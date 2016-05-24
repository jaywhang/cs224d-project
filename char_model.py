# Simple character-level language model.

import sys
import time
import numpy as np
import tensorflow as tf
import rnn_cell
from util import *

class CharacterModel(object):
  def __init__(self, config):
    self._config = config

    # Input placeholders
    self._input_seq = tf.placeholder(tf.int32,
                                     [None, config.seq_length],
                                     name='input_seq')
    self._target_seq = tf.placeholder(tf.int32,
                                      [None, config.seq_length],
                                      name='target_seq')

    embedding = tf.get_variable('embedding',
                                [config.vocab_size, config.hidden_size])
    inputs = tf.gather(embedding, self._input_seq)

    # Hidden layers: stacked LSTM cells with Dropout.
    if config.cell_type == rnn_cell.BasicLSTMCell:
      cell = rnn_cell.BasicLSTMCell(config.hidden_size)
    elif config.cell_type == rnn_cell.BNLSTMCell:
      cell = rnn_cell.BNLSTMCell(config.is_training, config.hidden_size)
    else:
      raise ValueError("Unknown cell_type")

    # Apply dropout if we're training.
    if config.is_training and config.keep_prob < 1.0:
      self._cell = cell = rnn_cell.DropoutWrapper(cell,
        input_keep_prob=config.keep_prob, output_keep_prob=config.keep_prob)

    # No implementation of MultiRNNCell in our own rnn_cell.py yet
    # self._multi_cell = multi_cell = (
    #  tf.nn.rnn_cell.MultiRNNCell([cell] * config.hidden_depth))

    self._cell = cell

    # Placeholder for initial hidden state.
    self._initial_state = tf.placeholder(tf.float32,
                                        [None, cell.state_size])

    # Split inputs into individual timesteps for BPTT.
    split_input = [tf.squeeze(_input, squeeze_dims=[1])
                   for _input in tf.split(1, config.seq_length, inputs)]

    # Create the recurrent network.
    state = self._initial_state
    outputs = []
    for time_step in range(config.seq_length):
      if config.cell_type == rnn_cell.BasicLSTMCell:
        cell_output, state = cell(split_input[time_step], state)
      elif config.cell_type == rnn_cell.BNLSTMCell:
        cell_output, state = cell(split_input[time_step], state,
                                    time_step=time_step)
      else:
        raise ValueError("Unknown cell_type")
      outputs.append(cell_output)
    self._final_state = state

    # Reshape the output to [(batch_size * seq_length), hidden_size]
    outputs = tf.reshape(tf.concat(1, outputs), [-1, config.hidden_size])

    # Softmax
    softmax_w = tf.get_variable('softmax_w',
                                [config.vocab_size, config.hidden_size],
                                #initializer=orthogonal_initializer)
                                initializer=None)
    softmax_b = tf.get_variable('softmax_b', [config.vocab_size])
    self._logits = tf.matmul(outputs, tf.transpose(softmax_w)) + softmax_b
    self._probs = tf.nn.softmax(self._logits)

    # Average cross-entropy loss within the batch.
    loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(
      self._logits, tf.reshape(self._target_seq, [-1]))
    self._loss = tf.reduce_mean(loss_tensor)
    self._perplexity = tf.exp(self._loss)

    # Optimizer
    if self.is_training:  # shouldn't need this if but just in case
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          config.max_grad_norm)
      optimizer = config.optimizer(config.learning_rate)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  @property
  def config(self):
    return self._config

  @property
  def input_seq(self):
    return self._input_seq

  @property
  def target_seq(self):
    return self._target_seq

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def final_state(self):
    return self._final_state

  @property
  def train_op(self):
    return self._train_op

  @property
  def logits(self):
    return self._logits

  @property
  def probs(self):
    return _probs

  @property
  def loss(self):
    return self._loss

  @property
  def perplexity(self):
    return self._perplexity

  @property
  def zero_state(self):
    return self._cell.zero_state(self._config.batch_size, tf.float32)

  def run_epoch(self, sess, data_iterator, verbose=True):
    """Runs one epoch of training."""
    start_time = time.time()
    losses, perplexities = [], []
    state, = sess.run([self.zero_state])

    if self._config.is_training:
      op = self.train_op
    else:
      op = tf.no_op()

    for inputs, labels, i, num_batches in data_iterator:
      # don't save the state, exactly seq_length sequences for now.
      # loss, perp, _, state = sess.run(
      loss, perp, _, _ = sess.run(
          [self.loss, self.perplexity, op, self.final_state],
          feed_dict={self.input_seq: inputs,
                     self.target_seq: labels,
                     self.initial_state: state})
      losses.append(loss)
      perplexities.append(perp)

      if verbose and (i % 10 == 0 or i == num_batches-1):
        sys.stdout.write('\r{} / {} : loss = {:.4f}, perp = {:.3f}'.format(
              i+1, num_batches, loss, np.exp(loss)))
        sys.stdout.flush()

    elapsed = time.time() - start_time
    if verbose:
      print ('\nEpoch finished in {} iterations ({:.2f} sec).'
             .format(num_batches, elapsed))

    return losses, perplexities, num_batches

  def sample(self, sess, indices, length, temperature=1.0):
    assert not self._config.is_training, 'This model has config for training.'
    assert indices, 'Must provide at least one token.'

    def sample_next(_idx, _state):
      new_state, logits = sess.run(
          [self._final_state, self._logits],
          feed_dict={self._input_seq: [[_idx]], self._initial_state: _state})

      probs = (logits / temperature).astype(np.float64)
      # For numerical stability
      probs = probs - np.max(probs)
      probs = np.exp(probs)
      probs = probs / np.sum(probs)
      probs = probs.flatten()

      return np.random.choice(len(probs), p=probs), new_state

    result = list(indices)
    state = self.zero_state.eval()

    # Warm up
    if len(indices) > 1:
      for idx in indices[:-1]:
        _, state = sample_next(idx, state)

    # Start sampling
    for _ in xrange(length):
      new_idx, state = sample_next(result[-1], state)
      result.append(new_idx)

    return result
