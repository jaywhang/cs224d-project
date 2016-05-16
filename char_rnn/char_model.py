# Simple character-level language model.

import numpy as np
import tensorflow as tf

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

    with tf.device("/cpu:0"):
      embedding = tf.get_variable('embedding',
                                  [config.vocab_size, config.hidden_size])
      inputs = tf.gather(embedding, self._input_seq)

    # Hidden layers: stacked LSTM cells with Dropout.
    cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size)

    # Apply dropout.
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                         output_keep_prob=config.keep_prob)
    self._multi_cell = multi_cell = (
      tf.nn.rnn_cell.MultiRNNCell([cell] * config.hidden_depth))

    # Placeholder for initial hidden state.
    self._initial_state = tf.placeholder(tf.float32,
                                        [None, multi_cell.state_size])

    # Split inputs into individual timesteps for BPTT.
    split_input = [tf.squeeze(_input, squeeze_dims=[1])
                   for _input in tf.split(1, config.seq_length, inputs)]

    # Create the recurrent network.
    outputs, self._final_state = tf.nn.rnn(multi_cell, split_input,
                                           initial_state=self._initial_state)

    # Reshape the output to [(batch_size * seq_length), hidden_size]
    outputs = tf.reshape(tf.concat(1, outputs), [-1, config.hidden_size])

    # Softmax
    softmax_w = tf.get_variable('softmax_w',
                                [config.vocab_size, config.hidden_size])
    softmax_b = tf.get_variable('softmax_b', [config.vocab_size])
    self._logits = tf.matmul(outputs, tf.transpose(softmax_w)) + softmax_b
    self._probs = tf.nn.softmax(self._logits)

    # Average cross-entropy loss within the batch.
    batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      self._logits, tf.reshape(self._target_seq, [-1]))
    self._loss = tf.reduce_mean(batch_loss)
    self._perplexity = tf.exp(self._loss)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    self._train_op = optimizer.minimize(self._loss)

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
    return self._multi_cell.zero_state(self._config.batch_size, tf.float32)

  def sample_indices(self, sess, indices, length, temperature=1.0):
    def sample_next_index(_idx, _state):
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
    for idx in indices[:-1]:
      _, state = sample_next_index(idx, state)

    # Start sampling
    for _ in xrange(length):
      new_idx, state = sample_next_index(result[-1], state)
      result.append(new_idx)

    return result


# Class holding model configuration.

class CharacterModelConfig(object):
  def __init__(self, vocab_size, inference=False):
    # Default model parameters
    self.batch_size = 256
    self.hidden_size = 512
    self.seq_length = 50
    self.hidden_depth = 3
    self.keep_prob = 0.5
    self.learning_rate = 0.001
    self.vocab_size = vocab_size

    if inference:
      self.batch_size = self.seq_length = 1
      self.keep_prob = 1.0

  def __str__(self):
    return ('Model Config:\n' +
            '\n'.join(['  -> %s: %s' % (k,v)
                       for k,v in self.__dict__.iteritems()]))
