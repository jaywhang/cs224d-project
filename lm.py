from __future__ import absolute_import

import tensorflow as tf
from tensorflow.models.rnn import rnn

class RNNLanguageModel(object):
  """
  A class to construct the computation graph for an RNN langauge model
  This class saves values on self for use externally:
    - self.input_data, self.target
    - self.initial_state
    - self.final_state
    - self.cost
    - self.lr
    - self.train_op if is_training=True
  """
  def __init__(self, config, is_training=True):
    self.config = config
    batch_size = config.batch_size
    num_steps = config.num_steps
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    dropout_keep_prob = config.dropout_keep_prob

    self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    cell = config.cell_class(hidden_size, forget_bias=0.0)
    if is_training:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell,
          input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob)

    self.initial_state = cell.zero_state(batch_size, tf.float32)

    embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
    input_embeddings = tf.gather(embedding, self.input_data)

    # input_embeddings is one (batch_size x num_steps x hidden_size) tensor
    # we want to transform this into a list of num_steps tensors of dims
    # [(batch_size x hidden_size)]
    inputs_by_timestep = [tf.squeeze(batch_for_timestep, [1])
       for batch_for_timestep in tf.split(1, num_steps, input_embeddings)]

    # this function unrolls the recursion. returns list of outputs for
    # softmax classification (dropped out) and the last hidden state.
    outputs, self.final_state = rnn.rnn(
        cell, inputs_by_timestep, initial_state = self.initial_state)

    # outputs is a list of (batch_size x hidden_size) tensors, and the list
    # is step_size long.  flatten them so that it is
    # ((batch_size * step_size) x hidden_size) long so we can make the
    # softmax prediction all at once.
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self.targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self.cost = cost = tf.reduce_sum(loss) / batch_size

    if not is_training:
      return

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = self.config.optimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
