import tensorflow as tf

import rnn_cell  # our own rnn_cell module.

# Class holding model configuration.
class CharacterModelLSTMConfig(object):

  def __init__(self, vocab_size):
    # Default model parameters
    self.batch_size = 64
    self.hidden_size = 500
    self.seq_length = 100
    self.init_scale = 0.1
    self.keep_prob = 0.5
    self.learning_rate = 0.002
    self.max_epoch = 50
    self.max_grad_norm = 1.0
    self.vocab_size = vocab_size
    self.optimizer = tf.train.AdamOptimizer
    self.cell_type = rnn_cell.BasicLSTMCell
    self.is_training = True

    self.hidden_depth = 1  # not used for now.

  def __str__(self):
    return ('Model Config:\n' +
            '\n'.join(['  -> %s: %s' % (k,v)
                       for k,v in self.__dict__.iteritems()]))

  def for_inference(self):
    self.is_training = False


class CharacterModelBNLSTMConfig(CharacterModelLSTMConfig):
  def __init__(self, vocab_size):
    CharacterModelLSTMConfig.__init__(self, vocab_size)
    self.cell_type = rnn_cell.BNLSTMCell
