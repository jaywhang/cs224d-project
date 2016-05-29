from collections import OrderedDict
import tensorflow as tf

import rnn_cell  # our own rnn_cell module.


FLAG_TO_NAME_MAP = {
    'bs': 'batch_size',
    'hs': 'hidden_size',
    'sl': 'seq_length',
    'kp': 'keep_prob',
    'lr': 'learning_rate',
    'me': 'max_epoch',
    'mgn': 'max_grad_norm',
    'vs': 'vocab_size',
    'op': 'optimizer',
    'ct': 'cell_type',
}


# Class holding model configuration.
class CharacterModelLSTMConfig(object):
  def __init__(self, vocab_size):
    # Default model parameters
    self.batch_size = 64
    self.hidden_size = 500
    self.seq_length = 100
    self.keep_prob = 0.5
    self.learning_rate = 0.002
    self.max_epoch = 50
    self.max_grad_norm = 1.0
    self.vocab_size = vocab_size
    self.optimizer = 'adam'  # or 'sgd'
    self.cell_type = 'lstm'  # or 'bnlstm', 'gru', 'bngru.full', 'bngru.simple'
    self.is_training = True

    # self.hidden_depth = 1  # not supported.

  def __str__(self):
    return ('Model Config:\n' +
            '\n'.join(sorted(['  -> %s: %s' % (k,v)
                      for k,v in self.__dict__.iteritems()])))

  def filename(self):
    ordered_map = OrderedDict(sorted(FLAG_TO_NAME_MAP.items(),
                                     key=lambda x: x[0]))

    pairs = [flag + '_' + str(getattr(self, name))
             for flag, name in ordered_map.iteritems() if flag != 'me']
    return '_'.join(pairs)
