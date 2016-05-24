import rnn_cell

class DefaultLSTMConfig(object):

  def __init__(self, vocab_size):
    # Default model parameters
    self.batch_size = 64
    self.hidden_size = 500
    self.seq_length = 100
    self.init_scale = 0.1
    self.hidden_depth = 1  # not used
    self.keep_prob = 0.5
    self.learning_rate = 0.002
    self.max_epoch = 50
    self.max_grad_norm = 1.0
    self.vocab_size = vocab_size
    self.cell_type = rnn_cell.BasicLSTMCell

  def __str__(self):
    return ('Model Config:\n' +
            '\n'.join(['  -> %s: %s' % (k,v)
                       for k,v in self.__dict__.iteritems()]))

class LSTMConfig64(DefaultLSTMConfig):
  pass

class BNLSTMConfig64(DefaultLSTMConfig):
  def __init__(self, vocab_size):
    DefaultLSTMConfig.__init__(self, vocab_size)
    self.cell_type = rnn_cell.BNLSTMCell

class LSTMConfig128(DefaultLSTMConfig):
  def __init__(self, vocab_size):
    DefaultLSTMConfig.__init__(self, vocab_size)
    self.batch_size = 128

class BNLSTMConfig128(DefaultLSTMConfig):
  def __init__(self, vocab_size):
    DefaultLSTMConfig.__init__(self, vocab_size)
    self.cell_type = rnn_cell.BNLSTMCell
    self.batch_size = 128
