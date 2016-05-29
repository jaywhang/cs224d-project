# Base class for data readers.
# Subclasses should implement the following methods.

class BaseReader(object):
  @property
  def vocab_size(self):
    # Returns number of unique tokens in the loaded data.
    raise NotImplemented

  def iterator(self, batch_size, seq_length, shuffle=True):
    # Generator that yields (input, label, current, total) tuple where
    # `current` is 0-indexed index for the example, and `total` is the total
    # number of examples.
    # Shuffles examples if shuffle=True.
    raise NotImplemented

  def endless_iterator(self, batch_size, seq_length, shuffle=True):
    # Generator that yields unlimited number of training examples by going
    # through multiple epochs if necessary.  Each yielded item is
    # (input, label, iter, epoch_size, epoch) where `iter` is the total number
    # of items produced so far, and `epoch_size` is the number of examples in
    # each epoch.
    # Shuffles examples if shuffle=True.
    raise NotImplemented

  def encode(self, token_list):
    raise NotImplemented

  def decode(self, index_list):
    raise NotImplemented
