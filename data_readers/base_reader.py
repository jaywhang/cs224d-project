# Base class for data readers.
# Subclasses should implement the following methods.

class BaseReader(object):
  @property
  def vocab_size(self):
    # Returns number of unique tokens in the loaded data.
    raise NotImplemented

  def iterator(self, batch_size, seq_length):
    # Generator that yields (input, label, current, total) tuple where
    # `current` is 0-indexed index for the example, and `total` is the total
    # number of examples.
    raise NotImplemented

  def encode(self, token_list):
    raise NotImplemented

  def decode(self, index_list):
    raise NotImplemented
