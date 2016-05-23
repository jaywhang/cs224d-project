# Helper class for loading plain text data for character-level models.

import numpy as np
from base_reader import BaseReader

class TextReader(BaseReader):
  def __init__(self, file_path):
    with open(file_path) as f:
      self._text = (''.join(f.readlines()))[:-1]

    # Generate vocabulary (i.e. unique characters in the text)
    self._char_to_idx = {}
    self._idx_to_char = {}
    idx = 0
    for char in set(self._text):
      self._char_to_idx[char] = idx
      self._idx_to_char[idx] = char
      idx += 1

    # Store the copy of text in indices.
    self._data = np.array([self._char_to_idx[ch] for ch in self._text])

    print 'Successfully read %s' % file_path
    print '  -- Total characters: %d' % len(self._text)
    print '  -- Unique characters: %d' % len(self._char_to_idx)
    print

  @property
  def vocab_size(self):
    return len(self._char_to_idx)

  def encode(self, string):
    return [self._char_to_idx[ch] for ch in string]

  def decode(self, indices):
    return [self._idx_to_char[idx] for idx in indices]

  def iterator(self, batch_size, seq_length):
    num_batches = (len(self._text)-1) // (batch_size * seq_length)
    assert num_batches != 0, (
      "Data is not big enough for given parameters. "
      "Try reducing batch_size and/or seq_length.")

    batch_idx = 0

    while batch_idx in xrange(num_batches):
      inputs, labels = [], []
      offset = batch_idx * seq_length
      width = num_batches * seq_length
      for i in xrange(batch_size):
        start = i * width + offset
        end = start + seq_length
        inputs.append(self._data[start:end])
        labels.append(self._data[start+1:end+1])
      inputs  = np.vstack(inputs)
      labels = np.vstack(labels)
      yield inputs, labels, batch_idx, num_batches
      batch_idx += 1

  def reset_batches(self, batch_size, seq_length):

    self._num_batches = (len(self._text)-1) // (batch_size * seq_length)
    assert self.num_batches != 0, (
      "Data is not big enough for given parameters. "
      "Try reducing batch_size and/or seq_length.")

