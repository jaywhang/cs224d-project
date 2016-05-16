# Helper class for loading training data for character-level models.

import numpy as np

class TextUtil(object):
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

  @property
  def text_size(self):
    return len(self._text)

  @property
  def num_batches(self):
    return self._num_batches

  def chars_to_indices(self, string):
    return [self._char_to_idx[ch] for ch in string]

  def indices_to_chars(self, indices):
    return [self._idx_to_char[idx] for idx in indices]

  def reset_batches(self, batch_size, seq_length):
    self._seq_length = seq_length
    self._batch_size = batch_size
    self._epoch_count = 0
    self._batch_pointer = 0

    self._num_batches = (len(self._text)-1) // (batch_size * seq_length)
    assert self.num_batches != 0, (
      "Data is not big enough for given parameters. "
      "Try reducing batch_size and/or seq_length.")

  def next_batch(self, as_string=False):
    # Restart from the beginning if we've already used the last batch.
    if self._batch_pointer >= self._num_batches:
      self._batch_pointer = 0
      self._epoch_count += 1

    xbatch = []
    ybatch = []
    offset = self._batch_pointer * self._seq_length
    width = self._num_batches * self._seq_length
    for i in xrange(self._batch_size):
      start = i * width + offset
      end = start + self._seq_length
      xbatch.append(self._data[start:end])
      ybatch.append(self._data[start+1:end+1])

    xbatch = np.vstack(xbatch)
    ybatch = np.vstack(ybatch)
    self._batch_pointer += 1

    if as_string:
      xbatch = (np.array(self.indices_to_chars(xbatch.flatten()))
                .reshape(xbatch.shape))
      ybatch = (np.array(self.indices_to_chars(ybatch.flatten()))
                .reshape(ybatch.shape))

    return xbatch, ybatch, self._epoch_count

  def single_epoch(self, batch_size, seq_length):
    self.reset_batches(batch_size, seq_length)

    while True:
      x, y, epoch = self.next_batch()
      if epoch > 0:
        break
      yield x, y
