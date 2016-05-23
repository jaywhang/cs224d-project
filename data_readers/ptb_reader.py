# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf
from .base_reader import BaseReader


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict(zip(range(len(words)), words))

  return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data]


def ptb_raw_data(data_path=None, char_model=False):
  pattern = "ptb.char.{}.txt" if char_model else "ptb.{}.txt"
  train_path = os.path.join(data_path, pattern.format("train"))
  valid_path = os.path.join(data_path, pattern.format("valid"))
  test_path = os.path.join(data_path, pattern.format("test"))

  word_to_id, id_to_word = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, word_to_id, id_to_word


class PTBReader(BaseReader):
  def __init__(self, raw_data, word_to_id, id_to_word):
    self._raw_data = raw_data
    self._word_to_id = word_to_id
    self._id_to_word = id_to_word

  @property
  def vocab_size(self):
    return len(self._word_to_id)

  def iterator(self, batch_size, seq_length):
    raw_data = np.array(self._raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
      data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // seq_length

    if epoch_size == 0:
      raise ValueError("epoch_size == 0, decrease batch_size or seq_length")

    for i in range(epoch_size):
      x = data[:, i*seq_length:(i+1)*seq_length]
      y = data[:, i*seq_length+1:(i+1)*seq_length+1]
      yield x, y, i, epoch_size

  def encode(self, word_list):
    return [self._word_to_id[word] for word in word_list]

  def decode(self, index_list):
    return [self._id_to_word[index] for index in index_list]


# def ptb_iterator(raw_data, batch_size, num_steps):
#   """Iterate on the raw PTB data.
#
#   This generates batch_size pointers into the raw PTB data, and allows
#   minibatch iteration along these pointers.
#
#   Args:
#     raw_data: one of the raw data outputs from ptb_raw_data.
#     batch_size: int, the batch size.
#     num_steps: int, the number of unrolls.
#
#   Yields:
#     Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
#     The second element of the tuple is the same data time-shifted to the
#     right by one.
#
#   Raises:
#     ValueError: if batch_size or num_steps are too high.
#   """
#   raw_data = np.array(raw_data, dtype=np.int32)
#
#   data_len = len(raw_data)
#   batch_len = data_len // batch_size
#   data = np.zeros([batch_size, batch_len], dtype=np.int32)
#   for i in range(batch_size):
#     data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
#
#   epoch_size = (batch_len - 1) // num_steps
#
#   if epoch_size == 0:
#     raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
#
#   for i in range(epoch_size):
#     x = data[:, i*num_steps:(i+1)*num_steps]
#     y = data[:, i*num_steps+1:(i+1)*num_steps+1]
#     yield (x, y)
