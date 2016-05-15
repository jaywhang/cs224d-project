import tensorflow as tf
from batch_norm import *

class WordSmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  dropout_keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  cell_class = tf.nn.rnn_cell.BasicLSTMCell
  optimizer = tf.train.GradientDescentOptimizer

class WordBNSmallConfig(WordSmallConfig):
  """Small config."""
  cell_class = BNLSTMCell

class WordMediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  dropout_keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  cell_class = tf.nn.rnn_cell.BasicLSTMCell
  optimizer = tf.train.GradientDescentOptimizer

class WordBNMediumConfig(WordMediumConfig):
  """Small config."""
  cell_class = BNLSTMCell
