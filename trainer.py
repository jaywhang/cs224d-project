import matplotlib
matplotlib.use('Agg')  # Avoid requiring X server.

from copy import deepcopy
import os, sys, time
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from data_readers import text_reader, ptb_reader
from char_model import CharacterModel
from configs import *

flags, logging = tf.flags, tf.logging

flags.DEFINE_string('config', None, 'config name')
flags.DEFINE_string('model_type', 'char', 'model type. "char" or "word"')
flags.DEFINE_string('data_path', None, 'path to data file/folder')
flags.DEFINE_string('data_type', 'text', 'type of training data.')
flags.DEFINE_string('plot_dir', None, 'folder path to loss/perplexity plot')
flags.DEFINE_string('sample_during_training', True,
                    'if True, produce sample phrases after every epoch')

FLAGS = flags.FLAGS


def get_config(vocab_size, inference=False):
  if not FLAGS.config or FLAGS.config not in ('lstm', 'bn_lstm'):
    raise ValueError("Invalid config.")
  elif FLAGS.config == "lstm":
    config = CharacterModelLSTMConfig(vocab_size)
  elif FLAGS.config == "bn_lstm":
    config = CharacterModelBNLSTMConfig(vocab_size)

  if inference:
    config.for_inference()

  return config


def save_plots(losses, perps, plot_dir):
  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

  x = np.arange(len(losses))
  plt.plot(x, losses, label='loss')
  plt.xlabel('iteration')
  plt.ylabel('loss')
  plt.legend()
  plt.savefig(os.path.join(plot_dir, 'loss.pdf'))

  plt.clf()
  plt.plot(x, perps, label='perplexity')
  plt.xlabel('iteration')
  plt.ylabel('perplexity')
  plt.legend()
  plt.savefig(os.path.join(plot_dir, 'perplexity.pdf'))


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to data file or directory")

  if FLAGS.data_type == 'text':
    # Plain text data doesn't have test/eval set.
    train_reader = text_reader.TextReader(FLAGS.data_path)
    valid_reader = test_reader = train_reader
  elif FLAGS.data_type == 'ptb':
    raw_data = ptb_reader.ptb_raw_data(
        FLAGS.data_path, char_model=(FLAGS.model_type=='char'))
    train_data, valid_data, test_data, w2i, i2w = raw_data
    train_reader = ptb_reader.PTBReader(train_data, w2i, i2w)
    valid_reader = ptb_reader.PTBReader(valid_data, w2i, i2w)
    test_reader = ptb_reader.PTBReader(test_data, w2i, i2w)
  else:
    raise ValueError('Invalid data_type %s. Must be "text" or "ptb"'
                     % FLAGS.data_type)

  train_config = get_config(train_reader.vocab_size)
  # Need to use train data's vocabulary for eval model.
  eval_config = get_config(train_reader.vocab_size, inference=True)
  if FLAGS.sample_during_training:
    sample_config = get_config(train_reader.vocab_size, inference=True)
    sample_config.seq_length = sample_config.batch_size = 1

  print (train_config)

  with tf.Graph().as_default(), tf.Session() as sess:
    initializer = None
    with tf.variable_scope('model', reuse=None):
      train_model = CharacterModel(train_config)
    with tf.variable_scope('model', reuse=True):
      eval_model = CharacterModel(eval_config)
      if FLAGS.sample_during_training:
        sample_model = CharacterModel(sample_config)

    tf.initialize_all_variables().run()
    losses, perps = [], []
    iters_total = 0

    print('Starting training.')
    train_start_time = time.time()

    for i in xrange(1, train_config.max_epoch+1):
      # Get new iterators.
      train_iterator = train_reader.iterator(train_config.batch_size,
                                             train_config.seq_length)
      valid_iterator = valid_reader.iterator(eval_config.batch_size,
                                             eval_config.seq_length)

      print('Starting epoch %d / %d' % (i, train_config.max_epoch))
      new_losses, new_perps, num_batches = train_model.run_epoch(
          sess, train_iterator)
      losses.extend(new_losses)
      perps.extend(new_perps)
      iters_total += num_batches
      print(' -- Average train loss: %.4f, perp: %.2f' %
            (np.mean(new_losses), np.mean(new_perps)))

      # Calculate validation loss.
      start_time = time.time()
      valid_losses, valid_perps, _ = eval_model.run_epoch(
          sess, valid_iterator, verbose=False)
      elapsed = time.time() - start_time
      print(' -- Valid loss: %.4f, perp: %.2f (took %.2f sec)' %
            (np.mean(valid_losses), np.mean(valid_perps), elapsed))

      if FLAGS.sample_during_training:
        if FLAGS.model_type == 'char':
          delim = ''
          start_token = 'i'
        else:
          delim = ' '
          start_token = '<eos>'
        sample = train_reader.decode(
            sample_model.sample(sess, train_reader.encode([start_token]), 30))
        elapsed = time.time() - start_time
        sample = delim.join(sample)
        print(' -- Sample from epoch %d (took %.2f sec)' % (i, elapsed))
        print('---------------------------')
        print(sample)
        print('---------------------------\n')

    total_train_time = time.time() - train_start_time
    print('\nTraining finished after %.2f sec' % total_train_time)

    # Calculate test loss.
    start_time = time.time()
    test_iterator = test_reader.iterator(eval_config.batch_size,
                                         eval_config.seq_length)
    test_losses, test_perps, _ = eval_model.run_epoch(
        sess, test_iterator, verbose=False)
    elapsed = time.time() - start_time
    print(' -- Test loss: %.4f, perp: %.2f (took %.2f sec)' %
          (np.mean(test_losses), np.mean(test_perps), elapsed))

    if FLAGS.plot_dir:
      save_plots(losses, perps, FLAGS.plot_dir)


if __name__ == "__main__":
  tf.app.run()

