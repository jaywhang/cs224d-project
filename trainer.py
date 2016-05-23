
from copy import deepcopy
import sys, time
import numpy as np
import tensorflow as tf

from data_readers import text_reader, ptb_reader
from char_model import CharacterModel, CharacterModelConfig

flags, logging = tf.flags, tf.logging

flags.DEFINE_string('data_path', None, 'path to data file/folder')
flags.DEFINE_string('data_type', 'text', 'type of training data.')
flags.DEFINE_string('sample_during_training', True,
                    'if True, produce sample phrases after every epoch')

FLAGS = flags.FLAGS


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to data file or directory")

  if FLAGS.data_type == 'text':
    train_reader = text_reader.TextReader(FLAGS.data_path)
    valid_reader = text_reader.TextReader(FLAGS.data_path)
    test_reader = text_reader.TextReader(FLAGS.data_path)
    print 'WARNING: This implementation is incomplete.'
  elif FLAGS.data_type == 'ptb':
    raw_data = ptb_reader.ptb_raw_data(FLAGS.data_path, char_model=True)
    train_data, valid_data, test_data, w2i, i2w = raw_data
    train_reader = ptb_reader.PTBReader(train_data, w2i, i2w)
    valid_reader = ptb_reader.PTBReader(valid_data, w2i, i2w)
    test_reader = ptb_reader.PTBReader(test_data, w2i, i2w)
  else:
    raise ValueError('Invalid data_type %s. Must be "text" or "ptb"'
                     % FLAGS.data_type)

  train_config = CharacterModelConfig(train_reader.vocab_size)
  # train_config.hidden_depth = 1
  train_config.batch_size = 64
  train_config.hidden_size = 512
  train_config.learning_rate = 0.002
  train_config.seq_length = 50
  train_config.max_epoch = 50
  train_config.keep_prob = 0.5

  valid_config = deepcopy(train_config)
  valid_config.batch_size = valid_config.seq_length = 1
  valid_config.keep_prob = 1.0
  test_config = deepcopy(valid_config)

  print (train_config)

  with tf.Graph().as_default(), tf.Session() as sess:
    initializer = None
    with tf.variable_scope('model', reuse=None):
      train_model = CharacterModel(train_config)
    with tf.variable_scope('model', reuse=True):
      valid_model = CharacterModel(valid_config)
      test_model = CharacterModel(test_config)

    tf.initialize_all_variables().run()
    losses = []
    iters_total = 0

    for i in xrange(1, train_config.max_epoch+1):
      print('Starting epoch %d / %d' % (i, train_config.max_epoch))
      train_iterator = train_reader.iterator(
          train_config.batch_size, train_config.seq_length)
      new_losses, num_batches = train_model.run_epoch(
          sess, train_config, train_iterator)
      losses.extend(new_losses)
      iters_total += num_batches
      print(' --> Average loss: %.4f' % np.mean(new_losses))

      if FLAGS.sample_during_training:
        sample = test_reader.decode(
            test_model.sample(sess, test_reader.encode('I'), 20))
        sample = ''.join(sample)
        print(' --> Sample from epoch %d:' % i)
        print('---------------------------')
        print(sample)
        print('---------------------------\n')

      # TODO: Add valid + test loss calculation


if __name__ == "__main__":
  tf.app.run()



