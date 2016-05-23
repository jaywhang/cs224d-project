from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn

from copy import deepcopy
import sys, time
import numpy as np

import ptb.word_reader as reader
from char_model import CharacterModel
from char_model import CharacterModelConfig

flags, logging = tf.flags, tf.logging

flags.DEFINE_string("data", None, "ptb data path")
FLAGS = flags.FLAGS


def run_epoch(sess, config, train_model, data, op,
              iters_total=0, verbose=False):
  start_time = time.time()
  iters = 0
  total_loss = 0.0
  iter_pp = []
  state = train_model.zero_state.eval()
  m = train_model

  # Run batch gradient descent for one epoch.
  for step, (inputs, labels) in enumerate(
      reader.ptb_iterator(data, config.batch_size, config.seq_length)):
    batch_loss, _, state = sess.run(
        [m.batch_loss, op, m.final_state],
        feed_dict={m.input_seq: inputs,
                   m.target_seq: labels,
                   m.initial_state: state})
    total_loss += batch_loss
    iters += 1
    iters_total += 1

    perp = np.exp(total_loss / config.batch_size / config.seq_length / iters)
    if iters_total % 50 == 0:
      iter_pp.append((iters_total, perp))

    if iters_total % 100 == 0 and verbose:
      print (" -- Iter %d: perplexity: %.3f speed: %.0f cps" %(
        iters_total, perp,
        iters * config.batch_size * config.seq_length / (time.time() - start_time)))
      sys.stdout.flush()

  perp = np.exp(total_loss / config.batch_size / config.seq_length / iters)
  iter_pp.append((iters_total, perp))
  return iter_pp, iters_total, perp


def main(_):
  if not FLAGS.data:
    raise ValueError("Must set --data to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data, char_model=True)
  train_data, valid_data, test_data, vocab_size = raw_data

  train_config = CharacterModelConfig(vocab_size)
  train_config.hidden_depth = 1
  train_config.batch_size = 64
  train_config.hidden_size = 500
  train_config.learning_rate = 0.002
  train_config.seq_length = 100
  train_config.max_epoch = 50

  valid_config = deepcopy(train_config)
  valid_config.batch_size = valid_config.seq_length = 1
  valid_config.keep_prob = 1.0
  test_config = deepcopy(valid_config)

  print (train_config)

  with tf.Graph().as_default(), tf.Session() as sess:
    # initializer = tf.random_uniform_initializer(-config.init_scale,
    #                                             config.init_scale)
    initializer = None
    with tf.variable_scope('model', reuse=None):
      train_model = CharacterModel(train_config)
    with tf.variable_scope('model', reuse=True):
      valid_model = CharacterModel(valid_config)
      test_model = CharacterModel(test_config)

    tf.initialize_all_variables().run()
    iter_pp = []
    iters_total = 0

    for i in range(train_config.max_epoch):
      _new_iter_pp, iters_total, train_perplexity = run_epoch(
          sess, train_config, train_model, train_data,
          train_model.train_op, iters_total=iters_total, verbose=True)
      iter_pp.extend(_new_iter_pp)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

      _, _, valid_perplexity = run_epoch(
          sess, valid_config, valid_model, valid_data, tf.no_op(), verbose=False)
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    _, _, test_perplexity = run_epoch(
        sess, test_config, test_model, test_data, tf.no_op(), verbose=False)
    print("Test Perplexity: %.3f" % test_perplexity)

  print (iter_pp)


if __name__ == "__main__":
  tf.app.run()

