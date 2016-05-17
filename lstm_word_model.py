from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn

import sys, time
import numpy as np

import ptb.word_reader as reader
from lm import RNNLanguageModel
from configs import *

flags, logging = tf.flags, tf.logging

flags.DEFINE_string("data", None, "ptb data path")
flags.DEFINE_string("config", None, "config name")
FLAGS = flags.FLAGS

def get_config():
  if not FLAGS.config:
    raise ValueError("Must set --config")
  if FLAGS.config == "small":
    return WordSmallConfig()
  elif FLAGS.config == "bn_small":
    return WordBNSmallConfig()
  if FLAGS.config == "medium":
    return WordMediumConfig()
  elif FLAGS.config == "bn_medium":
    return WordBNMediumConfig()
  else:
    raise ValueError("Unknown config")

def run_epoch(session, config, m, data, eval_op, iters_total=0, verbose=False):
  start_time = time.time()
  costs, iters = 0.0, 0
  state = session.run(m.initial_state)

  for step, (x, y) in enumerate(reader.ptb_iterator(data, config.batch_size,
                                                    config.num_steps)):
    cost, _, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += config.num_steps
    iters_total += config.num_steps

    if verbose and iters_total % 1000 == 0:
      print("%s perplexity: %.3f speed: %.0f wps" %
            (iters_total, np.exp(costs / iters),
             iters * config.batch_size / (time.time() - start_time)))
      sys.stdout.flush()

  return np.exp(costs / iters), iters_total

def main(_):
  if not FLAGS.data:
    raise ValueError("Must set --data to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = RNNLanguageModel(config=config, is_training=True)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      # These share the same trained parameters (tf variables) as above
      # but different input placeholders/output tensors.
      mvalid = RNNLanguageModel(is_training=False, config=config)
      mtest = RNNLanguageModel(is_training=False, config=eval_config)

    tf.initialize_all_variables().run()

    iters_total = 0

    train_writer = tf.train.SummaryWriter('train', session.graph)

    for i in range(config.max_max_epoch):
      #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      session.run(tf.assign(m.lr, config.learning_rate))

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity, iters_total = run_epoch(session, config, m, train_data,
          m.train_op, iters_total=iters_total, verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

      valid_perplexity, _ = run_epoch(session, config, mvalid, valid_data, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity, _ = run_epoch(session, eval_config, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
