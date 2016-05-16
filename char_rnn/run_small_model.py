# Runs a small model for demo purpose.

from copy import deepcopy
import numpy as np
import tensorflow as tf
from text_util import TextUtil
from char_model import CharacterModel, CharacterModelConfig
from trainer import train, sample, repl


# We're ready to train this model now!
def main():
  util = TextUtil('input.txt')
  train_config = CharacterModelConfig(util.vocab_size)
  train_config.hidden_depth = 2
  train_config.batch_size = 256
  eval_config = deepcopy(train_config)
  eval_config.batch_size = eval_config.seq_length = 1
  eval_config.keep_prob = 1.0

  print train_config

  tf.reset_default_graph()

  with tf.variable_scope('model', reuse=None):
    train_model = CharacterModel(train_config)
  with tf.variable_scope('model', reuse=True):
    eval_model = CharacterModel(eval_config)

  with tf.Session() as sess:
    loss_pp_iter = train(sess, train_model, util, num_epochs=50,
                        eval_model=eval_model)
    for i in xrange(10):
      print '\n\nSample sentence %d' % (i+1)
      print sample(sess, eval_model, util, 'The', length=60)
      print '\n'

if __name__ == '__main__':
  main()

