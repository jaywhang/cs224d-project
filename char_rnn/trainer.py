# Training and sampling routines.

import time
import numpy as np
import tensorflow as tf
from text_util import *
from char_model import *


def train(sess, model, text_util, save_file=None, num_epochs=10,
          eval_model=None):
  tf.initialize_all_variables().run()

  print 'Starting training...\n'

   # Run batch gradient descent.
  state = model.zero_state.eval()
  iter = 0
  loss_pp_iter = []
  for epoch in xrange(num_epochs):
    start_time = time.time()

    # Perform one epoch of training.
    for inputs, labels in text_util.single_epoch(
      model.config.batch_size, model.config.seq_length):
      loss, perp, _, state = sess.run(
        [model.loss, model.perplexity, model.train_op, model.final_state],
      feed_dict={model.input_seq: inputs,
                 model.target_seq: labels,
                 model.initial_state: state})
      iter += 1

      # Record loss and perplexity at every 10 iterations
      if iter % 10 == 0:
        loss_pp_iter.append((loss, perp, iter))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print ('Epoch %d finished after %d iterations (%.2f sec). '
           'Perplexity: %.3f, Loss: %.3f'
           %(epoch, iter, elapsed_time, perp, loss))

    # Sample a sentence after every epoch.
    if eval_model:
      sampled_sentence = sample(sess, eval_model, text_util, '.', length=80)
      print '  Sample: %s' % sampled_sentence[1:]


  # Record final perplexity and loss
  loss_pp_iter.append((loss, perp, iter))

  print '\nTraining finished!'
  print '  -- Final perplexity: %.3f' % perp
  print '  -- Final loss: %.3f' % loss

  # Final sample sentence.
  if eval_model:
    sampled_sentence = sample(sess, eval_model, text_util, '.', length=80)
    print '  -- Sample sentence: %s' % sampled_sentence[1:]

  return loss_pp_iter


def sample(sess, model, text_util, string, length=40, temperature=1.0,
           verbose=False):
  assert string  # string shouldn't be empty
  if verbose:
    print '\nSampling from the following with temperature %.3f' % temperature
    print ' --> %s' % string
  sampled = model.sample_indices(
    sess, text_util.chars_to_indices(string), length, temperature=temperature)
  result = ''.join(text_util.indices_to_chars(sampled))
  if verbose:
    print 'Sampled string:\n --> %s' % result
  return result


def repl(sess, model, text_util, temperature=1.0):
  pass



