import matplotlib
matplotlib.use('Agg')  # Avoid requiring X server.

from copy import deepcopy
import csv, os, sys, time
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from data_readers import text_reader, ptb_reader
from char_model import CharacterModel
from configs import CharacterModelLSTMConfig, FLAG_TO_NAME_MAP

flags, logging = tf.flags, tf.logging

flags.DEFINE_string('model_type', 'char', 'model type. "char" or "word"')
flags.DEFINE_string('data_path', None, 'path to data file/folder')
flags.DEFINE_string('data_type', 'text', 'type of training data.')
flags.DEFINE_string('output_dir', None, 'folder path to dump output files to.')

# For optionally overwriting hyperparameter values.
flags.DEFINE_string('ct', None, 'cell_type')
flags.DEFINE_integer('me', None, 'max_epoch')
flags.DEFINE_integer('bs', None, 'batch_size')
flags.DEFINE_integer('sl', None, 'seq_length')
flags.DEFINE_float('lr', None, 'learning_rate')
flags.DEFINE_integer('hs', None, 'hidden_size')
flags.DEFINE_float('kp', None, 'keep_prob')
flags.DEFINE_float('mgn', None, 'max_grad_norm')
# how often to evaluate on valid data.  either not set or a number.
flags.DEFINE_integer('ef', None, 'eval_frequency')
flags.DEFINE_string('op', None, 'optimizer')
# time step to start using last pop statistics
flags.DEFINE_integer('pop', None, 'pop_step')


FLAGS = flags.FLAGS


def get_config(vocab_size, inference=False):
  config = CharacterModelLSTMConfig(vocab_size)

  # Override values specified in commandline flags.
  params = ['ct', 'me', 'bs', 'sl', 'lr', 'hs', 'kp', 'mgn', 'ef', 'op', 'pop']

  for param in params:
    value = getattr(FLAGS, param)
    if value is not None:
      setattr(config, FLAG_TO_NAME_MAP[param], value)

  if inference:
    config.is_training = False

  return config


def save_training_info(config, test_loss, test_perp, output_dir):
  config_string = str(config)
  with open(os.path.join(output_dir, 'train_summary.txt'), 'w') as f:
    f.write(config_string + '\n')
    f.write('Test loss: %.4f, perplexity: %.2f' % (test_loss, test_perp))


def save_losses(train_loss_pp, eval_loss_pp, output_dir):
  with open(os.path.join(output_dir, 'train_loss_pp.csv'), 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)
    writer.writerows(train_loss_pp)

  with open(os.path.join(output_dir, 'eval_loss_pp.csv'), 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)
    writer.writerows(eval_loss_pp)


def save_plots(train_loss_pp, eval_loss_pp, output_dir):
  train_iters, train_losses, train_pps = zip(*train_loss_pp)
  eval_iters, eval_tl, eval_tp, eval_vl, eval_vp = zip(*eval_loss_pp)

  def plot_single(x, y1, y2, ylabel, filename,
                  legend=False, y1_label=None, y2_label=None):
    plt.clf()
    plt.grid(True)
    plt.plot(x, y1, label=y1_label)
    if y2:
      plt.plot(x, y2, label=y2_label)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    if legend: plt.legend()
    plt.savefig(os.path.join(output_dir, filename))

  # Plot of train loss from every iteration.
  plot_single(train_iters, train_losses, None, 'Train Loss', 'train_loss.pdf')

  # Plot of train perplexity from every iteration.
  plot_single(train_iters, train_pps, None, 'Train Perplexity', 'train_perplexity.pdf')

  # Sampled train vs valid losses.
  plot_single(eval_iters, eval_tl, eval_vl, 'Loss', 'eval_loss.pdf',
              legend=True, y1_label='Train', y2_label='Valid')

  # Sampled train vs valid perplexities.
  plot_single(eval_iters, eval_tp, eval_vp, 'Perplexity', 'eval_perplexity.pdf',
              legend=True, y1_label='Train', y2_label='Valid')


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

  print (train_config)

  with tf.Graph().as_default(), tf.Session() as sess:
    initializer = None
    with tf.variable_scope('model', reuse=None):
      train_model = CharacterModel(train_config)
    with tf.variable_scope('model', reuse=True):
      eval_model = CharacterModel(eval_config)

    saver = tf.train.Saver()

    tf.initialize_all_variables().run()
    # Losses and perplexities from all iterations from train data.
    # Contains (iteration, loss, pp) tuples.
    train_loss_pp = []
    # Train and valid losses/pp, either sampled every epoch or eval_frequency
    # if given.
    # Contains (iteration, train_loss, train_pp, valid_loss, valid_pp) tuples.
    eval_loss_pp = []

    if FLAGS.output_dir:
      outdir = os.path.join(FLAGS.output_dir, train_config.filename())
      if not os.path.exists(outdir):
        os.makedirs(outdir)

    test_writer = tf.train.SummaryWriter(
        os.path.join('tensorboard', train_config.filename(), 'test'),
        sess.graph, flush_secs=30)

    # Get endless iterator for training, and single-epoch iterator for eval.
    train_iterator = train_reader.endless_iterator(train_config.batch_size,
                                                    train_config.seq_length)
    valid_iterator = valid_reader.iterator(eval_config.batch_size,
                                           eval_config.seq_length)

    last_epoch = 0  # both epoch and iter are 1-based.
    train_start_time = epoch_start_time = time.time()
    print('Starting training.')
    total_start_time = time.time()
    for inputs, labels, iter, epoch_size, cur_epoch in train_iterator:
      if cur_epoch > last_epoch:
        last_epoch = cur_epoch

        if cur_epoch > train_config.max_epoch:
          break
        else:
          print('\r[Epoch %d / %d started]                     '
                % (cur_epoch, train_config.max_epoch))

      # We're not feeding in initial state on purpose.
      loss, pp = train_model.train(sess, inputs, labels)
      if iter % 20 == 0:
        train_loss_pp.append((iter, loss, pp))

      # We've reached the end of epoch.
      if iter % epoch_size == 0:
        elapsed = time.time() - epoch_start_time
        epoch_start_time = time.time()
        print('\r[Epoch %d / %d finished in %.2f sec]           '
              % (cur_epoch, train_config.max_epoch, elapsed))

      # Sample eval loss and pp.
      if ((train_config.eval_frequency is None  # every epoch
           and iter % epoch_size == 0) or
          (train_config.eval_frequency is not None  # every eval_frequency
           and iter % train_config.eval_frequency == 0)):

        train_elapsed = time.time() - train_start_time

        # Calculate validation loss.
        valid_start_time = time.time()
        valid_losses, valid_pps, _ = eval_model.run_epoch(
            sess, valid_iterator, verbose=False,
            summary_writer=test_writer, step=iter)
        valid_elapsed = time.time() - valid_start_time
        valid_loss, valid_pp = np.mean(valid_losses), np.mean(valid_pps)
        eval_loss_pp.append((iter, loss, pp, valid_loss, valid_pp))

        print('\r  Iter %d                                  ' % iter)
        print('\r    -- Train loss: %.4f, perp: %.2f (took %.2f sec)'
              % (loss, pp, train_elapsed))
        print('\r    -- Valid loss: %.4f, perp: %.2f (took %.2f sec)'
              % (valid_loss, valid_pp, valid_elapsed))

        save_plots(train_loss_pp, eval_loss_pp, outdir)
        save_losses(train_loss_pp, eval_loss_pp, outdir)

        # Reset valid iterator.
        valid_iterator = valid_reader.iterator(eval_config.batch_size,
                                              eval_config.seq_length)
        train_start_time = time.time()

      if (iter % epoch_size) % 10 == 0:
        _cur_iter = iter % epoch_size
        if _cur_iter == 0: _cur_iter = epoch_size
        sys.stdout.write('\r{} / {} : loss = {:.4f}, pp = {:.2f}'.format(
          _cur_iter, epoch_size, loss, pp))
        sys.stdout.flush()

      if FLAGS.output_dir and iter % epoch_size == 0 and cur_epoch % 5 == 0:
        saver.save(sess, os.path.join(outdir, 'parameters'))
        print("Checkpointed parameters")

    # Final sample of eval loss and pp.
    valid_start_time = time.time()
    valid_losses, valid_pps, _ = eval_model.run_epoch(
        sess, valid_iterator, verbose=False,
        summary_writer=test_writer, step=iter)
    valid_elapsed = time.time() - valid_start_time
    valid_loss, valid_pp = np.mean(valid_losses), np.mean(valid_pps)
    eval_loss_pp.append((iter, loss, pp, valid_loss, valid_pp))

    print('\r -- [Iter %d] Train loss: %.4f, perp: %.2f' % train_loss_pp[-1])
    print('\r -- [Iter %d] Valid loss: %.4f, perp: %.2f (took %.2f sec)' %
          (iter, valid_loss, valid_pp, valid_elapsed))

    # Training is done at this point.
    total_train_time = time.time() - total_start_time
    print('Training finished after %.2f sec' % total_train_time)

    # Calculate test loss.
    start_time = time.time()
    test_iterator = test_reader.iterator(eval_config.batch_size,
                                         eval_config.seq_length)
    test_losses, test_perps, _ = eval_model.run_epoch(
        sess, test_iterator, verbose=False)
    elapsed = time.time() - start_time
    print(' -- Test loss: %.4f, perp: %.2f (took %.2f sec)' %
          (np.mean(test_losses), np.mean(test_perps), elapsed))

    if FLAGS.output_dir:
      save_plots(train_loss_pp, eval_loss_pp, outdir)
      save_losses(train_loss_pp, eval_loss_pp, outdir)
      saver.save(sess, os.path.join(outdir, 'parameters'))
      save_training_info(train_config, np.mean(test_losses),
                         np.mean(test_perps), outdir)


if __name__ == "__main__":
  tf.app.run()
