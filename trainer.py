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
flags.DEFINE_bool('sample_during_training', False,
                    'if True, produce sample phrases after every epoch')
flags.DEFINE_bool('restart_training', False,
                    'if True, restart training from saved parameters')

# For optionally overwriting hyperparameter values.
flags.DEFINE_string('ct', None, 'cell_type')
flags.DEFINE_integer('me', None, 'max_epoch')
flags.DEFINE_integer('bs', None, 'batch_size')
flags.DEFINE_integer('sl', None, 'seq_length')
flags.DEFINE_float('lr', None, 'learning_rate')
flags.DEFINE_integer('hs', None, 'hidden_size')
flags.DEFINE_float('kp', None, 'keep_prob')


FLAGS = flags.FLAGS


def get_config(vocab_size, inference=False):
  config = CharacterModelLSTMConfig(vocab_size)

  # Override values specified in commandline flags.
  params = ['ct', 'me', 'bs', 'sl', 'lr', 'hs', 'kp']

  for param in params:
    value = getattr(FLAGS, param)
    if value:
      setattr(config, FLAG_TO_NAME_MAP[param], value)

  if inference:
    config.is_training = False

  return config


def save_training_info(config, test_loss, test_perp, output_dir):
  config_string = str(config)
  with open(os.path.join(output_dir, 'train_summary.txt'), 'w') as f:
    f.write(config_string + '\n')
    f.write('Test loss: %.4f, perplexity: %.2f' % (test_loss, test_perp))

def read_training_info(output_dir):
  loss_pp, epoch_losses, epoch_perps = [], [], []
  with open(os.path.join(output_dir, 'iter_loss_pp.csv'), 'r') as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONE)
    for row in reader:
      loss_pp.append((row[1], row[2]))

  with open(os.path.join(output_dir, 'train_valid_loss.csv'), 'r') as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONE)
    for row in reader:
      epoch_losses.append((row[1], row[2]))

  with open(os.path.join(output_dir, 'train_valid_perplexity.csv'), 'r') as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONE)
    for row in reader:
      epoch_perps.append((row[1], row[2]))

  return loss_pp, epoch_losses, epoch_perps

def save_losses(loss_pp, epoch_losses, epoch_perps, output_dir):
  with open(os.path.join(output_dir, 'iter_loss_pp.csv'), 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)
    for i, (loss, perp) in enumerate(loss_pp):
      writer.writerow((i+1, loss, perp))

  with open(os.path.join(output_dir, 'train_valid_loss.csv'), 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)
    for i, (train_loss, valid_loss) in enumerate(epoch_losses):
      writer.writerow((i+1, train_loss, valid_loss))

  with open(os.path.join(output_dir, 'train_valid_perplexity.csv'), 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)
    for i, (train_perp, valid_perp) in enumerate(epoch_perps):
      writer.writerow((i+1, train_perp, valid_perp))


def save_plots(loss_pp, epoch_losses, epoch_perps, output_dir):
  if FLAGS.restart_training: return
  x = np.arange(len(epoch_losses))
  plt.grid(True)

  # Train vs valid loss on.
  train_losses, valid_losses = zip(*epoch_losses)
  plt.plot(x, train_losses, label='Train')
  plt.plot(x, valid_losses, label='Valid')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(os.path.join(output_dir, 'loss.pdf'))

  plt.clf()
  plt.grid(True)
  # Train vs valid perplexity.
  train_pp, valid_pp = zip(*epoch_perps)
  plt.plot(x, train_pp, label='Train')
  plt.plot(x, valid_pp, label='Valid')
  plt.xlabel('Epoch')
  plt.ylabel('Perplexity')
  plt.legend()
  plt.savefig(os.path.join(output_dir, 'perplexity.pdf'))

  plt.clf()
  plt.grid(True)
  # Plot of train loss from every iteration.
  losses, perps = zip(*loss_pp)
  x = np.arange(len(loss_pp))
  plt.plot(x, losses)
  plt.xlabel('Iteration')
  plt.ylabel('Train Loss')
  plt.savefig(os.path.join(output_dir, 'train_loss.pdf'))

  plt.clf()
  plt.grid(True)
  # Plot of train perplexity from every iteration.
  plt.plot(x, perps)
  plt.plot(x, perps)
  plt.xlabel('Iteration')
  plt.ylabel('Train Perplexity')
  plt.savefig(os.path.join(output_dir, 'train_perplexity.pdf'))

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

    saver = tf.train.Saver()

    tf.initialize_all_variables().run()
    # Losses and perplexities from all iterations from train data.
    train_loss_pp = []
    # Per-epoch train and valid losses.
    epoch_losses = []
    epoch_perps = []

    if FLAGS.output_dir:
      outdir = os.path.join(FLAGS.output_dir, train_config.filename())
      if not os.path.exists(outdir):
        os.makedirs(outdir)

    if FLAGS.restart_training:
      if not FLAGS.output_dir:
        raise ValueError("Set restart_training flag but not output_dir flag")
      train_loss_pp, epoch_losses, epoch_perps = read_training_info(outdir)
      saver.restore(sess, outdir + "/parameters")
      print("Loaded %s epochs and last weights!" % len(epoch_losses))

    if (len(epoch_losses) >= train_config.max_epoch):
      print("Already exceeded max epochs of training: %s, %s" %
              (len(epoch_losses, train_config.max_epoch)))
      return

    print('Starting training.')
    train_start_time = time.time()

    for i in xrange(len(epoch_losses)+1, train_config.max_epoch+1):
      # Get new iterators.
      train_iterator = train_reader.iterator(train_config.batch_size,
                                             train_config.seq_length)
      valid_iterator = valid_reader.iterator(eval_config.batch_size,
                                             eval_config.seq_length)

      print('Starting epoch %d / %d' % (i, train_config.max_epoch))
      new_losses, new_perps, num_batches = train_model.run_epoch(
          sess, train_iterator)
      train_loss_pp.extend(zip(new_losses, new_perps))
      print(' -- Train loss: %.4f, perp: %.2f' %
            (new_losses[-1], new_perps[-1]))

      # Calculate validation loss.
      start_time = time.time()
      valid_losses, valid_perps, _ = eval_model.run_epoch(
          sess, valid_iterator, verbose=False)
      elapsed = time.time() - start_time
      print(' -- Valid loss: %.4f, perp: %.2f (took %.2f sec)' %
            (np.mean(valid_losses), np.mean(valid_perps), elapsed))

      epoch_losses.append((new_losses[-1], np.mean(valid_losses)))
      epoch_perps.append((new_perps[-1], np.mean(valid_perps)))

      if FLAGS.output_dir:
        save_plots(train_loss_pp, epoch_losses, epoch_perps, outdir)
        save_losses(train_loss_pp, epoch_losses, epoch_perps, outdir)
        if i % 5 == 0:
          saver.save(sess, os.path.join(outdir, 'parameters'))
          print("Checkpointed parameters")

      print("")

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

    if FLAGS.output_dir:
      save_plots(train_loss_pp, epoch_losses, epoch_perps, outdir)
      save_losses(train_loss_pp, epoch_losses, epoch_perps, outdir)
      save_training_info(train_config, np.mean(test_losses),
                         np.mean(test_perps), outdir)
      saver.save(sess, os.path.join(outdir, 'parameters'))


if __name__ == "__main__":
  tf.app.run()
