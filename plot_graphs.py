#!/bin/python2

import matplotlib
matplotlib.use('Agg')  # Avoid requiring X server.

import csv, os, sys
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

def read_training_info(output_dir):
  train_loss_pp, eval_loss_pp = [], []
  with open(os.path.join(output_dir, 'train_loss_pp.csv'), 'r') as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONE)
    for row in reader:
      train_loss_pp.append(row)

  with open(os.path.join(output_dir, 'eval_loss_pp.csv'), 'r') as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONE)
    for row in reader:
      eval_loss_pp.append(row)

  return train_loss_pp, eval_loss_pp

# data = { name: (train_loss_pp, eval_loss_pp) }
def save_plots(data, output_dir):
  # Valid losses across different models.
  plt.clf()
  plt.grid(True)
  plt.xlabel('Training Iterations', fontsize=18)
  plt.ylabel('Loss', fontsize=18)
  for name, (_, eval_loss_pp) in data.iteritems():
    iters, _, _, valid_losses, _ = zip(*eval_loss_pp)
    plt.plot(iters, valid_losses, label=name, linewidth=2.0)
  plt.legend(fontsize=18)
  plt.savefig(os.path.join(output_dir, 'valid_losses.pdf'))
  print('Created %s' % os.path.join(output_dir, 'valid_losses.pdf'))

  # Valid pps across different models.
  plt.clf()
  plt.grid(True)
  plt.xlabel('Training Iterations', fontsize=18)
  plt.ylabel('Perplexity', fontsize=18)
  for name, (_, eval_loss_pp) in data.iteritems():
    iters, _, _, _, valid_pps = zip(*eval_loss_pp)
    plt.plot(iters, valid_pps, label=name, linewidth=2.0)
  plt.legend(fontsize=18)
  plt.savefig(os.path.join(output_dir, 'valid_pps.pdf'))
  print('Created %s' % os.path.join(output_dir, 'valid_pps.pdf'))

  # # Train losses across different models.
  # plt.clf()
  # plt.grid(True)
  # plt.xlabel('Training Iterations')
  # plt.ylabel('Train Loss')
  # for name, (_, eval_loss_pp) in data.iteritems():
  #   iters, train_losses, _, _, _ = zip(*eval_loss_pp)
  #   plt.plot(iters, train_losses, label=name)
  # plt.legend(fontsize=8)
  # plt.savefig(os.path.join(output_dir, 'train_losses.pdf'))
  # print('Created %s' % os.path.join(output_dir, 'train_losses.pdf'))


def main(argv):
  if len(argv) < 2:
    print('Usage: %s [-a] folder1 [label1] folder2 [label2] ...' % argv[0])
    sys.exit(1)

  # Create {label: training_data} dict.
  data = OrderedDict()
  if argv[1] == '-a':
    for d in argv[2:]:
      if os.path.isdir(d):
        label = os.path.basename(d)
        data[label] = read_training_info(d)
      else:
        raise ValueError('Not a directory: %s' % d)
  else:
    for i in xrange(1, len(argv), 2):
      d = argv[i]
      label = argv[i+1]
      if os.path.isdir(d):
        data[label] = read_training_info(d)
      else:
        raise ValueError('Not a directory: %s' % d)

  print '\n   '.join(['\nPlots to be generated:\n'] + data.keys())
  print

  save_plots(data, '.')


if __name__ == "__main__":
  main(sys.argv)
