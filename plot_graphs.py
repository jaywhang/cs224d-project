#!/bin/python2

import matplotlib
matplotlib.use('Agg')  # Avoid requiring X server.

import csv, os, sys
import numpy as np
import matplotlib.pyplot as plt


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
  plt.xlabel('Epoch')
  plt.ylabel('Valid Loss')
  for name, (_, eval_loss_pp) in data.iteritems():
    iters, _, _, valid_losses, _ = zip(*eval_loss_pp)
    plt.plot(iters, valid_losses, label=name)
  plt.legend(fontsize=8)
  plt.savefig(os.path.join(output_dir, 'valid_losses.pdf'))
  print('Created %s' % os.path.join(output_dir, 'valid_losses.pdf'))

  # Train losses across different models.
  plt.clf()
  plt.grid(True)
  plt.xlabel('Epoch')
  plt.ylabel('Train Loss')
  for name, (_, eval_loss_pp) in data.iteritems():
    iters, train_losses, _, _, _ = zip(*eval_loss_pp)
    plt.plot(iters, train_losses, label=name)
  plt.legend(fontsize=8)
  plt.savefig(os.path.join(output_dir, 'train_losses.pdf'))
  print('Created %s' % os.path.join(output_dir, 'train_losses.pdf'))


def main(argv):
  if len(argv) < 2:
    print('Usage: %s [folders]' % argv[0])
    sys.exit(1)

  dirs = []
  for d in argv[1:]:
    if os.path.isdir(d):
      dirs.append(d)

  print '\n   '.join(['\nFound valid folders:\n'] + dirs)

  data = {}
  for d in dirs:
    config_name = os.path.basename(d)
    data[config_name] = read_training_info(d)

  print

  save_plots(data, '.')


if __name__ == "__main__":
  main(sys.argv)
