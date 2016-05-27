#!/bin/python2

import matplotlib
matplotlib.use('Agg')  # Avoid requiring X server.

import csv, os, sys
import numpy as np
import matplotlib.pyplot as plt


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


# data = { name: (loss_pp, epoch_loss, epoch_pp) }
def save_plots(data, output_dir):
  # Valid loss per epoch across different models.
  plt.clf()
  plt.grid(True)
  plt.xlabel('Epoch')
  plt.ylabel('Valid Loss')
  for name, (_, epoch_loss, _) in data.iteritems():
    valid_losses = zip(*epoch_loss)[1]
    x = np.arange(len(epoch_loss))
    print name
    plt.plot(x, valid_losses, label=name)
  plt.legend(fontsize=8)
  plt.savefig(os.path.join(output_dir, 'valid_losses.pdf'))


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

  save_plots(data, '.')


if __name__ == "__main__":
  main(sys.argv)
