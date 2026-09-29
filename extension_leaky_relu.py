"""Unique extension for Project 2 (Anjana Anand).

Question: does the paper's Figure 3 finding -- that NNGP predictive
variance correlates with actual squared error -- still hold for a
nonlinearity the paper never tested?

Lee et al. (ICLR 2018) derive and test only ReLU and tanh kernels. But
nngp.py's kernel computation is nonlinearity-agnostic: _compute_qmap_grid
numerically integrates nonlin_fn(z) over a Gaussian grid rather than
requiring a hand-derived closed form (the paper's own ReLU/tanh kernels,
Sec. 3 / App. B, are closed forms plugged into this same numerical
machinery purely for speed). This script exercises that path with Leaky
ReLU (tf.nn.leaky_relu, negative slope 0.2), which has no precomputed grid
in grid_data/ and is therefore integrated from scratch on first run --
the "numerically-approximated kernel" extension option in the assignment.

Usage (same flag conventions as uncertainty_plot.py):
python extension_leaky_relu.py \
    --num_train=1000 --num_eval=1000 \
    --hparams='depth=3,weight_var=2.0,bias_var=0.2' \
    --output_file=/nngp/output/extension_leaky_relu.png

Note: the first run for leaky_relu will be noticeably slower than tanh/relu
(it's computing a fresh 501x501x500-point interpolation grid instead of
loading a cached one). Once grid_data/grid_leaky_relu_... exists, later
runs are as fast as the baseline.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import load_dataset
import uncertainty_plot as up

FLAGS = up.FLAGS


def main(argv):
  del argv  # Unused

  hparams = up.set_default_hparams().parse(FLAGS.hparams)

  tf.logging.info('Loading data')
  if FLAGS.dataset == 'mnist':
    (train_image, train_label, _, _, test_image,
     test_label) = load_dataset.load_mnist(
         num_train=FLAGS.num_train, mean_subtraction=True,
         random_roated_labels=False)
  elif FLAGS.dataset == 'cifar10':
    (train_image, train_label, _, _, test_image,
     test_label) = up.load_cifar10(
         num_train=FLAGS.num_train, mean_subtraction=True)
  else:
    raise NotImplementedError(FLAGS.dataset)

  # Compare the paper's two nonlinearities against the extension, all on
  # identical data/hyperparameters, so the three-way comparison is fair.
  nonlinearities = ['tanh', 'relu', 'leaky_relu']
  runs = {}
  for nonlinearity in nonlinearities:
    runs[nonlinearity] = up.compute_uncertainty_and_error(
        hparams, nonlinearity, train_image, train_label, test_image,
        test_label)
    predicted_mse, actual_mse = runs[nonlinearity]
    corr = np.corrcoef(predicted_mse, actual_mse)[0, 1]
    print('[%s] raw per-example corr(predicted variance, actual MSE) = %.4f'
          % (nonlinearity, corr))

  dataset_label = up._DATASET_LABELS.get(FLAGS.dataset, FLAGS.dataset.upper())
  title = '%s Tanh/ReLU/LeakyReLU-%s' % (
      dataset_label, up._size_label(FLAGS.num_train))
  up.make_figure3(runs, FLAGS.output_file, title)


if __name__ == '__main__':
  tf.app.run(main)
