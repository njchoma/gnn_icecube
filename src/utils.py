import os
import csv
import argparse
import logging
import pickle
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib; matplotlib.use('Agg') # no display on clusters
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import model

#####################
#     CONSTANTS     #
#####################
ARGS_NAME  = 'args.pkl'
MODEL_NAME = 'model.pkl'
BEST_MODEL = 'best_model.pkl'
STATS_CSV  = 'training_stats.csv'
NB_ZERO_NODES = 30 # Drastically improves performance
PLOT_ZOOMS=[10**-k for k in [0,2,4,5]]

#####################################
#     EXPERIMENT INITIALIZATION     #
#####################################

def read_args():
  '''
  Parse stdin arguments
  '''

  parser = argparse.ArgumentParser(description=
                      'Arguments for GNN model and experiment')
  add_arg = parser.add_argument

  # Experiment
  add_arg('--name', help='Experiment reference name', required=True)
  add_arg('--eval_tpr',help='TPR at which FPR will be evaluated', default=0.2)
  add_arg('--evaluate', help='Perform evaluation on test set only',action='store_true')
  add_arg('--save_best', help='Save best model', action='store_true')
  add_arg('--save_every_epoch', help='Save model after every epoch. Good if training expected to be interrupted', action='store_true')

  # Training
  add_arg('--nb_epoch', help='Number of epochs to train', type=int, default=2)
  add_arg('--lrate', help='Initial learning rate', type=float, default = 0.005)
  add_arg('--lrate_decay',help='Exponential decay factor',type=float,default=0.96)
  add_arg('--batch_size', help='Size of each minibatch', type=int, default=4)

  # Dataset
  add_arg('--train_file', help='Path to train pickle file',type=str,default=None)
  add_arg('--val_file',   help='Path to val   pickle file',type=str,default=None)
  add_arg('--test_file',  help='Path to test  pickle file',type=str,default=None)
  add_arg('--nb_train', help='Number of training samples', type=int, default=10)
  add_arg('--nb_val', help='Number of validation samples', type=int, default=10)
  add_arg('--nb_test', help='Number of test samples', type=int, default=10)

  # Model Architecture
  add_arg('--nb_hidden', help='Number of hidden units per layer', default=32)
  add_arg('--nb_layer', help='Number of network grapn conv layers', default=6)


  return parser.parse_args()

def initialize_logger(experiment_dir):
  '''
  Logger prints to stdout and logfile
  '''
  logfile = os.path.join(experiment_dir, 'log.txt')
  logging.basicConfig(filename=logfile,format='%(message)s',level=logging.INFO)
  logging.getLogger().addHandler(logging.StreamHandler())

def get_experiment_dir(experiment_name):
  '''
  Saves all models within a 'models' directory where the experiment is run.
  Returns path to the specific experiment within the 'models' directory.
  '''
  current_dir = os.getcwd()
  save_dir = os.path.join(current_dir, 'models')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir) # Create models dir which will contain experiment data
  return os.path.join(save_dir, experiment_name)

def initialize_experiment_if_needed(model_dir, evaluate_only):
  '''
  Check if experiment initialized and initialize if not.
  Perform evaluate safety check.
  '''
  if not os.path.exists(model_dir):
    initialize_experiment(model_dir)
    if evaluate_only:
      logging.warning("EVALUATING ON UNTRAINED NETWORK")

def initialize_experiment(experiment_dir):
  '''
  Create experiment directory and initiate csv where epoch info will be stored.
  '''
  print("Initializing experiment.")
  os.mkdir(experiment_dir)
  csv_path = os.path.join(experiment_dir, STATS_CSV)
  with open(csv_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'lrate', 'train_fpr', 'train_roc', 'train_loss', 'val_fpr', 'val_roc', 'val_loss', 'running_loss'])


#############################
#     DATASET UTILITIES     #
#############################
def load_dataset(datafile, nb_ex):
  with open(datafile, 'rb') as filein:
    X, y, weights, event_id, filenames = pickle.load(filein)
  return X[:nb_ex], y[:nb_ex], weights[:nb_ex],event_id[:nb_ex],filenames[:nb_ex]

####################
#     BATCHING     #
####################

def get_batches(nb_samples, batch_size):
  '''
  Randomly permute sample indices, then divide into minibatches.
  '''
  # Create batch indices
  samples = np.arange(nb_samples)
  # Shuffle batch indices
  samples = np.random.permutation(samples)
  # Ensure all minibatches are same size
  if (nb_samples % batch_size) != 0:
    samples = samples[:-(nb_samples % batch_size)]
  # Reshape to shape (nb_batches, batch_size)
  batches = samples.reshape(-1, batch_size)
  return batches

def batch_sample(batch_X, batch_y, batch_w):
  '''
  Pad X to uniform size, then
  wrap batch in torch Tensors (cuda if available)
  '''
  padded_X, adj_mask, batch_nb_nodes, = pad_batch(batch_X)
  if torch.cuda.is_available():
    wrap = torch.FloatTensor
  else:
    wrap = torch.FloatTensor
  
  X = Variable(wrap(padded_X))
  y = Variable(wrap(batch_y))
  w = Variable(wrap(batch_w))
  adj_mask = Variable(wrap(adj_mask))
  batch_nb_nodes = Variable(wrap(batch_nb_nodes))
  return X, y, w, adj_mask, batch_nb_nodes

def pad_batch(X):
  '''
  Minibatches must be uniform size in order to be passed through the GNN.
  First a uniform zero-padding is applied to all samples.
  This is for performance only (an oddity, to be resolved).
  Next all samples are padded variably to bring the every sample up to a 
  uniform size.
  A mask for the adj matrix is returned,
  and the true (plus uniform padding) sizes of each sample are also returned.
  '''
  nb_samples = len(X)
  nb_features = X[0].shape[1]
  largest_size = 0
  batch_nb_nodes = np.zeros(nb_samples, dtype=int)
  # First add zero nodes to samples and find largest sample
  # Largest is the sample with the most points in point cloud
  for i in range(nb_samples):
    zeros = np.zeros((NB_ZERO_NODES, nb_features))
    X[i] = np.concatenate((zeros, X[i]),axis=0)
    batch_nb_nodes[i] = X[i].shape[0]
    largest_size = max(largest_size, X[i].shape[0])

  adj_mask = np.zeros(shape=(nb_samples, largest_size, largest_size))
  # Append zero nodes to features with fewer points in point cloud
  #   than largest_size
  for i in range(nb_samples):
    zeros = np.zeros((largest_size-X[i].shape[0], nb_features))
    X[i] = np.concatenate((X[i], zeros), axis=0)
    adj_mask[i, :batch_nb_nodes[i], :batch_nb_nodes[i]] = 1

  X = np.stack(X, axis=0)
  return X, adj_mask, batch_nb_nodes

###########################
#     MODEL UTILITIES     #
###########################
def create_or_restore_model(
                            experiment_dir,
                            nb_hidden,
                            nb_layer,
                            input_dim,
                            spat_dims
                            ):
  '''
  Checks if model exists and creates it if not.
  Returns model.
  '''
  model_file = os.path.join(experiment_dir, MODEL_NAME)
  if os.path.exists(model_file):
    logging.warning("Loading model...")
    m = load_model(model_file)
    logging.warning("Model restored.")
  else:
    logging.warning("Creating new model:")
    m = model.GNN(nb_hidden, nb_layer, input_dim, spat_dims)
    logging.info(m)
    save_model(m, model_file)
    logging.warning("Initial model saved.")
  return m

def load_model(model_file):
  m = torch.load(model_file)
  return m

def save_model(m, model_file):
  torch.save(m, model_file)
  logging.warning("Model saved.")

def save_best_model(experiment_dir, net):
  model_path = os.path.join(experiment_dir, BEST_MODEL)
  save_model(net, model_path)

def save_epoch_model(experiment_dir, net):
  model_path = os.path.join(experiment_dir, MODEL_NAME)
  save_model(net, model_path)


def load_args(experiment_dir):
  '''
  Restore experiment arguments
  args contain e.g. nb_epochs_complete, lrate
  '''
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'rb') as argfile:
    args = pickle.load(argfile)
  logging.warning("Model arguments restored.")
  return args

def save_args(experiment_dir, args):
  '''
  Save experiment arguments.
  args contain e.g. nb_epochs_complete, lrate
  '''
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'wb') as argfile:
    pickle.dump(args, argfile)
  logging.warning("Experiment arguments saved")

######################
#     EVALUATION     #
######################
def score_plot_preds(true_y, pred_y, weights, experiment_dir, plot_name, t=0.5):
  '''
  Compute and return weighted ROC AUC scores, FPR at given t (TPR).
  Plot ROC AUC curves and save plots.
  '''
  roc_score = roc_auc_score(true_y, pred_y, sample_weight=weights)
  fprs, tprs, thresholds = roc_curve(true_y, pred_y, sample_weight=weights)
  # Compute FPR at specified t
  fpr = 1.0
  for i, tpr in enumerate(tprs):
    if tpr > t:
      fpr = fprs[i]
      break

  # Plot ROC AUC curves
  for i, zoom in enumerate(PLOT_ZOOMS):
    plot_roc_curve(fprs, tprs, zoom, experiment_dir, plot_name, [fpr, t])

  return fpr, roc_score

def plot_roc_curve(fprs, tprs, zoom, experiment_dir, plot_name, performance):
  '''
  Plots and saves one ROC curve at specified zoom level.
  '''
  # Plot
  plt.clf()
  plt.plot(fprs, tprs)
  # Zooms
  plt.xlim([0,zoom])
  plt.ylim([0,zoom])
  # Style
  plt.xlabel("False Positive Rate (1- BG rejection)")
  plt.ylabel("True Positive Rate (Signal Efficiency)")
  plt.grid(linestyle=':')
  #Save
  plotfile = os.path.join(experiment_dir, '{}_{}.png'.format(plot_name, zoom))
  plt.savefig(plotfile)
  plt.clf()

def update_best_plots(experiment_dir):
  '''
  Rename .png plots to best
  '''
  for f in os.listdir(experiment_dir):
    # Write over old best curves
    if f.endswith(".png") and not f.startswith("best"):
      old_name = os.path.join(experiment_dir, f)
      new_name = os.path.join(experiment_dir, "best_"+f)
      os.rename(old_name, new_name)
      
def track_epoch_stats(epoch, lrate, train_loss, train_stats, val_stats, experiment_dir):
  csv_path = os.path.join(experiment_dir, STATS_CSV)
  with open(csv_path, 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow((epoch, lrate)+train_stats+val_stats+(train_loss,))
