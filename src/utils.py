import os
import csv
import argparse
import logging
import pickle
import yaml
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
ARGS_NAME  = 'args.yml'
MODEL_NAME = 'model.pkl'
BEST_MODEL = 'best_model.pkl'
STATS_CSV  = 'training_stats.csv'
CURRENT_BASELINE = [1.44576*10**-6, 0.04302]

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
  add_arg('--run', help='Experiment run number', default=0)
  add_arg('--eval_tpr',help='FPR at which TPR will be evaluated', default=0.000003)
  add_arg('--evaluate', help='Perform evaluation on test set only',action='store_true')

  # Training
  add_arg('--nb_epoch', help='Number of epochs to train', type=int, default=2)
  add_arg('--lrate', help='Initial learning rate', type=float, default = 0.005)
  add_arg('--batch_size', help='Size of each minibatch', type=int, default=4)

  # Dataset
  add_arg('--train_file', help='Path to train pickle file',type=str,default=None)
  add_arg('--val_file',   help='Path to val   pickle file',type=str,default=None)
  add_arg('--test_file',  help='Path to test  pickle file',type=str,default=None)
  add_arg('--nb_train', help='Number of training samples', type=int, default=10)
  add_arg('--nb_val', help='Number of validation samples', type=int, default=10)
  add_arg('--nb_test', help='Number of test samples', type=int, default=10)

  # Model Architecture
  add_arg('--nb_hidden', help='Number of hidden units per layer', type=int, default=32)
  add_arg('--nb_layer', help='Number of network grapn conv layers', type=int, default=6)


  return parser.parse_args()

def initialize_logger(experiment_dir):
  '''
  Logger prints to stdout and logfile
  '''
  logfile = os.path.join(experiment_dir, 'log.txt')
  logging.basicConfig(filename=logfile,format='%(message)s',level=logging.INFO)
  logging.getLogger().addHandler(logging.StreamHandler())

def get_experiment_dir(experiment_name, run_number):
  '''
  Saves all models within a 'models' directory where the experiment is run.
  Returns path to the specific experiment within the 'models' directory.
  '''
  current_dir = os.getcwd()
  save_dir = os.path.join(current_dir, 'models')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir) # Create models dir which will contain experiment data
  return os.path.join(save_dir, experiment_name, str(run_number))

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
  os.makedirs(experiment_dir)
  csv_path = os.path.join(experiment_dir, STATS_CSV)
  with open(csv_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'lrate', 'train_tpr', 'train_roc', 'train_loss', 'val_tpr', 'val_roc', 'val_loss', 'running_loss'])


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
  '''
  Load torch model.
  '''
  m = torch.load(model_file)
  return m

def load_best_model(experiment_dir):
  '''
  Load the model which performed best in training.
  '''
  best_model_path = os.path.join(experiment_dir, BEST_MODEL)
  return load_model(best_model_path)

def save_model(m, model_file):
  '''
  Save torch model.
  '''
  torch.save(m, model_file)

def save_best_model(experiment_dir, net):
  '''
  Called if current model performs best.
  '''
  model_path = os.path.join(experiment_dir, BEST_MODEL)
  save_model(net, model_path)
  logging.warning("Best model saved.")

def save_epoch_model(experiment_dir, net):
  '''
  Optionally called after each epoch to save current model.
  '''
  model_path = os.path.join(experiment_dir, MODEL_NAME)
  save_model(net, model_path)


def load_args(experiment_dir):
  '''
  Restore experiment arguments
  args contain e.g. nb_epochs_complete, lrate
  '''
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'r') as argfile:
    args = yaml.load(argfile)
  logging.warning("Model arguments restored.")
  return args

def save_args(experiment_dir, args):
  '''
  Save experiment arguments.
  args contain e.g. nb_epochs_complete, lrate
  '''
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'w') as argfile:
    yaml.dump(args, argfile, default_flow_style=False)

######################
#     EVALUATION     #
######################
def score_plot_preds(true_y, pred_y, weights, experiment_dir, plot_name, f=0.5):
  '''
  Compute and return weighted ROC AUC scores, TPR at given f (FPR).
  Plot ROC curves and save plots.
  '''
  roc_score = roc_auc_score(true_y, pred_y, sample_weight=weights)
  fprs, tprs, thresholds = roc_curve(true_y, pred_y, sample_weight=weights)
  # Compute TPR at specified f
  tpr = 0.0
  for i, fpr in enumerate(fprs):
    if fpr > f:
      '''
      try:
        logging.warning("FPR: {}, TPR: {}, Threshold: {}".format(
                          f, tpr, thresholds[i-1]))
      except:
        pass
      '''
      break
    tpr = tprs[i]

  # Plot ROC AUC curve
  plot_roc_curve(fprs, tprs,  experiment_dir, plot_name, [f, tpr])

  return tpr, roc_score

def plot_roc_curve(fprs, tprs, experiment_dir, plot_name, performance):
  '''
  Plot and save one ROC curve.
  '''
  # Plot
  plt.clf()
  plt.semilogx(fprs, tprs)
  # Zooms
  plt.xlim([10**-7,1.0])
  plt.ylim([0,1.0])
  # Style
  plt.xlabel("False Positive Rate (1- BG rejection)")
  plt.ylabel("True Positive Rate (Signal Efficiency)")
  plt.scatter(performance[0], performance[1], label='GNN')
  plt.scatter(CURRENT_BASELINE[0], CURRENT_BASELINE[1], label='Baseline')
  plt.legend()
  plt.grid(linestyle=':')
  #Save
  plotfile = os.path.join(experiment_dir, '{}.png'.format(plot_name))
  plt.savefig(plotfile)
  plt.clf()

def update_best_plots(experiment_dir):
  '''
  Rename .png plots to best when called.
  '''
  for f in os.listdir(experiment_dir):
    # Write over old best curves
    if f.endswith(".png") and not f.startswith("best"):
      old_name = os.path.join(experiment_dir, f)
      new_name = os.path.join(experiment_dir, "best_"+f)
      os.rename(old_name, new_name)
      
def track_epoch_stats(epoch, lrate, train_loss, train_stats, val_stats, experiment_dir):
  '''
  Write loss, fpr, roc_auc information to .csv file in model directory.
  '''
  csv_path = os.path.join(experiment_dir, STATS_CSV)
  with open(csv_path, 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow((epoch, lrate)+train_stats+val_stats+(train_loss,))

def save_preds(evt_id, f_name, pred_y, experiment_dir):
  '''
  Save predicted outputs for predicted event id, filename.
  '''
  pred_file = os.path.join(experiment_dir, 'preds.csv')
  with open(pred_file, 'x') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['event_id', 'filename', 'prediction'])
    for e, f, y in zip(evt_id, f_name, pred_y):
      writer.writerow((e, f, y))

def save_test_scores(nb_eval, epoch_loss, tpr, roc, experiment_dir):
  test_scores = {'nb_eval':nb_eval,
                 'epoch_loss':epoch_loss,
                 'tpr':float(tpr),
                 'roc auc':float(roc)}
  pred_file = os.path.join(experiment_dir, 'test_scores.yml')
  with open(pred_file, 'x') as f:
    yaml.dump(test_scores, f, default_flow_style=False)
