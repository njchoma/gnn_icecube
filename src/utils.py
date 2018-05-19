import os
import csv
import argparse
import logging
import pickle

import torch

import model

#####################
#     CONSTANTS     #
#####################
ARGS_NAME  = 'args.pkl'
MODEL_NAME = 'model.pkl'

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
  add_arg('--train_file', help='Path to train pickle file',type=str,default=None)
  add_arg('--val_file',   help='Path to val   pickle file',type=str,default=None)
  add_arg('--test_file',  help='Path to test  pickle file',type=str,default=None)
  add_arg('--nb_train', help='Number of training samples', type=int, default=10)
  add_arg('--nb_val', help='Number of validation samples', type=int, default=10)
  add_arg('--nb_test', help='Number of test samples', type=int, default=10)
  add_arg('--evaluate', help='Perform evaluation on test set only',action='store_true')

  # Model
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
  csv_path = os.path.join(experiment_dir, 'training_stats.csv')
  with open(csv_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'lrate', 'running_loss'])


##########################
#     DATA UTILITIES     #
##########################
def load_dataset(datafile, nb_ex):
  with open(datafile, 'rb') as filein:
    X, y, weights, event_id, filenames = pickle.load(filein)
  w = [] # Need to convert weights to float
  for weight in weights[:nb_ex]:
    w.append(float(weight))
  return X[:nb_ex], y[:nb_ex], w, event_id[:nb_ex], filenames[:nb_ex]

###########################
#     MODEL UTILITIES     #
###########################
def create_or_restore_model(
                            experiment_dir,
                            input_dim,
                            nb_hidden,
                            nb_layer
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
    m = model.GNN(input_dim, nb_hidden, nb_layer)
    logging.info(m)
    save_model(m, model_file)
    logging.warning("Initial model saved.")
  return m

def load_model(model_file):
  m = torch.load(model_file)
  return m

def save_model(m, model_file):
  torch.save(m, model_file)

def load_args(experiment_dir):
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'rb') as argfile:
    args = pickle.load(argfile)
  logging.warning("Model arguments restored.")
  return args

def save_args(experiment_dir, args):
  args_path = os.path.join(experiment_dir, ARGS_NAME)
  with open(args_path, 'wb') as argfile:
    pickle.dump(args, argfile)
  logging.warning("Experiment arguments saved")
