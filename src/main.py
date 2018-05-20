import os
import time
import logging

import torch
import torch.nn as nn

import utils
import model

def train_one_epoch(net, criterion, optimizer, args, train_X, train_y, train_w):
  '''
  Train network for one epoch over the training set
  '''
  nb_train = len(train_X)
  batches = utils.get_batches(nb_train, args.batch_size)
  nb_batches = len(batches)
  epoch_loss = 0
  for i, batch in enumerate(batches):
    optimizer.zero_grad()
    X, y, w, adj_mask, batch_nb_nodes = utils.batch_sample(
                                              train_X[batch],
                                              train_y[batch],
                                              train_w[batch]
                                              )
    out = net(X, adj_mask, batch_nb_nodes)
    loss = criterion(out, y, w)
    optimizer.step()
    epoch_loss += loss.data[0] 
    # Print running loss about 10 times during each epoch
    if (((i+1) % (nb_batches//10)) == 0):
      nb_proc = (i+1)*args.batch_size
      logging.info("  {:5d}: {:f}".format(nb_proc, epoch_loss/nb_proc))
    
  return epoch_loss / (nb_batches * args.batch_size)


def train(
          net,
          criterion,
          args, 
          experiment_dir, 
          train_X, train_y, train_w, 
          val_X, val_y, val_w
          ):
  '''
  Train network over all epochs.
  Optionally save model after every epoch.
  Optionally track best model.
  '''
  # Nb epochs completed tracked in case training interrupted
  for i in range(args.nb_epochs_complete, args.nb_epoch):
    # Update learning rate in optimizer
    optimizer = torch.optim.Adamax(net.parameters(), lr=args.lrate)
    t0 = time.time()
    logging.info("\nEpoch {}".format(i+1))
    logging.info("Learning rate: {0:.3g}".format(args.lrate))
    # Train for one epoch
    train_loss = train_one_epoch(net, criterion, optimizer, args, 
                                 train_X, train_y, train_w)
    # Update learning rate, remaining epochs to train
    args.lrate *= args.lrate_decay
    args.nb_epochs_complete += 1
    # Optionally save after each epoch
    if (args.save_every_epoch):
      utils.save_epoch_model(experiment_dir, net)
      utils.save_args(experiment_dir, args)

  logging.warning("Training completed.")



def main():
  input_dim=6
  spat_dims=[0,1,2]
  args = utils.read_args()

  # Get path to experiment directory
  experiment_dir = utils.get_experiment_dir(args.name)
  # Create experiment directory and csv file if not found
  utils.initialize_experiment_if_needed(experiment_dir, args.evaluate)
  # Logger will print to stdout and logfile
  utils.initialize_logger(experiment_dir)

  # Optionally restore arguments from previous training
  # Useful if training is interrupted
  if not args.evaluate:
    try:
      args = utils.load_args(experiment_dir)
    except:
      args.nb_epochs_complete = 0 # Track in case training interrupted
      utils.save_args(experiment_dir, args) # Save initial args

  # Create model if first time running
  # Restore model if continuing previous training or evaluating
  net = utils.create_or_restore_model(
                                    experiment_dir, 
                                    args.nb_hidden, 
                                    args.nb_layer,
                                    input_dim,
                                    spat_dims
                                    )
  criterion = nn.functional.binary_cross_entropy
  if not args.evaluate:
    # Before loading, ensure train, val file arguments not None
    assert (args.train_file != None)
    assert (args.val_file   != None)
    train_X,train_y,train_w,_,_=utils.load_dataset(args.train_file,args.nb_train)
    val_X,  val_y,  val_w, _,_ =utils.load_dataset(args.val_file,  args.nb_val)
    logging.info("Training on {} samples.".format(len(train_X)))
    logging.info("Validate on {} samples.".format(len(val_X)))
    train(
          net,
          criterion,
          args,
          experiment_dir,
          train_X, train_y, train_w,
          val_X, val_y, val_w
          )
    

if __name__ == "__main__":
  main()
