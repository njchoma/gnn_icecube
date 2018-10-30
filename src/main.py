import os
import time
import logging
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

import utils
import model

#####################
#     CONSTANTS     #
#####################
TEST_NAME='Test'

#######################
#     EXPERIMENT      #
#######################

def train_one_epoch(net, criterion, optimizer, args, experiment_dir,
                    train_X,
                    train_y,
                    train_w):
  '''
  Train network for one epoch over the training set
  '''
  net.train()
  nb_train = len(train_X)
  batches = utils.get_batches(nb_train, args.batch_size)
  nb_batches = len(batches)
  epoch_loss = 0
  pred_y = np.zeros((nb_train))
  true_y = np.zeros((nb_train))
  weights = np.zeros((nb_train))
  for i, batch in enumerate(batches):
    optimizer.zero_grad()
    X, y, w, adj_mask, batch_nb_nodes = utils.batch_sample(
                                              train_X[batch],
                                              train_y[batch],
                                              train_w[batch]
                                              )
    out = net(X, adj_mask, batch_nb_nodes)
    loss = criterion(out, y, w)
    loss.backward()
    optimizer.step()

    beg =     i * args.batch_size
    end = (i+1) * args.batch_size
    pred_y[beg:end]  = out.data.cpu().numpy()
    true_y[beg:end]  = train_y[batch]
    weights[beg:end] = train_w[batch]

    epoch_loss += loss.item() 
    # Print running loss about 10 times during each epoch
    if (((i+1) % (nb_batches//10)) == 0):
      nb_proc = (i+1)*args.batch_size
      logging.info("  {:5d}: {:.9f}".format(nb_proc, epoch_loss/nb_proc))

  tpr, roc = utils.score_plot_preds(true_y, pred_y, weights,
                                      experiment_dir, 'train', args.eval_tpr)
  epoch_loss /= nb_batches * args.batch_size
  return (tpr, roc, epoch_loss)


def train(
          net,
          criterion,
          args, 
          experiment_dir, 
          train_X, train_y, train_w, 
          val_X, val_y, val_w
          ):
  optimizer = torch.optim.Adamax(net.parameters(), lr=args.lrate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
  # Nb epochs completed tracked in case training interrupted
  for i in range(args.nb_epochs_complete, args.nb_epoch):
    # Update learning rate in optimizer
    t0 = time.time()
    logging.info("\nEpoch {}".format(i+1))
    logging.info("Learning rate: {0:.3g}".format(args.lrate))
    
    train_stats = train_one_epoch(net,
                                  criterion,
                                  optimizer,
                                  args,
                                  experiment_dir,
                                  train_X, train_y, train_w)
    val_stats = evaluate(net, criterion, experiment_dir, args,
                            val_X,val_y,val_w, 'Valid')
                                
    utils.track_epoch_stats(i, args.lrate, 0, train_stats, val_stats, experiment_dir)

    # Update learning rate, remaining nb epochs to train
    scheduler.step(val_stats[0])
    args.lrate = optimizer.param_groups[0]['lr']
    args.nb_epochs_complete += 1

    # Track best model performance
    if (val_stats[0] > args.best_tpr):
      logging.warning("Best performance on valid set.")
      args.best_tpr = float(val_stats[0])
      utils.update_best_plots(experiment_dir)
      utils.save_best_model(experiment_dir, net)

    utils.save_epoch_model(experiment_dir, net)
    utils.save_args(experiment_dir, args)
    logging.info("Epoch took {} seconds.".format(int(time.time()-t0)))

  logging.warning("Training completed.")


def evaluate(net,
             criterion,
             experiment_dir,
             args,
             in_X, in_y, in_w,
             plot_name,
             in_e=None,
             in_f=None):
    net.eval()
    epoch_loss = 0
    # Get minibatches
    nb_samples = len(in_X)
    batches = utils.get_batches(nb_samples, args.batch_size)
    nb_batches = len(batches)
    nb_eval = nb_batches * args.batch_size
    # Track samples by batches for scoring
    pred_y = np.zeros((nb_eval))
    true_y = np.zeros((nb_eval))
    weights = np.zeros((nb_eval))
    evt_id = []
    f_name = []
    # Get predictions and loss over batches
    logging.info("Evaluating {} {} samples.".format(nb_eval,plot_name))
    with torch.autograd.no_grad():
        for i, batch in enumerate(batches):
            # Wrap samples in torch Variables
            X, y, w, adj_mask, batch_nb_nodes = utils.batch_sample(
                                                      in_X[batch],
                                                      in_y[batch],
                                                      in_w[batch]
                                                      )
            # Make predictions over batch
            out = net(X, adj_mask, batch_nb_nodes)
            loss = criterion(out, y, w)
            epoch_loss += loss.item() 
            # Track predictions, truth, weights over batches
            beg =     i * args.batch_size
            end = (i+1) * args.batch_size
            pred_y[beg:end] = out.data.cpu().numpy()
            true_y[beg:end] = in_y[batch]
            weights[beg:end] = in_w[batch]
            if in_e is not None:
                evt_id.extend(in_e[batch])
                f_name.extend(in_f[batch])
            # Print running loss 2 times 
            if (((i+1) % (nb_batches//2)) == 0):
                nb_proc = (i+1)*args.batch_size
                logging.info("  {:5d}: {:.9f}".format(nb_proc,
                                                      epoch_loss/nb_proc))

    # Score predictions, save plots, and log performance
    epoch_loss /= nb_eval # Normalize loss
    tpr, roc = utils.score_plot_preds(true_y, pred_y, weights,
                                      experiment_dir, plot_name, args.eval_tpr)
    logging.info("{}: loss {:>.3E} -- AUC {:>.3E} -- TPR {:>.3e}".format(
                                      plot_name, epoch_loss, roc, tpr))
    if in_e is not None:
        utils.save_preds(evt_id, f_name, pred_y, experiment_dir)
    # Save test numbers if evaluating on test set
    if plot_name == TEST_NAME:
        utils.save_test_scores(nb_eval, epoch_loss, tpr, roc, experiment_dir)
    return (tpr, roc, epoch_loss)


def main():
  input_dim=7
  spatial_dims=[0,1,2]
  args = utils.read_args()

  experiment_dir = utils.get_experiment_dir(args.name, args.run)
  utils.initialize_experiment_if_needed(experiment_dir, args.evaluate)
  # Logger will print to stdout and logfile
  utils.initialize_logger(experiment_dir)

  # Optionally restore arguments from previous training
  # Useful if training is interrupted
  if not args.evaluate:
    try:
      args = utils.load_args(experiment_dir)
    except:
      args.best_tpr = 0.0
      args.nb_epochs_complete = 0 # Track in case training interrupted
      utils.save_args(experiment_dir, args) # Save initial args

  net = utils.create_or_restore_model(
                                    experiment_dir, 
                                    args.nb_hidden, 
                                    args.nb_layer,
                                    input_dim,
                                    spatial_dims
                                    )
  if torch.cuda.is_available():
    net = net.cuda()
    logging.warning("Training on GPU")
  criterion = nn.functional.binary_cross_entropy
  if not args.evaluate:
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

  # Perform evaluation over test set
  try:
    net = utils.load_best_model(experiment_dir)
    logging.warning("\nBest model loaded for evaluation on test set.")
  except:
    logging.warning("\nCould not load best model for test set. Using current.")
  assert (args.test_file != None)
  test_X, test_y, test_w, test_e, test_f= utils.load_dataset(args.test_file,args.nb_test)
  test_stats = evaluate(net, criterion, experiment_dir, args,
                            test_X, test_y, test_w, TEST_NAME, test_e, test_f)

if __name__ == "__main__":
  main()
