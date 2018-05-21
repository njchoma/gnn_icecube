#!/bin/bash

# Dataset
TRAINFILE='/scratch/nc2201/data/icecube/smtrain.pickle'
VALFILE='/scratch/nc2201/data/icecube/smtrain.pickle'
TESTFILE='/scratch/nc2201/data/icecube/smtrain.pickle'

NB_TRAIN=70
NB_VAL=70
NB_TEST=70

# Experiment
NAME='test_model'
NB_EPOCH=20
BATCH_SIZE=7

# Network hyperparameters
NB_LAYER=6
NB_HIDDEN=32

OPTIONS="--save_every_epoch --evaluate"

PYARGS="--name $NAME --train_file $TRAINFILE --val_file $VALFILE --test_file $TESTFILE $OPTIONS --nb_train $NB_TRAIN --nb_val $NB_VAL --nb_test $NB_TEST --batch_size $BATCH_SIZE --nb_epoch $NB_EPOCH --nb_layer $NB_LAYER --nb_hidden $NB_HIDDEN"

echo -e "\nStarting experiment with name $NAME...\n"

python3 src/main.py $PYARGS
