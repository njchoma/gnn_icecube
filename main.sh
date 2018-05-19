#!/bin/bash

NAME='test_model'
TRAINFILE='/scratch/nc2201/data/icecube/smtrain.pickle'
VALFILE='/scratch/nc2201/data/icecube/smtrain.pickle'

OPTIONS="--save_every_epoch"

PYARGS="--name $NAME --train_file $TRAINFILE --val_file $VALFILE $OPTIONS"

echo -e "\nStarting experiment with name $NAME...\n"

python3 src/main.py $PYARGS
