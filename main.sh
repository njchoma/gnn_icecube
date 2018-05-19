#!/bin/bash

NAME='test_model'
TRAINFILE='/scratch/nc2201/data/icecube/smtrain.pickle'
VALFILE='/scratch/nc2201/data/icecube/smtrain.pickle'

PYARGS="--name $NAME --train_file $TRAINFILE --val_file $VALFILE"

echo -e "\nStarting experiment with name $NAME...\n"

python3 src/main.py $PYARGS
