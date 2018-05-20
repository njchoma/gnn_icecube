#!/bin/bash

NAME='test_model'
TRAINFILE='/scratch/nc2201/data/icecube/smtrain.pickle'
VALFILE='/scratch/nc2201/data/icecube/smtrain.pickle'

NB_TRAIN=50
NB_VAL=50
BATCH_SIZE=5

OPTIONS="--save_every_epoch"

PYARGS="--name $NAME --train_file $TRAINFILE --val_file $VALFILE $OPTIONS --nb_train $NB_TRAIN --nb_val $NB_VAL --batch_size $BATCH_SIZE"

echo -e "\nStarting experiment with name $NAME...\n"

python3 src/main.py $PYARGS
