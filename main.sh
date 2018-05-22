#!/bin/bash

#SBATCH --job-name=GCNN
#SBATCH --output=slurm_out/GCNN_%A_%a.out
# SBATCH --error=GPUTFtest.err
#SBATCH --time=2-00:00:00
#SBATCH --gres gpu:1
# SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=10000
# SBATCH --mail-type=FAIL # notifications for job done & fail
#SBATCH --mail-user=nc2201@courant.nyu.edu


# Dataset
TRAINFILE='/home/nc2201/data/icecube/orig/train.pickle'
VALFILE='/home/nc2201/data/icecube/orig/val.pickle'
TESTFILE='/home/nc2201/data/icecube/orig/test.pickle'

NB_TRAIN=70000
NB_VAL=40000
NB_TEST=40000

# Experiment
NAME="smtest_""$SLURM_ARRAY_TASK_ID"
NB_EPOCH=100
BATCH_SIZE=5

# Network hyperparameters
NB_LAYER=6
NB_HIDDEN=64

OPTIONS="--save_best --save_every_epoch"

PYARGS="--name $NAME --train_file $TRAINFILE --val_file $VALFILE --test_file $TESTFILE $OPTIONS --nb_train $NB_TRAIN --nb_val $NB_VAL --nb_test $NB_TEST --batch_size $BATCH_SIZE --nb_epoch $NB_EPOCH --nb_layer $NB_LAYER --nb_hidden $NB_HIDDEN"

echo -e "\nStarting experiment with name $NAME...\n"

module load pytorch/python3.6/0.3.0_4
source ~/pyenv/py3.6.3/bin/activate
python3 src/main.py $PYARGS
