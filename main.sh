#!/bin/bash

#SBATCH --job-name=IceCube_GNN
#SBATCH --output=slurm_out/GCNN_%A_%a.out
#SBATCH --time=2-00:00:00
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --mem=14000
# SBATCH --mail-type=FAIL # notifications for job fail
#SBATCH --mail-user=nc2201@courant.nyu.edu

mkdir -p slurm_out

# Dataset
TRAINFILE='/misc/vlgscratch4/BrunaGroup/choma/icecube/train.pickle'
VALFILE='/misc/vlgscratch4/BrunaGroup/choma/icecube/val.pickle'
TESTFILE='/misc/vlgscratch4/BrunaGroup/choma/icecube/test.pickle'

NB_TRAIN=500000
NB_VAL=500000
NB_TEST=500000

# Experiment
NAME="aa_test_updates"
RUN="$SLURM_ARRAY_TASK_ID"
NB_EPOCH=500
BATCH_SIZE=5

# Network hyperparameters
NB_LAYER=6
NB_HIDDEN=64

OPTIONS=""

PYARGS="--name $NAME --run $RUN --train_file $TRAINFILE --val_file $VALFILE --test_file $TESTFILE $OPTIONS --nb_train $NB_TRAIN --nb_val $NB_VAL --nb_test $NB_TEST --batch_size $BATCH_SIZE --nb_epoch $NB_EPOCH --nb_layer $NB_LAYER --nb_hidden $NB_HIDDEN"

echo -e "\nStarting experiment with name $NAME...\n"

source ~/pyenv/torch4/bin/activate
python src/main.py $PYARGS
