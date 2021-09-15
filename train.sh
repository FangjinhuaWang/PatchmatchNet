#!/usr/bin/env bash

# train on DTU's training set
MVS_TRAINING="/home/mvs_training/dtu/"
python train_dtu.py --batch_size 4 --epochs 8 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt \
--vallist lists/dtu/val.txt --logdir ./checkpoints "$@"
