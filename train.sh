#!/usr/bin/env bash

MVS_TRAINING="/home/mvs_training/dtu/"

# Train on converted DTU training set
python train.py --batch_size 4 --epochs 8 --num_light_idx 7 --input_folder=$MVS_TRAINING --output_folder=$MVS_TRAINING \
--train_list=lists/dtu/train.txt --test_list=lists/dtu/val.txt "$@"

# Legacy train on DTU's training set
#python train_dtu.py --batch_size 4 --epochs 8 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt \
#--vallist lists/dtu/val.txt --logdir ./checkpoints "$@"
