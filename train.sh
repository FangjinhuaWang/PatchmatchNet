#!/usr/bin/env bash

# train on DTU's training set
MVS_TRAINING="/home/mvs_training/dtu/"

python train.py --dataset dtu_yao --batch_size 4 --epochs 8 \
--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
--patchmatch_interval_scale 0.005 0.0125 0.025 \
--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --vallist lists/dtu/val.txt \
--logdir ./checkpoints $@
