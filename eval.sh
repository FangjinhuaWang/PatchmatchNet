#!/usr/bin/env bash

# test on DTU's evaluation set
DTU_TESTING="/home/dtu/"
CHECKPOINT_FILE="./checkpoints/patchmatchnet-params.pt"
python eval.py --num_views 5 --scan_list lists/dtu/test.txt --input_folder=$DTU_TESTING --output_folder=$DTU_TESTING \
--checkpoint_path $CHECKPOINT_FILE --photo_thresholdhold 0.8 $@

# -------------------------------------------------------------------------------------
# test on eth3d benchmark
ETH3d_TESTING="/home/eth3d_high_res_test/"
# python eval_eth.py --num_views 7 --eval_type eth3d_train --input_folder=$ETH3d_TESTING --output_folder=$ETH3d_TESTING \
# --checkpoint_path $CHECKPOINT_FILE --photo_threshold=0.6 $@

# -------------------------------------------------------------------------------------
# test on tanks & temples
TANK_TESTING="/home/TankandTemples/"
# python eval_tank.py --num_views 7 --eval_type tanks_intermediate --input_folder=$TANK_TESTING --output_folder=$TANK_TESTING \
# --checkpoint_path $CHECKPOINT_FILE $@

# -------------------------------------------------------------------------------------
# test on your custom dataset
CUSTOM_TESTING="/home/custom/"
# python eval_custom.py --num_views 5 --input_folder=$CUSTOM_TESTING --output_folder=$CUSTOM_TESTING \
# --checkpoint_path $CHECKPOINT_FILE --photo_threshold 0.8 $@