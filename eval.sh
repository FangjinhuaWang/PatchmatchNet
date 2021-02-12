#!/usr/bin/env bash

# test on DTU's evaluation set
DTU_TESTING="/home/dtu/"
CHECKPOINT_FILE="./checkpoints/model_000007.ckpt"
python eval.py --dataset=dtu_yao_eval --batch_size=1 --n_views 5 \
--patch_match_iteration 1 2 2 --patch_match_range 6 4 2 \
--patch_match_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
--patch_match_interval_scale 0.005 0.0125 0.025 \
--test_path=$DTU_TESTING --geom_pixel_threshold=1 --geom_depth_threshold=0.01 --photo_thresholdhold 0.8 \
--out_dir=./outputs --testlist lists/dtu/test.txt --checkpoint_path $CHECKPOINT_FILE $@

# -------------------------------------------------------------------------------------
# test on eth3d benchmark
ETH3d_TESTING="/home/eth3d_high_res_test/"
# python eval_eth.py --dataset=eth3d --split train --batch_size=1 --n_views 7 \
# --patch_match_iteration 1 2 2 --patch_match_range 6 4 2 \
# --patch_match_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
# --patch_match_interval_scale 0.005 0.0125 0.025 \
# --test_path=$ETH3d_TESTING --geom_pixel_threshold=1 --geom_depth_threshold=0.01 --photo_threshold=0.6 \
# --out_dir ./outputs_eth --checkpoint_path $CHECKPOINT_FILE $@

# -------------------------------------------------------------------------------------
# test on tanks & temples
TANK_TESTING="/home/TankandTemples/"
# python eval_tank.py --dataset=tanks --split intermediate --batch_size=1 --n_views 7 \
# --patch_match_iteration 1 2 2 --patch_match_range 6 4 2 \
# --patch_match_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
# --patch_match_interval_scale 0.005 0.0125 0.025 \
# --test_path=$TANK_TESTING --geom_pixel_threshold=1 --geom_depth_threshold=0.01 \
# --out_dir ./outputs_tanks --checkpoint_path $CHECKPOINT_FILE $@

# -------------------------------------------------------------------------------------
# test on your custom dataset
CUSTOM_TESTING="/home/custom/"
# python eval_custom.py --dataset=custom --batch_size=1 --n_views 5 \
# --patch_match_iteration 1 2 2 --patch_match_range 6 4 2 \
# --patch_match_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
# --patch_match_interval_scale 0.005 0.0125 0.025 \
# --test_path=$CUSTOM_TESTING --geom_pixel_threshold=1 --geom_depth_threshold=0.01 --photo_threshold 0.8 \
# --out_dir ./outputs_custom --checkpoint_path $CHECKPOINT_FILE $@