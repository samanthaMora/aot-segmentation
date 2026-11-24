#!/bin/zsh
# Last-submission experiment (user allowed external wiki frames on 2026-11-21).
# A/B (honest): same recipe as ds_long (DinoSeg slim, 5000 its) + external frames from episodes NOT in the private exam.
# Compare with ds_long: EXAM 0.7717 / VAL 0.8777. Kit labels repeated x3 so the hand-reviewed frames keep weight.
cd "${0:A:h}/.."
V="vit_small_patch16_dinov3.lvd1689m|c24,32,48,32"
./work/run_exp.sh ds_ext_ab "DinoSeg slim 5000 its + external frames (non-exam episodes, kit x3), honest" "" --train data/train/images:work/labels_train_v3:3 data/extra/images:work/labels_extra_ab --arch dinoseg --h 448 --w 800 --crop-h 448 --crop-w 800 --zoom 1.0,2.0 --closeup 0.2 --lr 1e-3 --bs 8 --encoder "$V" --enc-mult 0.02 --iters 5000 --seed 0 --lazy
