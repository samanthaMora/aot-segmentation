#!/bin/zsh
cd "${0:A:h}/.."
C="--train data/train/images:work/labels_train_v3 --arch dinoseg --h 448 --w 800 --crop-h 448 --crop-w 800 --zoom 1.0,2.0 --closeup 0.2 --lr 1e-3 --iters 3000 --bs 8 --seed 0"
V=vit_small_patch16_dinov3.lvd1689m
./work/run_exp.sh ds_slim_ft02 "DinoSeg slim decoder (c24,32,48,32), backbone lr x0.02, v3, honest" "" ${=C} --encoder "$V|c24,32,48,32" --enc-mult 0.02
./work/run_exp.sh ds_ft05 "DinoSeg, backbone lr x0.05, v3, honest" "" ${=C} --encoder "$V" --enc-mult 0.05
./work/run_exp.sh ds_ft10 "DinoSeg, backbone lr x0.1, v3, honest" "" ${=C} --encoder "$V" --enc-mult 0.1
