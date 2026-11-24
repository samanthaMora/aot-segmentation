#!/bin/zsh
# After ds_long (5000 its) gained +2.1 on val (7/0 frames) with the exam unchanged: does 8000 keep helping? Honest.
cd "${0:A:h}/.."
while pgrep -f "work/queue_ds3.sh" >/dev/null; do /bin/sleep 30; done
V="vit_small_patch16_dinov3.lvd1689m|c24,32,48,32"
./work/run_exp.sh ds_8k "DinoSeg slim, 8000 its, honest" "" --train data/train/images:work/labels_train_v3 --arch dinoseg --h 448 --w 800 --crop-h 448 --crop-w 800 --zoom 1.0,2.0 --closeup 0.2 --lr 1e-3 --bs 8 --encoder "$V" --enc-mult 0.02 --iters 8000 --seed 0
