#!/bin/zsh
# Final candidate with the one change that clearly won on val (5000 its: +2.1, 7/0 frames): kit train v3 + kit val x4, nothing else.
cd "${0:A:h}/.."
while pgrep -f "work/queue_ds4.sh" >/dev/null; do /bin/sleep 30; done
V="vit_small_patch16_dinov3.lvd1689m|c24,32,48,32"
./work/run_exp.sh ds_final_b "FINAL candidate: DinoSeg slim, 5000 its, v3 + val x4 (VAL not honest)" "" --train data/train/images:work/labels_train_v3 data/val_in_domain/images:data/val_in_domain/masks:4 --val none --arch dinoseg --h 448 --w 800 --crop-h 448 --crop-w 800 --zoom 1.0,2.0 --closeup 0.2 --lr 1e-3 --bs 8 --encoder "$V" --enc-mult 0.02 --iters 5000 --seed 0
