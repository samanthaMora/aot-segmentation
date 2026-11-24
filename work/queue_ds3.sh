#!/bin/zsh
# Higher input resolution for DinoSeg (ViT at 240x432 = 15x27 tokens), honest. Server latency calibration: 392.4 ms for 200 ms M4 1-thread.
cd "${0:A:h}/.."
while pgrep -f "work/queue_ds2.sh|work/queue_final_ds.sh" >/dev/null; do /bin/sleep 30; done
V="vit_small_patch16_dinov3.lvd1689m|c24,32,48,32"
./work/run_exp.sh ds_480 "DinoSeg slim at 480x864, honest" "" --train data/train/images:work/labels_train_v3 --arch dinoseg --h 480 --w 864 --crop-h 480 --crop-w 864 --zoom 1.0,2.0 --closeup 0.2 --lr 1e-3 --bs 8 --encoder "$V" --enc-mult 0.02 --iters 3000 --seed 0
