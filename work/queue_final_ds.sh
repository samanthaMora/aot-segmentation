#!/bin/zsh
cd "${0:A:h}/.."
C="--arch dinoseg --h 448 --w 800 --crop-h 448 --crop-w 800 --zoom 1.0,2.0 --closeup 0.2 --lr 1e-3 --iters 3000 --bs 8 --enc-mult 0.02"
V="vit_small_patch16_dinov3.lvd1689m|c24,32,48,32"
T=data/train/images:work/labels_train_v3
# candidate for submission: kit train (v3 labels) + kit val GT x4, nothing else
./work/run_exp.sh ds_final_a "FINAL candidate: DinoSeg slim, v3 + val x4 (VAL not honest)" "" --train $T data/val_in_domain/images:data/val_in_domain/masks:4 --val none ${=C} --encoder "$V" --seed 0
# honest checks that keep running afterwards: second seed, and a smaller/faster ViT input as latency fallback
./work/run_exp.sh ds_slim_ft02_s1 "DinoSeg slim, backbone lr x0.02, seed 1, v3, honest" "" --train $T ${=C} --encoder "$V" --seed 1
./work/run_exp.sh ds_slim_416 "DinoSeg slim at 416x736 (ViT 13x23 tokens), v3, honest" "" --train $T ${=C/--h 448 --w 800 --crop-h 448 --crop-w 800/--h 416 --w 736 --crop-h 416 --crop-w 736} --encoder "$V" --seed 0
