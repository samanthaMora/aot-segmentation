#!/bin/zsh
# Follow-ups on the DinoSeg recipe (honest: kit train only, v3 labels). Reference: ds_slim_ft02 (EXAM 0.767 / horse 36% / VAL 0.857) and its seed 1.
cd "${0:A:h}/.."
while pgrep -f "work/queue_final_ds.sh" >/dev/null; do /bin/sleep 30; done
C="--arch dinoseg --h 448 --w 800 --crop-h 448 --crop-w 800 --zoom 1.0,2.0 --closeup 0.2 --lr 1e-3 --bs 8"
V="vit_small_patch16_dinov3.lvd1689m|c24,32,48,32"
T=data/train/images:work/labels_train_v3
./work/run_exp.sh ds_horses "DinoSeg slim + horse copy-paste (kit cut-outs), honest" "" --train $T --horses $T:work/horses_ok_train ${=C} --encoder "$V" --enc-mult 0.02 --iters 3000 --seed 0
./work/run_exp.sh ds_ft01 "DinoSeg slim, backbone lr x0.01, honest" "" --train $T ${=C} --encoder "$V" --enc-mult 0.01 --iters 3000 --seed 0
./work/run_exp.sh ds_long "DinoSeg slim, 5000 its, honest" "" --train $T ${=C} --encoder "$V" --enc-mult 0.02 --iters 5000 --seed 0
