#!/bin/zsh
# Second seed of the stage-1 recipe (448x800, 5000 its) for old vs v3 labels with the current train.py, plus a second seed of the final.
cd "${0:A:h}/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
V=data/val_in_domain/images:data/val_in_domain/masks:4
S1="--arch segformer --encoder mit_b1 --h 448 --w 800 --iters 5000 --bs 8"
run() {  # name note train-args...
  n=$1; note=$2; shift 2
  .venv/bin/python aot/train.py "$@" ${=S1} --out work/ckpt/$n.pt > work/logs/$n.log 2>&1 || return
  .venv/bin/python aot/export.py --ckpt work/ckpt/$n.pt --fast-up --fold --h 320 --w 576 --out work/onnx_exp/$n.onnx >> work/logs/$n.log 2>&1 || return
  { echo "# $(date '+%F %T') $n: $note"; .venv/bin/python aot/exam.py work/onnx_exp/$n.onnx 2>&1 | grep EXAM; } >> work/logs/exams.txt
}
run hon_old_s1 "old labels, stage 1, seed 1 (honest)" --train data/train/images:work/labels_train --seed 1
run hon3_s1 "v3 labels, stage 1, seed 1 (honest)" --train data/train/images:work/labels_train_v3 --seed 1
run final3_s1 "v3 labels + val x4, stage 1, seed 1 (VAL not honest)" --train data/train/images:work/labels_train_v3 $V --val none --seed 1
run hon_old_s0 "old labels, stage 1, seed 0, current train.py (honest)" --train data/train/images:work/labels_train --seed 0
