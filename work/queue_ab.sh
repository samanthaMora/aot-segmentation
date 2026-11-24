#!/bin/zsh
# A/B of the training labels (honest: kit train only, val never seen). A = labels_train (old), B = labels_train_v3 (re-reviewed).
cd "${0:A:h}/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
while ! grep -q "^f2_b0_448_zoom.onnx" work/logs/exams.txt; do
  pgrep -f "out work/ckpt/f2_b0_448_zoom.pt" >/dev/null || pgrep -f "f2_b0_448_zoom" >/dev/null || break
  sleep 20
done
run() {
  n=$1; lab=$2; seed=$3
  args="--encoder mit_b1 --h 320 --w 576 --crop-h 320 --crop-w 448 --zoom 0.8,1.5 --iters 4000 --seed $seed"
  .venv/bin/python aot/train.py --train data/train/images:$lab --arch segformer --bs 8 --eval-every 500 ${=args} --out work/ckpt/$n.pt > work/logs/$n.log 2>&1 || return
  .venv/bin/python aot/export.py --ckpt work/ckpt/$n.pt --fast-up --fold --out work/onnx_exp/$n.onnx >> work/logs/$n.log 2>&1 || return
  { echo "# $(date '+%F %T') $n: labels=$lab $args"; .venv/bin/python aot/exam.py work/onnx_exp/$n.onnx 2>&1 | grep EXAM; } >> work/logs/exams.txt
}
run ab_old_s0 work/labels_train 0
run ab_new_s0 work/labels_train_v3 0
run ab_new_s1 work/labels_train_v3 1
run ab_old_s1 work/labels_train 1
