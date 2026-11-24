#!/bin/zsh
# Close-up augmentation test (honest: kit train only, v3 labels, stage-1 recipe), two seeds. Compare with hon3 / hon3_s1.
cd "${0:A:h}/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
while pgrep -f "work/queue_seeds.sh" >/dev/null; do sleep 30; done
S1="--arch segformer --encoder mit_b1 --h 448 --w 800 --iters 5000 --bs 8"
run() {  # name note train-args...
  n=$1; note=$2; shift 2
  .venv/bin/python aot/train.py "$@" ${=S1} --out work/ckpt/$n.pt > work/logs/$n.log 2>&1 || return
  .venv/bin/python aot/export.py --ckpt work/ckpt/$n.pt --fast-up --fold --h 320 --w 576 --out work/onnx_exp/$n.onnx >> work/logs/$n.log 2>&1 || return
  { echo "# $(date '+%F %T') $n: $note"; .venv/bin/python aot/exam.py work/onnx_exp/$n.onnx 2>&1 | grep EXAM; } >> work/logs/exams.txt
}
run cu25_s0 "v3 labels, stage 1, --closeup 0.25, seed 0 (honest)" --train data/train/images:work/labels_train_v3 --closeup 0.25 --seed 0
run cu25_s1 "v3 labels, stage 1, --closeup 0.25, seed 1 (honest)" --train data/train/images:work/labels_train_v3 --closeup 0.25 --seed 1
