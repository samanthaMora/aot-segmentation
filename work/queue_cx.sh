#!/bin/zsh
# Ceiling test of a stronger pretrained backbone (ConvNeXt-tiny DINOv3 + FPN), honest (kit train only, v3 labels).
# Runs right after cu25_s0; takes over cu25_s1 from queue_cu.sh so the backbone test goes first.
cd "${0:A:h}/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
while ! grep -q "^cu25_s0.onnx" work/logs/exams.txt; do pgrep -f "work/queue_cu.sh" >/dev/null || break; sleep 15; done
pkill -f "work/queue_cu.sh"; sleep 2; pkill -f "out work/ckpt/cu25_s1.pt"; sleep 5
ex() {  # name ckpt note export-args...
  n=$1; ck=$2; note=$3; shift 3
  .venv/bin/python aot/export.py --ckpt $ck --fast-up "$@" --out work/onnx_exp/$n.onnx >> work/logs/$n.log 2>&1 || return
  { echo "# $(date '+%F %T') $n: $note"; .venv/bin/python aot/exam.py work/onnx_exp/$n.onnx 2>&1 | grep EXAM; } >> work/logs/exams.txt
}
ex hon3_448 work/ckpt/hon3.pt "hon3 (B1, v3, honest) exported at 448x800 (too slow to submit; reference)" --fold --h 448 --w 800
.venv/bin/python aot/train.py --train data/train/images:work/labels_train_v3 --arch fpn --encoder tu-convnext_tiny.dinov3_lvd1689m --h 448 --w 800 --iters 5000 --bs 8 --lr 2e-4 --seed 0 --out work/ckpt/cx_s0.pt > work/logs/cx_s0.log 2>&1 && {
  ex cx_s0_448 work/ckpt/cx_s0.pt "ConvNeXt-tiny DINOv3 + FPN, v3, honest, exported at 448x800 (latency not met; ceiling test)" --h 448 --w 800
  ex cx_s0_320 work/ckpt/cx_s0.pt "same, exported at 320x576" --h 320 --w 576
  ex cx_s0_256 work/ckpt/cx_s0.pt "same, exported at 256x448" --h 256 --w 448
}
S1="--arch segformer --encoder mit_b1 --h 448 --w 800 --iters 5000 --bs 8"
.venv/bin/python aot/train.py --train data/train/images:work/labels_train_v3 --closeup 0.25 --seed 1 ${=S1} --out work/ckpt/cu25_s1.pt > work/logs/cu25_s1.log 2>&1 && ex cu25_s1 work/ckpt/cu25_s1.pt "v3 labels, stage 1, --closeup 0.25, seed 1 (honest)" --fold --h 320 --w 576
