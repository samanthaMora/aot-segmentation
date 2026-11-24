#!/bin/zsh
# Two-stage recipe with the re-reviewed labels (labels_train_v3), kit data only.
# hon3/ft3 never see val (honest, comparable with segb1/ft_a); final3/final3_ft add val GT x4 (candidates for submission).
cd "${0:A:h}/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
L=work/labels_train_v3
V=data/val_in_domain/images:data/val_in_domain/masks:4
S1="--arch segformer --encoder mit_b1 --h 448 --w 800 --iters 5000 --bs 8"
S2="--arch segformer --encoder mit_b1 --h 320 --w 576 --crop-h 320 --crop-w 448 --zoom 0.8,1.5 --iters 1500 --lr 1e-4 --bs 8 --eval-every 500"
exam() {  # name ckpt note
  .venv/bin/python aot/export.py --ckpt $2 --fast-up --fold --h 320 --w 576 --out work/onnx_exp/$1.onnx >> work/logs/$1.log 2>&1 || return
  { echo "# $(date '+%F %T') $1: $3"; .venv/bin/python aot/exam.py work/onnx_exp/$1.onnx 2>&1 | grep EXAM; } >> work/logs/exams.txt
}
exam segb1_320 work/ckpt/segb1.pt "old labels, stage 1 only (448x800), exported at 320x576"
.venv/bin/python aot/train.py --train data/train/images:$L ${=S1} --out work/ckpt/hon3.pt > work/logs/hon3.log 2>&1 && exam hon3 work/ckpt/hon3.pt "v3 labels, stage 1 (honest)"
.venv/bin/python aot/train.py --train data/train/images:$L --resume work/ckpt/hon3.pt ${=S2} --out work/ckpt/ft3.pt > work/logs/ft3.log 2>&1 && exam ft3 work/ckpt/ft3.pt "v3 labels, stage 2 from hon3 (honest)"
.venv/bin/python aot/train.py --train data/train/images:$L $V --val none ${=S1} --out work/ckpt/final3.pt > work/logs/final3.log 2>&1 && exam final3 work/ckpt/final3.pt "v3 labels + val x4, stage 1 (VAL not honest)"
.venv/bin/python aot/train.py --train data/train/images:$L $V --val none --resume work/ckpt/final3.pt ${=S2} --out work/ckpt/final3_ft.pt > work/logs/final3_ft.log 2>&1 && exam final3_ft work/ckpt/final3_ft.pt "v3 labels + val x4, stage 2 from final3 (VAL not honest)"
