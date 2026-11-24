#!/bin/zsh
# run_exp.sh NAME "NOTE" "EXPORT ARGS" TRAIN ARGS...   -> train, export to work/onnx_exp/NAME.onnx, append exam line to work/logs/exams.txt
cd "${0:A:h}/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
n=$1; note=$2; exp=$3; shift 3
.venv/bin/python aot/train.py "$@" --out work/ckpt/$n.pt > work/logs/$n.log 2>&1 || exit 1
.venv/bin/python aot/export.py --ckpt work/ckpt/$n.pt --fast-up ${=exp} --out work/onnx_exp/$n.onnx >> work/logs/$n.log 2>&1 || exit 1
{ echo "# $(date '+%F %T') $n: $note | $*"; .venv/bin/python aot/exam.py work/onnx_exp/$n.onnx 2>&1 | grep EXAM; } >> work/logs/exams.txt
tail -1 work/logs/exams.txt
