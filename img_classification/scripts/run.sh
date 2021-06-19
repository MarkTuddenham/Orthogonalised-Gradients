#!/bin/bash

trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

source /home/mark/phd/venv/bin/activate

epochs=200
wd=5e-4
bs=1024
# bs=128
lr=1e-3

models=(
    BasicCNN
    resnet18
    # resnet34
    # resnet50
    resnet20
    resnet44
    densenet121
    # resnext50_32x4d
    wide_resnet50_2
  )

for model in "${models[@]}"; do
  python3 run.py --save --lr=${lr} --batch-size=${bs} --weight-decay=${wd} --epochs=${epochs} --model=${model}
  python3 run.py --save --lr=${lr} --batch-size=${bs} --weight-decay=${wd} --epochs=${epochs} --model=${model} --orth
done

