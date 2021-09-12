#!/bin/bash

trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

source ../venv/bin/activate

# epochs=100
epochs=10
wd=5e-4
bs=1024
lr=1e-2

models=(
    # BasicCNN
    resnet18
    # resnet34
    # resnet50
    # resnet20
    # resnet44
    # densenet121
    resnext50_32x4d
    # wide_resnet50_2
    # vgg11
    # vgg13
    # vgg16
  )

for model in "${models[@]}"; do
  python3 run.py --lr ${lr} --bs ${bs} --wd ${wd} --eps ${epochs} -m ${model} -sn
  python3 run.py --lr ${lr} --bs ${bs} --wd ${wd} --eps ${epochs} -m ${model} -son
# Run twice to rest the error bars
  python3 run.py --lr ${lr} --bs ${bs} --wd ${wd} --eps ${epochs} -m ${model} -sn
  python3 run.py --lr ${lr} --bs ${bs} --wd ${wd} --eps ${epochs} -m ${model} -son
done

