#!/bin/bash

trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

source ../venv/bin/activate

wd=5e-4
lr=1e-2
model=resnet18
epochs=50

b_sizes=( 4 8 16 32 64 128 256 512 1024 2048 )

for bs in "${b_sizes[@]}"; do
  python3 run.py --lr ${lr} --bs ${bs} --wd ${wd} --eps ${epochs} -m ${model} -as
  python3 run.py --lr ${lr} --bs ${bs} --wd ${wd} --eps ${epochs} -m ${model} -aso
done

