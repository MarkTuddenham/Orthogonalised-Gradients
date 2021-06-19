#!/bin/sh
#SBATCH --mail-user=mt9g15@soton.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=ecsall
#SBATCH --account=ecsstaff
#SBATCH --requeue
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=Orthogonal_Gradients
#SBATCH --gres=gpu:4

#SBATCH --cpus-per-task=60
#SBATCH --time=120:00:00
#SBATCH --mem=300G


source /ECShome/mt9g15/venv/bin/activate

epochs=100
wd=5e-4
bs=1024
lr=1e-1

models=(
    # BasicCNN
    # resnet18
    # resnet34
    resnet50
    # inception_v3 # Only for imagenet -> requires input Nx3x299x299
    # densenet121
    # densenet161
    # resnext50_32x4d
    # wide_resnet50_2
    # vgg11
    # vgg13
    # vgg16
  )


for model in "${models[@]}"; do
  python3 run.py --save --avoid-tqdm --lr=${lr} --batch-size=${bs} --weight-decay=${wd} --epochs=${epochs} --model=${model}
  # python3 run.py --save --avoid-tqdm --lr=${lr} --batch-size=${bs} --weight-decay=${wd} --epochs=${epochs} --model=${model} --orth
done

python3 analysis.py

