#!/bin/sh
#SBATCH --mail-user=mt9g15@soton.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=ecsall
#SBATCH --account=ecsstaff
#SBATCH --requeue
#SBATCH --output=logs/coco/%A_%a.out
#SBATCH --error=logs/coco/%A_%a.err
#SBATCH --job-name=Orthogonal_Gradients_COCO
#SBATCH --gres=gpu:4

#SBATCH --cpus-per-task=60
#SBATCH --time=120:00:00
#SBATCH --mem=300G

##SBATCH --array=0-5%2

source /ECShome/mt9g15/orth_sgd/bin/activate
# module load gcc

cd UniverseNet

# GPUS=4 ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py 4 
GPUS=4 ./tools/dist_train.sh ../img_seg_configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_orth.py 4 

