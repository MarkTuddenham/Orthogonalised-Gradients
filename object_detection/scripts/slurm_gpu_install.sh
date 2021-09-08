#!/bin/sh
#SBATCH --partition=ecsall
#SBATCH --account=ecsstaff
#SBATCH --requeue
#SBATCH --output=logs/install_%A_%a.out
#SBATCH --error=logs/install_%A_%a.err
#SBATCH --job-name=Install_with_gpu_avail
#SBATCH --gres=gpu:1

#SBATCH --time=1:00:00

module load gcc/6.4.0
module load cuda/10.2

# source /ECShome/mt9g15/venv/bin/activate
source /ECShome/mt9g15/orth_sgd/bin/activate

# pip uninstall -y apex maskrcnn-benchmark mmdet

export INSTALL_DIR=/ECShome/mt9g15/install/
mkdir -p $INSTALL_DIR

echo $INSTALL_DIR
ls -al $INSTALL_DIR

# install pycocotools
cd $INSTALL_DIR
# git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
# git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
# git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
# git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

# pip install mmcv-full --force-reinstall --no-cache-dir -U -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.1/index.html
cd $INSTALL_DIR
# git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -r requirements.txt
MMCV_WITH_OPS=1 pip install -e .

cd $INSTALL_DIR
# git clone https://github.com/open-mmlab/mmdetection.git 
cd mmdetection
pip install --force-reinstall --no-cache-dir -U -r requirements/build.txt
pip install --force-reinstall --no-cache-dir -U -e .
cd -


# cd UniverseNet
# pip install --force-reinstall --no-cache-dir -U -r requirements.txt
# pip install --force-reinstall --no-cache-dir -U -e .

unset INSTALL_DIR

# git clone https://github.com/cocodataset/cocoapi.git
# git clone https://github.com/mcordts/cityscapesScripts.git
# git clone https://github.com/NVIDIA/apex.git
# git clone https://github.com/open-mmlab/mmdetection.git 
# git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
