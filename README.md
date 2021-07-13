# Orthogonalised SGD


## Install & use package

```bash
pip install .
```
And then at the top of your main python script:

```python
from orth_optim import hook
hook()
```
Now SGD has an orthogonal option
```python
torch.optim.SGD(model.parameters(),
                lr=1e-3,
                momentum=0.9,
                orth=True)
```



## Install experiment dependencies

This may take some time
```bash
pip install -r requirements.txt
pip install .
```

## Image Classification (CIFAR10 & ImageNet)

```bash
cd img_classificaton
python run.py -h
```

## Image Segmentation (COCO)

Sync the submodules.

Add

```python
from orth_optim import hook_mmcv
hook_mmcv()
```
to the top of `object_detection/UniverseNet/tools/train.py`

For distributed training
```bash
GPUS=4 ./UniverseNet/tools/dist_train.sh  UniverseNet/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py 4 
GPUS=4 ./UniverseNet/tools/dist_train.sh ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_orth.py 4 
```

## NLP

