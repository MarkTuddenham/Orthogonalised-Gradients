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
to the top of img_segmentation/UniverseNet/tools/train.py

## NLP

