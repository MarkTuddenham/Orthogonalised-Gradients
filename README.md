# Orthogonalised Gradients

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

## Barlow Twins

```bash
python main.py --workers 12 --batch-size 1024 -o /path/to/imagenet
```
