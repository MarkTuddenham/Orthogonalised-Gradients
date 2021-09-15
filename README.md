# Orthogonalised Gradients

### Anonymous Repo
Since you can't clone directly from the anonymous repo, try using the script here:
[https://github.com/ShoufaChen/clone-anonymous4open](https://github.com/ShoufaChen/clone-anonymous4open)

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
