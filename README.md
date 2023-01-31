# Orthogonalised Gradients

### Anonymous Repo
Since you can't clone directly from the anonymous repo, try using the script here:
[https://github.com/ShoufaChen/clone-anonymous4open](https://github.com/ShoufaChen/clone-anonymous4open)

The contrastive_learning submodule is Facebook's [Barlow Twins](https://github.com/facebookresearch/barlowtwins) and then adapted with an orthogonalised version of LARS; found [here](https://anonymous.4open.science/r/barlowtwins-2D78/).

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
