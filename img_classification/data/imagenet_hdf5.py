import io
import os
import pickle
import logging

import torch as th

import h5py
from PIL import Image
from torchvision.datasets import VisionDataset
# from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
# from torchvision.transforms import RandomCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Resize
from torchvision.transforms import CenterCrop

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


# from utils import split

image_size = 224
# image_size = 32

input_size = (3, image_size, image_size)
batch_size = 1024
num_workers = 20
pin_memory = False

train_set = None
test_set = None
data_root = '/ECShome/ECSdata/data_sets/imagenet_hdf5'


def set_data_path(path):
    global data_root
    data_root = path


def get_train_gen(batch_size=batch_size):
    """Get the generator for the train set."""
    global train_set
    if not train_set:
        imagenet()
    return th.utils.data.DataLoader(
        train_set,
        pin_memory=pin_memory,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


def get_valid_gen(batch_size=batch_size):
    """Get the generator for the validation set."""
    global valid_set, test_set
    if not test_set:
        imagenet()
    valid_set = test_set
    return th.utils.data.DataLoader(
        valid_set,
        pin_memory=pin_memory,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def get_test_gen(batch_size=batch_size):
    """Get the generator for the test set."""
    global test_set
    if not test_set:
        imagenet()
    return th.utils.data.DataLoader(
        test_set,
        pin_memory=pin_memory,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


# @split
def imagenet(args=None):
    global train_set, test_set, data_root
    transform_train, transform_test = imagenet_transforms(args)

    train_set = ImageNetHDF5(root=f'{data_root}/train', transform=transform_train)
    test_set = ImageNetHDF5(root=f'{data_root}/val', transform=transform_test)

    # root = '/ECShome/ECSdata/data_sets/ILSVRC2012'
    # train_set = ImageFolder(root=f'{root}/train', transform=transform_train)
    # test_set = ImageFolder(root=f'{root}/val', transform=transform_test)

    return train_set, test_set


def imagenet_transforms(args):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base = [ToTensor(), normalize]

    # if args.augment:
    transform = [RandomResizedCrop(image_size), RandomHorizontalFlip()] + base
    # else:
    #     transform = base

    transform_train = Compose(transform)
    transform_test = Compose([Resize(256), CenterCrop(image_size)] + base)
    return transform_train, transform_test


class ImageNetHDF5(VisionDataset):
    def __init__(self, root, cache_size=500, transform=None):
        super(ImageNetHDF5, self).__init__(root, transform=transform, target_transform=None)

        self.dest = pickle.load(open(os.path.join(root, 'dest.p'), 'rb'))
        self.cache = {}
        self.cache_size = cache_size

        targets = sorted(list(filter(lambda f: '.hdf5' in f, os.listdir(root))))
        self.targets = {f[:-5]: i for i, f in enumerate(targets)}
        self.fill_cache()

    def load(self, file, i):
        with h5py.File(os.path.join(self.root, file + '.hdf5'), 'r') as f:
            return f['data'][i]

    def fill_cache(self):
        logger.info(f'Filling Cache with {self.cache_size} files from {self.root}')
        files = (f[:-5] for f in list(
            filter(lambda f: '.hdf5' in f, os.listdir(self.root))
        )[:self.cache_size])

        last_ten_percent = 0
        for i, file in enumerate(files):
            if (i / self.cache_size) > last_ten_percent + 0.1:
                last_ten_percent += 0.1
                logger.info(f'Filling Cache: {last_ten_percent:.0%}')

            with h5py.File(os.path.join(self.root, file + '.hdf5'), 'r') as f:
                self.cache[file] = list(f['data'])

        logger.info('Filled Cache')

    def load_from_cache(self, file, i):
        if file in self.cache:
            return self.cache[file][i]
        return self.load(file, i)

    def __getitem__(self, index):
        dest, i = self.dest[index]

        sample = self.load_from_cache(dest, i)

        sample = Image.open(io.BytesIO(sample))
        sample = sample.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.targets[dest]

    def __len__(self):
        return len(self.dest)
