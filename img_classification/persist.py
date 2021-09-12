#!/home/mark/phd/venv/bin/python
# coding: utf-8
"""Functions to persist experiment results."""
from typing import Dict
from typing import Any
from typing import Optional

import torch as th

from hashlib import sha256

from base64 import b64encode
from os import makedirs
from os.path import isdir
from os.path import basename
from os.path import getctime
from json import dumps
from json import dump
from glob import glob


Params = Dict[str, Any]
HASH_LEN: int = 10


def save_tensor(t: th.Tensor, path: str, params: Optional[Params]=None, overwrite:bool=False) -> None:
    """Save a tensor.

        Parameters:
            t:          Tensor to save
            path:       Place to save tensor to (includes filename)
            overwrite:    Overwrite the last save.
    """
    path = get_path_with_hash(path, params)
    file_name: int = 0

    if isdir(path):
        last_file = get_last_file(path)
        file_name = int(basename(last_file)) + int(not overwrite)
    else:
        makedirs(path, exist_ok=True)
        with open(f'{path}/params.json', 'w') as f:
            dump(params, f)

    th.save(t, f'{path}/{file_name}')



def load_tensor(path: str, params: Optional[Params]) -> th.Tensor:
    """Load all the tensors into a stack."""
    path = get_path_with_hash(path, params)
    files = glob(f'{path}/*')
    if len(files) == 0:
        return None
    tensors = [th.load(f) for f in files if not f.endswith('.json')]
    tensors = [i if isinstance(i, th.Tensor) else th.tensor(i) for i in tensors]
    return th.stack(tensors)

def load_last_tensor(path: str, params: Optional[Params]) -> Optional[th.Tensor]:
    """Load only the last saved tensor."""
    path = get_path_with_hash(path, params)
    f = get_last_file(path)
    return th.load(f) if f else None


def get_last_file(path: str) -> Optional[str]:
    try:
        path = max(glob(f'{path}/*'), key=getctime)
    except ValueError:
        path = None
    return path


def get_path_with_hash(path: str, params: Optional[Params]) -> str:
    if params is not None:
        h = hash_p(params)
    return f'{path}_{h}' if params else path


def hash_p(params: Params) -> str:
    s: str = dumps(params)
    h: str = sha256(s.encode()).hexdigest()
    b64: bytes = b64encode(h.encode())
    return b64.decode('utf-8')[:HASH_LEN]
