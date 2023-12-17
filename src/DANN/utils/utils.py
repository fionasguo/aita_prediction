"""Utility functions"""

import random
import os
import psutil
import subprocess
import logging
import numpy as np
import torch
from ast import literal_eval
import transformers


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_gpu_memory_map():
    """
    Get the current gpu usage.
    returns a dict: key - device id int; val - memory usage in GB (int).
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return {
            'cpu_count': os.cpu_count(),
            '% RAM used': psutil.virtual_memory()[2]
        }

    gpu_memory_map = {}
    for i in range(count_devices()):
        gpu_memory_map[i] = round(torch.cuda.memory_allocated(i)/1024/1024/1024,2)
    return gpu_memory_map


def count_devices():
    """
    Get the number of GPUs, if using CPU only, return 1
    """
    n_devices = torch.cuda.device_count()

    if n_devices == 0:
        n_devices = 1

    return n_devices
