
import time
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as metrics
from typing import Literal, get_args

import torch
import torchinfo
from torch_geometric.loader import DataLoader

from classes import GNNClassifier

import pandas as pd


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)


_TYPES = Literal["test", "train"]

def load_data(type_data: _TYPES, path = "/vols/sbn/uboone/rn325/my_analysis/mlbd_dt_models/DM_GNN/GNN_data"):

    '''
    Function to load pickle files into the script.

    inputs:
        path: str, path to the data that should be loaded. Default: /vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs/
        type_data: str, type of data to be loaded, either "test" or "train"

    output:
        sample: list, list of the graphs, note that this still needs to be passed to device using .to(device)
    '''

    options = get_args(_TYPES)
    assert type_data in options, f"'{type_data}' is not in {options}"

    graphs_dir  = path

    start = time.time()

    print(f"Loading {type_data} graphs...")
    sample = []
    files = sorted([f for f in os.listdir(graphs_dir) if f.startswith(f"{type_data}_graphs_") and f.endswith('.pt')])
    for f in files:
        chunk = torch.load(os.path.join(graphs_dir, f), weights_only=False)
        sample.extend(chunk)
        print(f"Loaded {f}, total so far: {len(sample)}")

    end = time.time()
    print(f"Loaded {len(sample)} graphs in {end-start} seconds.")

    return sample
