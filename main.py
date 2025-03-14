# built-in
import os
import gc
import sys
import math
import random
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing

# ml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

# own
from config import cfg
from dataset.load_data import *
from utils.utils import *
from trainer import train
from tester import test


def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_env(cfg.seed)

    if cfg.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    # data loader
    adj = load_adj(cfg.dataset)

    if cfg.dataset == 'metr-la':
        n_vertex = 207
    elif cfg.dataset == 'pems-bay':
        n_vertex = 325
    elif cfg.dataset == 'pemsd7-m':
        n_vertex = 228
    else:
        print(f"[ERROR] Invalid dataset {cfg.dataset}")
        sys.exit()

    cfg.n = n_vertex

    # calculate graph kernel
    L = scaled_laplacian(adj)

    # alternative approximation method: 1st approx - first_approx(W, n)
    Lk = torch.tensor(cheb_poly_approx(L, cfg.Ks, n_vertex),
                      dtype=torch.float32,
                      device=device)

    data_dict, x_stats = data_gen(file_path=f"./data/{cfg.dataset}/vel.csv",
                                  data_config=(34, 5, 5),
                                  n_route=cfg.n)

    train_data = data_dict['train']
    val_data = data_dict['val']
    test_data = data_dict['test']

    blocks = [[1, 32, 64], [64, 32, 128]]

    # train
    model = train(
        train_data=train_data,
        val_data=val_data,
        blocks=blocks,
        kernel=Lk,
        x_stats=x_stats,
        device=device
    )

    # test
    test(
        x_test=test_data,
        x_stats=x_stats,
        model=model,
        n_hist=cfg.n_hist,
        n_pred=cfg.n_pred,
        device=device
    )


