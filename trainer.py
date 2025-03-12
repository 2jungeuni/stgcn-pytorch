import sys
import time
import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from dataset.load_data import *
from model.base import STGCN
from utils.utils import *

def l2_loss(pred, target):
    return 0.5 * (pred - target).pow(2).sum()

def compute_loss_and_pred(model,
                          x,
                          n_hist,
                          loss_fn,
                          device=None):
    if device is not None:
        x = x.to(device)

    y_hat = model(x[:, :n_hist, :, :])

    copy_l = loss_fn(
        x[:, n_hist - 1: n_hist, :, :],
        x[:, n_hist: n_hist + 1, :, :]
    )

    train_l = loss_fn(
        y_hat,
        x[:, n_hist: n_hist + 1, :, :]
    )

    single_pred = y_hat[:, 0, :, 0]

    return train_l, copy_l, single_pred


def train(train_data,
          blocks,
          kernel,
          device='cpu'):
    """
    :param train_data: tensor of shape [B, n_hist + 1, N, 1]
    :param blocks: blocks in STGCN
    :param device: 'cuda' / 'mps' / 'cpu'
    """
    n_hist = cfg.n_hist
    n_pred = cfg.n_pred
    Ks, Kt = cfg.Ks, cfg.Kt
    batch_size = cfg.batch_size
    max_epoch = cfg.epochs
    lr = cfg.lr
    opt_str = cfg.opt
    inf_mode = cfg.inf_mode

    n_route = cfg.n

    model = STGCN(n_hist=n_hist,
                  Ks=Ks,
                  Kt=Kt,
                  blocks=blocks,
                  kernels=kernel,
                  dropout=0.0)
    model = model.to(device)

    loss_fn = nn.L1Loss()

    # optimizer
    if opt_str == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif opt_str == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'[ERROR] optimizer {opt_str} is not defined')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    train_loss_dict = {"train loss": [],
                       "copy loss": [],
                       "execution time": []}
    for epoch in range(max_epoch):
        start_time = time.time()
        model.train()

        train_loss_val = 0.0
        copy_loss_val = 0.0
        num_batch = 0
        count = 0
        for x_batch in train_loader:
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            train_loss, copy_loss, _ = compute_loss_and_pred(
                model,
                x_batch,
                n_hist,
                loss_fn,
                device
            )

            loss = train_loss

            loss.backward()
            optimizer.step()

            train_loss_val += train_loss.item()
            copy_loss_val += copy_loss.item()
            count += 1

            if (count % 50) == 0:
                print(
                    f'Epoch {epoch:4d} | Step {count:3d} | Avg. Train Loss {train_loss_val / count:.3f} | Avg. Copy Loss {copy_loss_val / count:.3f}')

        scheduler.step()
        training_time = time.time() - start_time

        train_loss_dict["train loss"].append(train_loss_val / count)
        train_loss_dict["copy loss"].append(copy_loss_val / count)
        train_loss_dict["execution time"].append(training_time)

        print(f'Epoch {epoch:4d} | Training Time {training_time:.3f} secs')

        if (epoch + 1) % cfg.save == 0:
            with open("output/pemsd7-m/train_loss.pickle", "wb") as f:
                pickle.dump(train_loss_dict, f)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            f'./output/{cfg.dataset}_{epoch+1}.pth')

if __name__ == '__main__':
    device ='cuda:0'
    data_config = (34, 5, 5)

    # data loader
    adj = load_adj(cfg.dataset)

    if cfg.dataset == 'metr-la':
        n_vertex = 207
        cfg.n = n_vertex
    elif cfg.dataset == 'pems-bay':
        n_vertex = 325
        cfg.n = n_vertex
    elif cfg.dataset == 'pemsd7-m':
        n_vertex = 228
        cfg.n = n_vertex
    else:
        print(f"[ERROR] Invalid dataset {cfg.dataset}")
        sys.exit()

    # calculate graph kernel
    L = scaled_laplacian(adj)

    # alternative approximation method: 1st approx - first_approx(W, n)
    Lk = torch.tensor(cheb_poly_approx(L, cfg.Ks, n_vertex), dtype=torch.float32, device=device)

    data_dict, x_stats = data_gen(file_path="./data/pemsd7-m/vel.csv", data_config=data_config, n_route=cfg.n)
    train_loader = DataLoader(data_dict['train'], batch_size=32, shuffle=True)

    blocks = [[1, 32, 64], [64, 32, 128]]
    train(train_loader, blocks, Lk, device)