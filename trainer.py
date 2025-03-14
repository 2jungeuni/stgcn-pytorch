import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import cfg
from dataset.load_data import *
from model.base import STGCN
from utils.utils import *
from tester import multi_pred, evaluation


def l2_loss(pred, target):
    return 0.5 * (pred - target).pow(2).sum()


def inference(model,
              val_data,
              x_stats,
              batch_size,
              n_hist,
              n_pred,
              device='cpu'):
    model.eval()
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    y_val, len_val = multi_pred(
        model,
        val_loader,
        n_hist,
        n_pred,
        device,
    )

    y_true = val_data[:len_val, -n_pred:, :, :].transpose(0, 1)
    evl_val = evaluation(y_true, y_val, x_stats)

    for i in range(n_pred):
        print(f"[Val] Step {i + 1}: "
              f"MAPE={evl_val[0, i] * 100:6.2f}%, "
              f"MAE={evl_val[1, i]:.4f}, "
              f"RMSE={evl_val[2, i]:.4f}")

    mape_mean = evl_val[0].mean() * 100
    mae_mean = evl_val[1].mean()
    rmse_mean = evl_val[2].mean()
    print(f"[Val] Average: MAPE={mape_mean:.2f}%, MAE={mae_mean:.4f}, RMSE={rmse_mean:.4f}")

    return mape_mean, mae_mean, rmse_mean


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
          val_data,
          blocks,
          kernel,
          x_stats,
          device='cpu'):
    n_hist = cfg.n_hist
    n_pred = cfg.n_pred
    Ks, Kt = cfg.Ks, cfg.Kt
    batch_size = cfg.batch_size
    max_epoch = cfg.epochs
    lr = cfg.lr
    opt_str = cfg.opt

    model = STGCN(n_hist=n_hist,
                  Ks=Ks,
                  Kt=Kt,
                  blocks=blocks,
                  kernels=kernel,
                  dropout=0.0).to(device)

    loss_fn = nn.MSELoss()

    if opt_str == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif opt_str == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'[ERROR] the optimizer {opt_str} is not defined')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    train_loss_dict = {"train_loss": [],
                       "copy_loss": [],
                       "val_loss": [],
                       "val_mape": [],
                       "val_rmse": [],
                       "val_mae": [],
                       "execution_time": []}

    best_val_loss = float('inf')
    best_epoch = 0
    for epoch in range(max_epoch):
        start_time = time.time()

        # train
        model.train()

        train_loss_val = 0.0
        copy_loss_val = 0.0
        count = 0
        for x_batch in train_loader:
            x_batch = x_batch.to(device)

            optimizer.zero_grad()

            train_loss, copy_loss, _ = compute_loss_and_pred(model,
                                                             x_batch,
                                                             n_hist,
                                                             loss_fn,
                                                             device)

            loss = train_loss
            loss.backward()
            optimizer.step()

            train_loss_val += train_loss.item()
            copy_loss_val += copy_loss.item()
            count += 1

        scheduler.step()
        training_time = time.time() - start_time
        train_loss_val /= count
        copy_loss_val /= count

        # validation
        model.eval()

        val_loss_val = 0.0
        val_count = 0
        with torch.no_grad():
            for x_batch in val_loader:
                x_batch = x_batch.to(device)

                v_loss, c_loss, _ = compute_loss_and_pred(
                    model,
                    x_batch,
                    n_hist,
                    loss_fn,
                    device
                )

                val_loss_val += v_loss.item()
                val_count += 1

        val_loss_val /= val_count

        train_loss_dict["train_loss"].append(train_loss_val)
        train_loss_dict["copy_loss"].append(copy_loss_val)
        train_loss_dict["val_loss"].append(val_loss_val)
        train_loss_dict["execution_time"].append(training_time)

        print(f"Epoch {epoch + 1:4d} | "
              f"Train Loss: {train_loss_val:.4f} | "
              f"Copy Loss: {copy_loss_val:.4f} | "
              f"Val Loss: {val_loss_val:.4f} | "
              f"Train Time: {training_time:.2f} secs")

        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'./output/{cfg.dataset}_best.pth')

        with torch.no_grad():
            print('[Validation Metrics]')
            mape, mae, rmse = inference(
                model=model,
                val_data=val_data,
                x_stats=x_stats,
                batch_size=batch_size,
                n_hist=n_hist,
                n_pred=n_pred,
                device=device
            )

        train_loss_dict["val_mape"].append(mape)
        train_loss_dict["val_mae"].append(mae)
        train_loss_dict["val_rmse"].append(rmse)

        if (epoch + 1) % cfg.save == 0:
            with open("./output/pemsd7-m/train_loss.pickle", "wb") as f:
                pickle.dump(train_loss_dict, f)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'./output/{cfg.dataset}_{epoch+1}.pth')

    print(f"Training done. Best val loss = {best_val_loss:.4f} at epoch {best_epoch}.")
    return model


if __name__ == '__main__':
    device ='cuda:0' if torch.cuda.is_available() else 'cpu'
    data_config = (34, 5, 5)

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

    data_dict, x_stats = data_gen(file_path="./data/pemsd7-m/vel.csv",
                                  data_config=data_config,
                                  n_route=cfg.n)

    train_data = data_dict['train']
    val_data = data_dict['val']

    blocks = [[1, 32, 64], [64, 32, 128]]

    train(
        train_data=train_data,
        val_data=val_data,
        blocks=blocks,
        kernel=Lk,
        x_stats=x_stats,
        device=device
    )