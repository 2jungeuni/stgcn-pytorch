import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import torch

from config import cfg
from dataset.load_data import *
from model.base import STGCN
from utils.utils import *


def multi_pred(model,
               test_dataloader,
               n_hist,
               n_pred,
               device='cpu'):
    pred_list = []
    for batch in test_dataloader:
        test_seq = np.copy(batch[:, :n_hist, :, :])

        step_outputs = []
        for j in range(n_pred):
            x_input = torch.tensor(test_seq, dtype=torch.float32, device=device)

            with torch.no_grad():
                pred = model(x_input)
            pred_np = pred.cpu().numpy()

            test_seq[:, :n_hist - 1, :, :] = test_seq[:, 1: n_hist, :, :]
            test_seq[:, n_hist - 1, :, :] = pred_np[:, 0, :, :]

            step_outputs.append(pred_np)

        step_outputs = np.stack(step_outputs, axis=0)
        step_outputs = step_outputs[..., 0, :, :]
        pred_list.append(step_outputs)
    pred_array = np.concatenate(pred_list, axis=1)

    y_ = pred_array
    len_ = pred_array.shape[1]
    return y_, len_


def evaluation(y,
               y_,
               x_stats):
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if not isinstance(y_, torch.Tensor):
        y_ = torch.tensor(y_, dtype=torch.float32)

    mean_ = torch.tensor(x_stats['mean'], dtype=torch.float32)
    std_ = torch.tensor(x_stats['std'], dtype=torch.float32)

    # dimension check
    # - single-step: [batch_size, n_route, 1]
    # - multi-step: [time_step, batch_size, n_route, 1]
    dim = y_.ndim

    if dim == 3:
        v = z_inverse(y, mean_, std_)
        v_ = z_inverse(y_, mean_, std_)

        mape_val = MAPE(v, v_)
        mae_val = MAE(v, v_)
        rmse_val = RMSE(v, v_)

        return torch.tensor([mape_val.item(), mae_val.item(), rmse_val.item()])
    else:
        y = y.permute(1, 0, 2, 3).contiguous()
        y_ = y_.permute(1, 0, 2, 3).contiguous()

        tmp_list = []
        time_step = y_.shape[1]
        for i in range(time_step):
            tmp_res = evaluation(y[:, i, :, :], y_[:, i, :, :], x_stats)
            tmp_list.append(tmp_res.unsqueeze(-1))

        return np.concatenate(tmp_list, axis=1)


def test(x_test,
         x_stats,
         model,
         n_hist,
         n_pred,
         load_path='./output/pemsd7-m_best.pth',
         device='cpu'):
    start_time = time.time()
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f'>> Loading saved model from {load_path} ...')

    # test
    model.eval()

    test_dataloader = DataLoader(x_test, batch_size=32, shuffle=False)
    y_test, len_test = multi_pred(model,
                                  test_dataloader,
                                  n_hist,
                                  n_pred,
                                  device)
    evl = evaluation(x_test[:len_test, -n_pred:, :, :].transpose(0, 1),
                     y_test,
                     x_stats)

    x_test_sample = x_test[:len_test, -n_pred:, :, :].transpose(0, 1)[:, 1, 1, 0].cpu().detach().numpy()
    y_test_sample = y_test[:, 1, 1, 0]

    plt.figure(figsize=(6, 4))
    plt.plot(x_test_sample, label='gt', marker='o', markersize=2)
    plt.plot(y_test_sample, label='pred', marker='o', markersize=2)
    plt.xlabel('time step')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./plot/pemsd7-m/example.png')
    plt.close()

    for ix in range(n_pred):
        te = evl[:, ix]
        print(f'Time step {ix + 1}: MAPE {te[0]:7.3%}; MAE {te[1]:4.3f}; RMSE {te[2]:6.3f}')

    mape_mean = evl[0].mean() * 100
    mae_mean = evl[1].mean()
    rmse_mean = evl[2].mean()
    print(f"[Test] Average: MAPE={mape_mean:.2f}%, MAE={mae_mean:.4f}, RMSE={rmse_mean:.4f}")

    print(f'Model Test Time {time.time() - start_time:.3f} s')
    print('Testing model finished!')


if __name__ == "__main__":
    from dataset.load_data import *

    if cfg.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    # data loader
    adj = load_adj(cfg.dataset)

    if cfg.dataset == 'metra-la':
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

    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    blocks = [[1, 32, 64], [64, 32, 128]]

    model = STGCN(n_hist=cfg.n_hist,
                  Ks=cfg.Ks,
                  Kt=cfg.Kt,
                  blocks=blocks,
                  kernels=Lk,
                  dropout=0.0).to(device)

    checkpoint = torch.load("../stgcn/output/pemsd7-m/pred_45mins/pemsd7-m_best.pth",
                            map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # loss function (no updating, for validating)
    loss_fn = torch.nn.MSELoss()

    # test
    model.eval()
    count = 0
    pred = []
    gt = []
    for x_batch in test_loader:
        x_batch = x_batch.to(device)                    # 32, 21 (# hist + # pred), 228, 1
        y_hat = model(x_batch[:, :cfg.n_hist, :, :])    # 32, 1, 228, 1

        pred.append(y_hat[0, 0, 0, 0].item())
        gt.append(x_batch[:, cfg.n_hist: cfg.n_hist + 1, :, :][0, 0, 0, 0].item())

    plt.plot(pred, label='pred', markersize=2, marker='o', linewidth=2)
    plt.plot(gt, label='gt', markersize=2, marker='o', linewidth=2)

    plt.title("Velocity Prediction (45 mins)", fontsize=16, fontweight='bold')
    plt.xlabel('Time step', fontsize=12, fontweight='bold')
    plt.ylabel('Velocity', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.savefig("test_45.svg")