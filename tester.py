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
         load_path='./output/pemsd7-m/pemsd7-m.pth',
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

    x_test_sample = x_test[:len_test, -n_pred, :, :].transpose(0, 1)[:, 1, 1, 0].cpu().detach().numpy()
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

    print(f'Model Test Time {time.time() - start_time:.3f} s')
    print('Testing model finished!')

