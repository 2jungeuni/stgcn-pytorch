import sys
import time
import numpy as np

import torch

from config import cfg
from dataset.load_data import *
from model.base import STGCN
from utils.utils import *


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


def multi_pred(model,
               test_dataloader,
               batch_size,
               n_hist,
               n_pred,
               step_idx,
               device='cpu',
               dynamic_batch=True):
    """
    :param model: pretrained model
    :param test_dataloader: sequence for testing
    :param batch_size: the number of batch
    :param n_hist: history size
    :param n_pred: predicted future horizons
    :return:
    """
    model.eval()
    pred_list = []
    print("n hist: ", n_hist)
    print("n_pred: ", n_pred)
    for batch in test_dataloader:
        test_seq = np.copy(batch[:, :n_hist, :, :])

        step_outputs = []

        for j in range(n_pred):
            x_input = torch.tensor(test_seq, dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = model(x_input)
            pred_np = pred.cpu().numpy()

            test_seq[:, :n_hist - 1, :, :] = test_seq[:, 1:n_hist, :, :]
            test_seq[:, n_hist - 1, :, :] = pred_np[:, 0, :, :]

            step_outputs.append(pred_np)

        step_outputs = np.stack(step_outputs, axis=0)
        step_outputs = step_outputs[..., 0, :, :]           # n_pred, B, N, 1
        pred_list.append(step_outputs)

    pred_array = np.concatenate(pred_list, axis=1)          # n_pred, num_batch * batch_size, N, 1

    y_= pred_array
    len_ = pred_array.shape[1]
    return y_, len_


def test(inputs,
         x_stats,
         model,
         batch_size,
         n_hist,
         n_pred,
         inf_mode,
         load_path='./output/pemsd7-m/pemsd7-m_1000.pth',
         device='cpu'):
    start_time = time.time()

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'>> Loading saved model from {load_path} ...')

    step_idx = n_pred - 1
    time_idx = [step_idx]
    if inf_mode == 'sep':
        step_idx = n_pred - 1
        tme_idx = [step_idx]        # 2, 5, 8
    elif inf_mode == 'merge':
        step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
    else:
        raise ValueError(f'[ERROR] Test mode {inf_mode} is not defined.')

    x_test = data_dict['test']
    test_dataloader = DataLoader(x_test, batch_size=32, shuffle=True)
    y_test, len_test = multi_pred(model,
                                  test_dataloader,
                                  batch_size,
                                  n_hist,
                                  n_pred,
                                  step_idx,
                                  device)
    evl = evaluation(x_test[:len_test, -n_pred:, :, :].transpose(0, 1),
                     y_test,
                     x_stats)

    for ix in range(n_pred):
        te = evl[:, ix]
        print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')

    print(f'Model Test Time {time.time() - start_time:.3f} s')
    print('Testing model finished!')


if __name__ == '__main__':
    device = 'cuda:0'
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
        print(f'[ERROR] Invalid dataset {cfg.dataset}')
        sys.exit()

    # calculate graph kernel
    L = scaled_laplacian(adj)

    # alternative approximation method: 1st approx - first_approx(W, n)
    Lk = torch.tensor(cheb_poly_approx(L, cfg.Ks, n_vertex), dtype=torch.float32, device=device)

    data_dict, x_stats = data_gen(file_path='./data/pemsd7-m/vel.csv', data_config=data_config, n_route=cfg.n)

    blocks = [[1, 32, 64], [64, 32, 128]]
    model = STGCN(
        n_hist=cfg.n_hist,
        Ks=cfg.Ks,
        Kt=cfg.Kt,
        blocks=blocks,
        kernels=Lk,
        dropout=0.0
    )
    model = model.to(device)

    test(data_dict,
         x_stats,
         model,
         cfg.batch_size,
         n_hist=cfg.n_hist,
         n_pred=cfg.n_pred,
         inf_mode=cfg.inf_mode,
         device=device)