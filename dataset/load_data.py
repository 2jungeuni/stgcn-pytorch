import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset, DataLoader

from utils.utils import *

def load_adj(name):
    path = "./data"
    path = os.path.join(path, name)
    adj = sp.load_npz(os.path.join(path, "adj.npz"))
    adj = adj.tocsc()

    return adj.toarray()

def seq_gen(len_seq,
            data_seq,
            offset,
            n_frame,
            n_route,
            day_slot,
            C_0=1):
    """
    :param len_seq: the number of days
    :param data_seq: a raw sequence [T, n_route]
    :param offset: the start day index
    :param n_frame: the length of a sequence
    :param n_route:the number of nodes
    :param day_slot: the number of time slots in a day, 288 (12 * 24)
    :param C_0: the number of channels
    :return:
    """
    n_slot = day_slot - n_frame + 1 # the number of sliding windows in a day

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))

    for i in range(len_seq):        # for each day
        for j in range(n_slot):     # for each sliding window
            sta = (i + offset) * day_slot + j
            end = sta + n_frame

            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(
                data_seq[sta:end, :], [n_frame, n_route, C_0]
            )

    return tmp_seq


def data_gen(file_path,
             data_config,
             n_route,
             n_frame=21,
             day_slot=288):
    """
    :param file_path: csv file path
    :param data_config: (n_train, n_val, n_test)
    :param n_route: the number of node
    :param n_frame: the number of frame in a sequence (21 = 12 + 9)
    :param day_slot: the number of time slots in a day (288 = 5 mins * 24)
    :return:
        data_dict: {
            'train': STGDataset,
            'val': STGDataset,
            'test': STGDataset,
        }

        x_stats: {
            'mean': float,
            'std': float
        }
    """
    n_train, n_val, n_test = data_config

    try:
        data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        raise FileNotFoundError(f'[ERROR] The file {file_path} does not exist')

    # make temporal sequence (train, val, test)
    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)

    x_stats = {
        'mean': np.mean(seq_train),
        'std': np.std(seq_train)
    }

    # z-score normalization
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    # make dataset
    train_dataset = STGDataset(x_train)
    val_dataset = STGDataset(x_val)
    test_dataset = STGDataset(x_test)

    data_dict = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    return data_dict, x_stats

class STGDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


if __name__ == '__main__':
    data_config = (34, 5, 5)
    data_dict, x_stats = data_gen(file_path="./data/pemsd7-m/vel.csv", data_config=data_config, n_route=228)