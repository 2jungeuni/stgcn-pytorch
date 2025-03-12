# built-in
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, norm

# ml
import torch

def z_score(x, mean, std):
    return (x - mean) / std

def z_inverse(y, mean, std):
    return y * std + mean

def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return torch.mean(torch.abs(v_ - v) / (v + 1e-5))


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return torch.mean(torch.abs(v_ - v))

def scaled_laplacian(W):
    # d -> diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)

    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])

    # lambda_max \approx 2.0, the largest eigenvalues of l.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))

        return np.concatenate(L_list, axis=1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)

    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id

    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')

    # if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
    #         or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
    #     row_sum = adj.sum(axis=1).A1
    #     row_sum_inv_sqrt = np.power(row_sum, -0.5)
    #     row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
    #     deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
    #     # A_{sym} = D^{-0.5} * A * D^{-0.5}
    #     sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)
    #
    #     if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
    #         sym_norm_lap = id - sym_norm_adj
    #         gso = sym_norm_lap
    #     else:
    #         gso = sym_norm_adj
    #
    # elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
    #         or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
    #     row_sum = np.sum(adj, axis=1).A1
    #     row_sum_inv = np.power(row_sum, -1)
    #     row_sum_inv[np.isinf(row_sum_inv)] = 0.
    #     deg_inv = np.diag(row_sum_inv)
    #     # A_{rw} = D^{-1} * A
    #     rw_norm_adj = deg_inv.dot(adj)
    #
    #     if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
    #         rw_norm_lap = id - rw_norm_adj
    #         gso = rw_norm_lap
    #     else:
    #         gso = rw_norm_adj
    #
    # else:
    #     raise ValueError(f'{gso_type} is not defined.')
    #
    # return gso