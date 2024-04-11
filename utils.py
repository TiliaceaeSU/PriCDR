import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from sklearn.metrics import auc
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm
from random_proj_svd import *

def mmd_loss(f_of_X, f_of_Y):

    delta = f_of_X - f_of_Y
    loss = delta.dot(delta.T)
   
    return loss

def MMD(X, Y, biased=True):
    # set params to calculate MMD distance
    sigma_list = [1e-2, 1e-1, 1, 10, 100]
    sigma_list = torch.FloatTensor(np.array(sigma_list))
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2

def evaluation(preds, topk):
    sort = np.argsort(-preds, axis=1)[:, :topk]
    hr_arr = np.zeros(shape=[sort.shape[0]])
    ndcg_arr = np.zeros(shape=[sort.shape[0]])
    mrr_arr = np.zeros(shape=[sort.shape[0]])
    rows = np.where(sort==99)[0]
    cols = np.where(sort==99)[1]
    hr_arr[rows] = 1.0
    ndcg_arr[rows] = np.log(2) / np.log(cols + 2.0)
    mrr_arr[rows] = 1.0 / (cols + 1.0)
    return hr_arr.tolist(), ndcg_arr.tolist(), mrr_arr.tolist()

def test_process(model, train_loader, feed_data, is_cuda, topK,eps,sp,pmode,et,mode='val'):

    all_hr2_list = []
    all_ndcg2_list = []
    all_mrr2_list = []
    fts1 = feed_data['fts1']
    fts2 = feed_data['fts2']

    if mode == 'val':
        book_nega = feed_data['book_nega']
        book_test = feed_data['book_vali']
    elif mode=='test':
        book_nega = feed_data['book_nega']
        book_test = feed_data['book_test']
    else:
        raise Exception


    for batch_idx, data in enumerate(train_loader):
        data = data.reshape([-1])
        val_user_arr = data.numpy()
        v_item1 = fts1[val_user_arr]
        v_item2 = fts2[val_user_arr]


        if is_cuda:
            v_user = torch.LongTensor(val_user_arr).cuda()
            v_item1 = torch.FloatTensor(v_item1).cuda()
            v_item2 = torch.FloatTensor(v_item2).cuda()
        else:
            v_user = torch.LongTensor(val_user_arr)
            v_item1 = torch.FloatTensor(v_item1)
            v_item2 = torch.FloatTensor(v_item2)
        
        
        
        res = model.forward_dmf(v_user,v_item1)  
        #res = model.forward(v_user,v_item1)  
        y2 = res[1]
        if is_cuda:
            y2 = y2.detach().cpu().numpy()
        else:
            y2 = y2.detach().numpy()

        
    
        nega_vali2 = np.array([book_nega[ele] + [book_test[ele]] for ele in val_user_arr])
        pred2 = np.stack([y2[xx][nega_vali2[xx]] for xx in range(nega_vali2.shape[0])])
        hr2_list, ndcg2_list, mrr2_list = evaluation(pred2, topK)

        all_hr2_list += hr2_list
        all_ndcg2_list += ndcg2_list
        all_mrr2_list += mrr2_list


    avg_hr2 = np.mean(all_hr2_list)
    avg_ndcg2 = np.mean(all_ndcg2_list)
    avg_mrr2 = np.mean(all_mrr2_list)

    return avg_hr2, avg_ndcg2, avg_mrr2


def dup_rows(a, indx, num_dups=1):
    return np.insert(a,[indx+1]*num_dups,a[indx],axis=0)


