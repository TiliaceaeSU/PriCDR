
"""
This is a pytorch implementation of the paper: 
"Differential Private Knowledge Transfer for Privacy-Preserving Cross-Domain Recommendation."
Chaochao Chen, Huiwen Wu, Jiajie Su, Lingjuan Lyu, Xiaolin Zheng and Li Wang.
Proceedings of the ACM Web Conference 2022. 2022: 1455-1465.
"""

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import time
import random
import os
import math
import pickle
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from Dataset_PP import Dataset
from model import PriCDR
import time
import itertools
import pandas as pd
from utils import *
from scipy.sparse import csr_matrix
from random_proj_svd import *

method_name = 'my'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
topK_list = [5, 10]

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=128, help='batch size.')
parser.add_argument('--emb_size', type=int, default=100, help='embed size.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--log', type=str, default='logs/{}'.format(method_name), help='log directory')
parser.add_argument('--pos-weight', type=float, default=1.0, help='weight for positive samples')

parser.add_argument('--self', type=float, default=1.0, help='lambda rec')
parser.add_argument('--d_epoch', type=int, default=2, help='d epoch')
parser.add_argument('--t_percent', type=float, default=1.0, help='target sparsity')
parser.add_argument('--s_percent', type=float, default=1.0, help='source sparsity')
parser.add_argument('--dataset', type=str, default='sample', help='a sample dataset')

parser.add_argument('--eps', type=int, default=0, help='decide epss')
parser.add_argument('--sp', type=float, default=0.5, help='sparsity parameter for FJLT')
parser.add_argument('--mode', type=str, default='FJLT', help='JLT or FJLT')
parser.add_argument('--align', type=float, default=100.0, help='decide alignment parameter')
parser.add_argument('--et', type=float, default=0.05, help='decide latent dim')



args = parser.parse_args()

args.cuda = torch.cuda.is_available()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    """
    We adopt the evaluation method of cdr from ETL(https://github.com/xuChenSJTU/ETL-master) in this framework.
    """

    log = os.path.join(args.log, '{}_{}_{}_{}_{}_{}'.format(args.eps,args.sp,args.mode,args.align,args.et,args.t_percent,
                                                     ))
    if os.path.isdir(log):
        print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort." % log)
        time.sleep(5)
        os.system('rm -rf %s/' % log)

    os.makedirs(log)
    print("made the log directory", log)

    print('preparing data...')
    dataset = Dataset(args.batch, dataset=args.dataset)

    NUM_USER = dataset.num_user
    NUM_MOVIE = dataset.num_movie
    NUM_BOOK = dataset.num_book

    print('Preparing the training data......')
    # prepare data for X
    row, col = dataset.get_part_train_indices('movie', args.s_percent)
    values = np.ones(row.shape[0])
    user_x = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_MOVIE)).toarray()
    print(user_x.shape)
    # prepare  data fot Y
    row, col = dataset.get_part_train_indices('book', args.t_percent)
    values = np.ones(row.shape[0])
    user_y = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_BOOK)).toarray()
    print(user_y.shape)
    weight_user_item = torch.FloatTensor(user_y)
    weight_item_user = torch.FloatTensor(user_y.T)

    print('Preparing the training data over......')

    user_id = np.arange(NUM_USER).reshape([NUM_USER, 1])

    user_x = torch.FloatTensor(user_x)
    user_y = torch.FloatTensor(user_y)
    print("check shape x")
    print(user_x.shape)
    print(torch.sum(user_x))

    print("check shape y")
    print(user_y.shape)
    print(torch.sum(user_y))  

    user_x = prosvd(user_x,args.eps,args.sp,args.mode,args.et)
    user_x = user_x.to(torch.float32)

    train_loader = torch.utils.data.DataLoader(torch.from_numpy(user_id),
                                                     batch_size=args.batch,
                                                     shuffle=True)
    

    pos_weight = torch.FloatTensor([args.pos_weight])

    if args.cuda:
        pos_weight = pos_weight.cuda()

    model = PriCDR(NUM_USER=NUM_USER, NUM_MOVIE=user_x.shape[1], NUM_BOOK=user_y.shape[1],
                 EMBED_SIZE=args.emb_size, dropout=args.dropout,USER_ITEM=weight_user_item,ITEM_USER=weight_item_user,OUT_DIM=NUM_BOOK)
    
    optimizer_g = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    BCEWL = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    BCEL = torch.nn.MSELoss(reduction='none')

    if args.cuda:
        model = model.cuda()
        
        

    # prepare data fot test process
    movie_vali, movie_test, movie_nega = dataset.movie_vali, dataset.movie_test, dataset.movie_nega
    book_vali, book_test, book_nega = dataset.book_vali, dataset.book_test, dataset.book_nega
    feed_data = {}
    feed_data['fts1'] = user_x
    feed_data['fts2'] = user_y
    feed_data['movie_vali'] = movie_vali
    feed_data['book_vali'] = book_vali
    feed_data['movie_test'] = movie_test
    feed_data['book_test'] = book_test
    feed_data['movie_nega'] = movie_nega
    feed_data['book_nega'] = book_nega

    
    best_hr2, best_ndcg2, best_mrr2 = 0.0, 0.0, 0.0
    val_hr2_list, val_ndcg2_list, val_mrr2_list = [], [], []

    
    loss_list = []
    

    epoch_time_list = []
    for epoch in range(args.epochs):
        model.train()
        
        batch_loss_list = []
        

        epoch_time = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.reshape([-1])

            
            optimizer_g.zero_grad()

            if args.cuda:
                batch_user = data.cuda()
                batch_user_x = user_x[data].cuda()
                batch_user_y = user_y[data].cuda()
                
            else:
                batch_user = data
                batch_user_x = user_x[data]
                batch_user_y = user_y[data]
                
               
            time1 = time.time()
            
            pred_x,pred_y,z_x,z_y = model.forward_dmf(batch_user,batch_user_x)
            #pred_x,pred_y,z_x,z_y = model.forward(batch_user,batch_user_x)
            time2 = time.time()
            epoch_time += time2 - time1
            
            
            loss_x = BCEL(pred_x, batch_user_x).sum()
            
            loss_y = BCEWL(pred_y, batch_user_y).sum()
            #align_loss = torch.norm(map_x-z_y) + torch.norm(map_y-z_x)

            #align_loss = torch.norm(batch_user_x-user_emb)
            align_loss = torch.norm(z_x-z_y)
            #align_loss = mmd_loss(torch.mean(z_x,axis=0),torch.mean(z_y,axis=0))
            #align_loss = w_loss_a.sum() + w_loss_b.sum()
            # get the whole loss
            loss = loss_x + loss_y + args.align * align_loss 
            #loss = loss_x + loss_y
            loss.backward()
            optimizer_g.step()

            
            batch_loss_list.append(loss.item())
            
       
        epoch_loss = np.mean(batch_loss_list)
       
        loss_list.append(epoch_loss)
        

        print('epoch:{}, self loss:{:.4f}'.format(epoch,epoch_loss,))
       

        if epoch % 1 == 0:
            model.eval()

            avg_hr2, avg_ndcg2, avg_mrr2 = test_process(model, train_loader, feed_data,args.cuda, topK_list[1],args.eps,args.sp,args.mode,args.et, mode='val',)

            val_hr2_list.append(avg_hr2)
            val_ndcg2_list.append(avg_ndcg2)
            val_mrr2_list.append(avg_mrr2)

            print('test:  book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}'
                  .format( avg_hr2, avg_ndcg2, avg_mrr2))
            with open(log + '/tmp.txt', 'a') as f:
                f.write('test:  book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}\n'
                        .format( avg_hr2, avg_ndcg2, avg_mrr2))

            if avg_hr2 > best_hr2:
                torch.save(model.state_dict(), os.path.join(log, 'best_hr2.pkl'))
                best_hr2 = avg_hr2
            if avg_ndcg2 > best_ndcg2:
                torch.save(model.state_dict(), os.path.join(log, 'best_ndcg2.pkl'))
                best_ndcg2 = avg_ndcg2
            if avg_mrr2 > best_mrr2:
                torch.save(model.state_dict(), os.path.join(log, 'best_mrr2.pkl'))
                best_mrr2 = avg_mrr2

    
    print('Val process over!')
    print('Test process......')
    for topK in topK_list:
        model.load_state_dict(torch.load(os.path.join(log, 'best_hr2.pkl')))
        test_hr2, _, _ = test_process(model, train_loader, feed_data, args.cuda, topK,args.eps,args.sp,args.mode,args.et, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_ndcg2.pkl')))
        _, test_ndcg2, _ = test_process(model, train_loader, feed_data, args.cuda, topK,args.eps,args.sp,args.mode,args.et, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_mrr2.pkl')))
        _, _, test_mrr2 = test_process(model, train_loader, feed_data, args.cuda,topK,args.eps,args.sp,args.mode,args.et, mode='test')
        print('Test TopK:{} --->  book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}'
              .format(topK,  test_hr2, test_ndcg2, test_mrr2))
        with open(log + '/tmp.txt', 'a') as f:
            f.write('Test TopK:{} ---> book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}\n'
                    .format(topK, test_hr2, test_ndcg2, test_mrr2))



if __name__ == "__main__":
    print(args)
    main()
    print(args)
