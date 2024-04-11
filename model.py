from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
from utils import *

class PriCDR(nn.Module):
    def __init__(self, NUM_USER, NUM_MOVIE, NUM_BOOK,  EMBED_SIZE, dropout, USER_ITEM, ITEM_USER, OUT_DIM, is_sparse=False):
        super(PriCDR, self).__init__()
        self.NUM_MOVIE = NUM_MOVIE
        self.NUM_BOOK = NUM_BOOK
        self.NUM_USER = NUM_USER
        self.emb_size = EMBED_SIZE
        self.OUT_DIM = OUT_DIM

        self.user_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()
        self.item_embeddings = nn.Embedding(self.NUM_BOOK, EMBED_SIZE, sparse=is_sparse)
        self.item_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_BOOK, EMBED_SIZE])).float()
        
        # self.user_embed = nn.Embedding.from_pretrained(USER_ITEM, freeze=True)
        # self.item_embed = nn.Embedding.from_pretrained(ITEM_USER, freeze=True)
        
        self.user_embed = USER_ITEM
        self.item_embed = ITEM_USER

        self.encoder_x = nn.Sequential(
            nn.Linear(self.NUM_MOVIE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
            )
        self.decoder_x = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, self.NUM_MOVIE)
            )

        #self.encoder_y = nn.Sequential(
        #    nn.Linear(self.NUM_BOOK, EMBED_SIZE),
        #    nn.ReLU(),
        #    nn.Linear(EMBED_SIZE, EMBED_SIZE)
        #    )
        #self.decoder_y = nn.Sequential(
        #    nn.Linear(EMBED_SIZE, EMBED_SIZE),
        #    nn.ReLU(),
        #    nn.Linear(EMBED_SIZE, self.NUM_BOOK)
        #    )
        
        self.nt_user = nn.Sequential(
            nn.Linear(self.NUM_BOOK, EMBED_SIZE),
            nn.Linear(EMBED_SIZE,EMBED_SIZE),
            )
        self.nt_item = nn.Sequential(
            nn.Linear(self.NUM_USER, EMBED_SIZE),
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            )
        
        
        self.orthogonal_w = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(EMBED_SIZE, EMBED_SIZE).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                              requires_grad=True)
        
        self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU

    def orthogonal_map(self, z_x, z_y):
        mapped_z_x = torch.matmul(z_x, self.orthogonal_w)
        mapped_z_y = torch.matmul(z_y, torch.transpose(self.orthogonal_w, 1, 0))
        return mapped_z_x, mapped_z_y


    def forward_sym(self, batch_user, batch_user_x, batch_user_y):
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user = self.user_embeddings(batch_user)
        feature_x = torch.add(h_user_x, h_user)
        feature_y = torch.add(h_user_y, h_user)
        z_x = F.relu(feature_x)
        z_y = F.relu(feature_y)
        preds_x = self.decoder_x(z_x)
        preds_y = self.decoder_y(z_y)
        # mapped_z_x, mapped_z_y = self.orthogonal_map(z_x, z_y)
        # preds_x2y = self.decoder_y(mapped_z_x)
        # preds_y2x = self.decoder_x(mapped_z_y)

        # define orthogonal constraint loss
        # z_x_ = torch.matmul(mapped_z_x, torch.transpose(self.orthogonal_w, 1, 0))
        # z_y_ = torch.matmul(mapped_z_y, self.orthogonal_w)
        # z_x_reg_loss = torch.norm(z_x - z_x_, p=1, dim=1)
        # z_y_reg_loss = torch.norm(z_y - z_y_, p=1, dim=1)

        return preds_x, preds_y, feature_x, feature_y


    def forward_dmf(self, batch_user, batch_user_x):

        user_y = self.user_embed[batch_user]
        allitem = torch.LongTensor(np.array(range(self.NUM_BOOK)))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            allitem =allitem.cuda()
            user_y = user_y.cuda()
        user_item = self.item_embed[allitem]
        if use_cuda:
            user_item = user_item.cuda()
        out_user_y = self.nt_user(user_y)
        out_item_y = self.nt_item(user_item)
        #print("checking shape")
        
        out_user_y = out_user_y.unsqueeze(1)
        out_item_y = out_item_y.unsqueeze(0)
        # print(feature_user_y.shape)
        # print(out_item_y.shape)
        # out_user_y = F.relu(feature_user_y)
        # out_item_y = F.relu(feature_item_y)

        # print(out_user_y)
        # print(out_item_y)

    
        norm_user_output = torch.sqrt(torch.sum(out_user_y**2, dim=2))
        norm_item_output = torch.sqrt(torch.sum(out_item_y**2, dim=2))
        preds_y = torch.sum(out_user_y*out_item_y,dim=2)/(norm_user_output*norm_item_output)
        
        #print(preds_y)

        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        z_x = F.relu(h_user_x)
        #mapped_z_x, mapped_z_y = self.orthogonal_map(z_x, user_f)
        #map_loss = torch.norm(mapped_z_x-user_f) + torch.norm(mapped_z_y-z_x)
        preds_x = self.decoder_x(z_x)

        return preds_x, preds_y, h_user_x ,out_user_y
    
    def forward(self, batch_user,batch_user_x):

        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        z_x = F.relu(h_user_x)
        #mapped_z_x, mapped_z_y = self.orthogonal_map(z_x, user_f)
        #map_loss = torch.norm(mapped_z_x-user_f) + torch.norm(mapped_z_y-z_x)
        preds_x = self.decoder_x(z_x)

        user_f = self.user_embeddings(batch_user)
        user_f = torch.div(user_f.add(h_user_x),2.0)
        allitem = torch.LongTensor(np.array(range(self.NUM_BOOK)))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            allitem =allitem.cuda()
        out_user_y = user_f.unsqueeze(1)
        
        out_item_y = self.item_embeddings(allitem)
    
        preds_y = torch.sum(out_user_y*out_item_y,dim=2)
        # print(preds_y.shape)

        # alignment part
        mapped_z_x, mapped_z_y = self.orthogonal_map(h_user_x, user_f)
        z_x_ = torch.matmul(mapped_z_x, torch.transpose(self.orthogonal_w, 1, 0))
        z_y_ = torch.matmul(mapped_z_y, self.orthogonal_w)
        z_x_reg_loss = torch.norm(h_user_x - z_x_)+torch.norm(mapped_z_x-user_f)
        z_y_reg_loss = torch.norm(user_f - z_y_)+torch.norm(mapped_z_y-h_user_x)


        return preds_x, preds_y, h_user_x, user_f
    
    
    
    def forward_single(self, batch_user, batch_user_y):
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user = self.user_embeddings(batch_user)
        feature_y = torch.add(h_user_y, h_user)
        z_y = F.relu(feature_y)
        preds_y = self.decoder_y(z_y)

        return preds_y


    def test0(self, test_user):
        user = self.user_embed(test_user)
        allitem = torch.LongTensor(np.array(range(self.NUM_BOOK)))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            allitem =allitem.cuda()
        item = self.item_embed(allitem)
        out_user_y = F.relu(self.nt_user(user))
        out_item_y = F.relu(self.nt_item(item))
        print(out_user_y.shape)
        print(out_item_y.shape)
        norm_user_output = torch.sqrt(torch.sum(out_user_y**2, dim=1))
        norm_item_output = torch.sqrt(torch.sum(out_item_y**2, dim=1))
        preds_y = torch.sum(out_user_y*out_item_y,dim=1)/(norm_user_output*norm_item_output)

        return preds_y

    def test(self, batch_user):

        user_f = self.user_embeddings(batch_user)
        allitem = torch.LongTensor(np.array(range(self.NUM_BOOK)))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            allitem =allitem.cuda()
        out_user_y = user_f.unsqueeze(1)
        
        out_item_y = self.item_embeddings(allitem)
        
        
        preds_y = torch.sum(out_user_y*out_item_y,dim=2)

        return preds_y





