import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import pdb

class UniEmbeddingTable():

    def __init__(self, args):

        self.data2embedding = None
        self.num_items = 0

        text_embed = np.fromfile(os.path.join(args.path, args.dataset, f"{args.dataset}.feat1CLS"), dtype=np.float32).reshape(-1, 768)
        self.num_items = len(text_embed)

        self.data2embedding = self.weight2emb(text_embed, text_embed.shape[-1])

    def weight2emb(self, weight, plm_dim):

        embedding = nn.Embedding(self.num_items+1, plm_dim, padding_idx=0)
        embedding.weight.requires_grad = False
        weight = np.concatenate((np.zeros((1, plm_dim)), weight), axis=0)
        embedding.weight.data.copy_(torch.from_numpy(weight))

        return embedding
        
class EmbeddingTable():

    def __init__(self, args):

        self.data2embedding = {}
        self.num_items = {}
        for dataset in args.datasets:
            self.data2embedding[dataset] = nn.ModuleDict()
            text_embed = np.fromfile(os.path.join(args.path, dataset, f"{dataset}_text_0.feat"), dtype=np.float32).reshape(-1, 768)
            img_embed  = np.fromfile(os.path.join(args.path, dataset, f"{dataset}_img_0_4.feat"), dtype=np.float64).reshape(-1, 4, args.image_dim)
            price_embed= np.fromfile(os.path.join(args.path, dataset, f"{dataset}_price_0.feat"), dtype=np.float64).reshape(-1, 64)

            item_num = len(img_embed)
            assert (len(img_embed) == item_num) and (len(price_embed) == item_num) and (len(cate_embed) == item_num)
            self.num_items[dataset]= item_num

            self.data2embedding[dataset]["text_embed"] = self.weight2emb(text_embed, text_embed.shape[-1], item_num)
            self.data2embedding[dataset]["img_embed"]  = self.weight2emb(img_embed, img_embed.shape[-1], item_num)
            self.data2embedding[dataset]["price_embed"]= self.weight2emb(price_embed, price_embed.shape[-1], item_num)

    def weight2emb(self, weight, plm_dim, item_num):
        if len(weight.shape) == 3:
            lens = np.sum(np.sum(weight, axis=-1) != 0, axis=-1) + 1e-8
            weight = np.sum(weight, axis=1) / lens.reshape(-1,1)

        embedding = nn.Embedding(item_num+1, plm_dim, padding_idx=0)
        embedding.weight.requires_grad = False
        #zero padding
        weight = np.concatenate((np.zeros((1, plm_dim)), weight), axis=0)
        embedding.weight.data.copy_(torch.from_numpy(weight))

        return embedding

class CustomizeSequentialRecDataset(Dataset):

    #follow that in UniSRec, max_seq_length is 50
    def __init__(self, args, fname, max_seq_length=50):

        self.max_seq_length = max_seq_length
        data = pd.read_csv(fname, sep='\t')
        self.df2csr(data)

    def df2csr(self, data, split=True):
        interns = data["item_id_list:token_seq"]
        interns = interns.values.tolist()
        padded_interns = self.padding(interns, split=split)
        self.interns = csr_matrix(padded_interns)

        self.label   = data['item_id:token'].values.tolist()
        self.label   = [int(label) for label in self.label]
        self.label   = np.asarray(self.label)
        return

    def padding(self, interns, split=True):

        self.seq_length = []
        #item index start from zero, increase to 1
        for i, x in enumerate(interns):
            if split:
                x = list(map(int, x.split()))
            else:
                x = x.tolist()
            x = [e + 1 for e in x]
            num_pads = self.max_seq_length - len(x)
            interns[i] = x + [0] * num_pads

            self.seq_length.append(len(x))

        self.seq_length = np.asarray(self.seq_length)
        return np.array(interns)

    def __len__(self):
        return self.interns.shape[0]

    def __getitem__(self, idx):

        #increase the index for both interns and label
        batch_interns = self.interns[idx].toarray().squeeze()
        batch_label   = np.array(self.label[idx]+1)
        batch_length  = np.array(self.seq_length[idx])

        return torch.from_numpy(batch_interns), torch.from_numpy(batch_label).squeeze(), torch.from_numpy(batch_length).squeeze()
        
