import numpy as np 
import torch 
import torch.nn as nn 
import pandas as pd 
import os 
from scipy.sparse import csr_matrix 
import pdb
from dataset.Dataset import CustomizeSequentialRecDataset

class SequentialRecDataset(CustomizeSequentialRecDataset):

    def __init__(self, args, fnames, max_seq_length=50):

        self.max_seq_length = max_seq_length
        self.dataset_split = [0]
        data = pd.DataFrame()
        self.EmbeddingTable = None

        item_offset = 0
        user_offset = 0
        for fname in fnames:

            single_data = pd.read_csv(fname, sep='\t')
            self.dataset_split.append(len(single_data) + self.dataset_split[-1])
              
            single_data['user_id:token'] += user_offset
            single_data['item_id_list:token_seq'] = single_data['item_id_list:token_seq'].apply(lambda x: np.array(x.split()).astype(int))
            single_data['item_id_list:token_seq'] += item_offset
            single_data['item_id:token'] += item_offset

            item_offset = max(single_data['item_id_list:token_seq'].apply(max).max(), single_data['item_id:token'].max()) + 1
            user_offset = single_data['user_id:token'].max() + 1
              
            if data.empty:
                data = single_data
            else:
                data = pd.concat([data, single_data])

        self.df2csr(data, split=False)


class PretrainSequentialRecDataset(CustomizeSequentialRecDataset):

    def __init__(self, args, fnames, max_seq_length=50):

        self.max_seq_length = max_seq_length
        self.dataset_split = [0]
        data = pd.DataFrame()
        self.EmbeddingTable = None

        #concatenate pandas
        item_offset = 0
        user_offset = 0
        for fname in fnames:

            single_data = pd.read_csv(fname, sep='\t')
            self.dataset_split.append(len(single_data) + self.dataset_split[-1])

            single_data['user_id:token'] += user_offset
            single_data['item_id_list:token_seq'] = single_data['item_id_list:token_seq'].apply(lambda x: np.array(x.split()).astype(int))
            single_data['item_id_list:token_seq'] += item_offset
            single_data['item_id:token'] += item_offset

            item_offset = max(single_data['item_id_list:token_seq'].apply(max).max(), single_data['item_id:token'].max()) + 1
            user_offset = single_data['user_id:token'].max() + 1

            if data.empty:
                data = single_data
            else:
                data = pd.concat([data, single_data])

        self.df2csr(data, split=False)

    def __getitem__(self, idx):
        #increase the index for both interns and label
        batch_interns = self.interns[idx].toarray().squeeze()
        batch_label   = np.array(self.label[idx]+1)
        batch_length  = np.array(self.seq_length[idx])

        batch_interns, batch_label, batch_length = torch.from_numpy(batch_interns), torch.from_numpy(batch_label).squeeze(), torch.from_numpy(batch_length).squeeze()

        batch_img_emb   = self.EmbeddingTable.data2embedding["img_embed"](batch_interns)
        batch_text_emb  = self.EmbeddingTable.data2embedding["text_embed"](batch_interns)
        batch_price_emb = self.EmbeddingTable.data2embedding["price_embed"](batch_interns)

        label_img_emb = self.EmbeddingTable.data2embedding["img_embed"](batch_label)
        label_text_emb = self.EmbeddingTable.data2embedding["text_embed"](batch_label)
        label_price_emb = self.EmbeddingTable.data2embedding["price_embed"](batch_label)

        return (batch_interns, batch_img_emb, batch_text_emb, batch_price_emb), (batch_label, label_img_emb, label_text_emb, label_price_emb), batch_length

class PretrainEmbeddingTable:

    def __init__(self, args):

        self.embedding = {}
        self.data2embedding = nn.ModuleDict()
        self.num_items = 0

        for dataset in args.datasets:
            text_embed = np.fromfile(os.path.join(args.path, dataset, f"{dataset}_text_0.feat"), dtype=np.float32).reshape(-1, 768)
            img_embed  = np.fromfile(os.path.join(args.path, dataset, f"{dataset}_img_0_4.feat"), dtype=np.float64).reshape(-1, 4, args.image_dim)
            price_embed= np.fromfile(os.path.join(args.path, dataset, f"{dataset}_price_0.feat"), dtype=np.float64).reshape(-1, 64)

            item_num = len(img_embed)
            assert (len(img_embed) == item_num) and (len(price_embed) == item_num) and (len(text_embed) == item_num)
            self.num_items += item_num

            if "text_embed" not in self.embedding:
                self.embedding["text_embed"] = self.weight2emb(text_embed)
                self.embedding["img_embed"]  = self.weight2emb(img_embed)
                self.embedding["price_embed"]= self.weight2emb(price_embed)

            else:
                self.embedding["text_embed"] = np.concatenate((self.embedding["text_embed"], self.weight2emb(text_embed)), axis=0)
                self.embedding["img_embed"]  = np.concatenate((self.embedding["img_embed"], self.weight2emb(img_embed)), axis=0)
                self.embedding["price_embed"]= np.concatenate((self.embedding["price_embed"], self.weight2emb(price_embed)), axis=0)

        self.data2embedding["text_embed"] = self.np2torch(self.embedding["text_embed"], text_embed.shape[-1])
        self.data2embedding["img_embed"]  = self.np2torch(self.embedding["img_embed"], img_embed.shape[-1])
        self.data2embedding["price_embed"]= self.np2torch(self.embedding["price_embed"], price_embed.shape[-1])

    def weight2emb(self, weight):
        weight = weight.astype(np.float16)
        if len(weight.shape) == 3:
            lens = np.sum(np.sum(weight, axis=-1) != 0, axis=-1) + 1e-8
            weight = np.sum(weight, axis=1) / lens.reshape(-1,1)

        return weight

    def np2torch(self, weight, plm_dim):
        embedding = nn.Embedding(self.num_items+1, plm_dim, padding_idx=0)
        embedding.weight.requires_grad = False
        weight = np.concatenate((np.zeros((1, plm_dim)), weight), axis=0)
        embedding.weight.data.copy_(torch.from_numpy(weight))
         
        return embedding

