import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import models.modules as modules
import pytorch_lightning as pl
from models.layers import TransformerEncoder
from models.eval_metrics import hit_at_k, ndcg_at_k
import sys
import pdb
import numpy as np

class M2RLayer(nn.Module):
    """
    projection layer between the modality space and recommendation space
    """
    def __init__(self, input_size, output_size, n_exps, mode, dropout=0.0):
        super(M2RLayer, self).__init__()

        self.n_exps = n_exps
        if mode == 'whitening':
            self.proj = nn.ModuleList([modules.Whitening(input_size, output_size, dropout) for i in range(n_exps)])
        elif mode == 'linear':
            self.proj = nn.ModuleList([modules.Linear(input_size, output_size, dropout) for i in range(n_exps)])
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        input:  (N, d)
        output: (N, n_exps, d)
        """
        output = [self.proj[i](x) for i in range(self.n_exps)]
        output = torch.stack(output, dim=-2)

        return output

class MPoolingLayer(nn.Module):
    """
    Pooling layer for single modality or combine embeddings from multi-modalities
    """

    def __init__(self, mode, config, in_dim, out_dim, n_exps):
        super(MPoolingLayer, self).__init__()

        if mode == 'gate':
            self.pooling = modules.gate(config, in_dim, out_dim, n_exps)
        elif mode == 'WFree':
            self.pooling = modules.WFree(config)
        else:
            raise NotImplementedError

    def forward(self, x, xo=None):
        return self.pooling(x, xo)

class MMFusion(nn.Module):
    """
    Fusion component for the multi-modalities
    """

    def __init__(self, mode, config):
        super(MMFusion, self).__init__()

        if mode == 'transformer':
            self.Fusion = modules.Transformer_Fusion(config)
        else:
           raise NotImplementedError

    def forward(self, text, image):
        """
        text (B, n_exps, D)
        """

        x = torch.cat((text, image), dim=-2).reshape(-1, 2, text.shape[-1])
        return self.Fusion(x)

class ANT(pl.LightningModule):
    def __init__(self, config, EmbeddingTable):
        super(ANT, self).__init__()

        pl.seed_everything(123)

        self.NAME2OPT = {'Adam':torch.optim.Adam, 'Adagrad':torch.optim.Adagrad, 'NAdam':torch.optim.NAdam}
        self.mask_exist = -1e4
        self.best_results = {"r10":0, "r50":0, "n10":0, "n50":0}
        self.temperature = config["temperature"]
        self.optim = config["optim"]
        self.max_seq_length = 50
        self.initializer_range = config["initializer_range"]
        self.dim = config["hidden_size"]
        self.fusion_lam = config["fusion_lam"]
        self.dataset_split = None
        self.training_stage = config["training_stage"]
        self.mode = config["mode"]

        self.counter = 0

        self.num_items = EmbeddingTable.num_items

        #model parameters
        self.layer_norm_eps = float(config["layer_norm_eps"])
        self.position_embedding = nn.Embedding(self.max_seq_length, self.dim)

        self.trm_encoder = TransformerEncoder(
            n_layers=config["n_layers_rec"],
            n_heads=config["n_heads_rec"],
            hidden_size=config["hidden_size"],
            inner_size=config["hidden_size"],
            hidden_dropout_prob=config["hidden_dropout_prob"],
            attn_dropout_prob=config["attn_dropout_prob"],
            hidden_act=config["hidden_act"],
            layer_norm_eps=self.layer_norm_eps,
            last_n_MoE=config["last_n_MoE"],
            n_exps=config["n_exps_model"]
         )

        if self.mode == 'NoMoE':
            self.img2proj    = nn.Linear(config["image_dim"], self.dim)
            self.text2proj   = nn.Linear(768, self.dim)
            self.meta2proj   = nn.Linear(64, self.dim)

        self.img2rec    = M2RLayer(config["image_dim"], self.dim, config["n_exps"], 'whitening', config["adaptor_dropout_prob"])
        self.text2rec   = M2RLayer(768, self.dim, config["n_exps"], 'whitening', config["adaptor_dropout_prob"])
        self.meta2rec   = M2RLayer(64, self.dim, config["n_exps"], 'linear', config["adaptor_dropout_prob"])

        self.img_pooling = MPoolingLayer(mode='gate', config=config, in_dim=config["image_dim"], out_dim=self.dim, n_exps=config["n_exps"])
        self.text_pooling= MPoolingLayer(mode='gate', config=config, in_dim=768, out_dim=self.dim, n_exps=config["n_exps"])
        self.meta_pooling= MPoolingLayer(mode='gate', config=config, in_dim=64, out_dim=self.dim, n_exps=config["n_exps"])

        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        if self.training_stage == 'finetune' and self.mode == 'transductive':
            self.item_embedding = nn.Embedding(self.num_items+1, self.dim, padding_idx=0)

        self.apply(self._init_weights)

        if self.training_stage != 'pretrain':
            self.EmbeddingTable = EmbeddingTable.data2embedding.to(self.device)
            self.indices = torch.nn.Parameter(torch.arange(self.num_items+1, dtype=torch.long), requires_grad=False)

        else:
            self.EmbeddingTable = None
            self.indices = None

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def Modality2Item(self, item_seq, split=False):

        if split:
            item_seq, item_emb_img, item_emb_text, item_emb_price = item_seq
        else:
            item_emb_price = self.EmbeddingTable["price_embed"](item_seq).squeeze() #(B, L, D)
            item_emb_img   = self.EmbeddingTable["img_embed"](item_seq).squeeze()
            item_emb_text  = self.EmbeddingTable["text_embed"](item_seq).squeeze()

        image_zero_id = (item_emb_img.sum(-1) == 0)

        if self.mode == 'NoMoE':
            img_emb = self.img2proj(item_emb_img).squeeze()
            text_emb = self.text2proj(item_emb_text).squeeze()
            price_emb = self.meta2proj(item_emb_price).squeeze()

        else:
            img_emb  = self.img2rec(item_emb_img).squeeze()
            text_emb = self.text2rec(item_emb_text).squeeze() #(B, L, n_exps, D)
            price_emb = self.meta2rec(item_emb_price).squeeze()

            img_emb, text_emb, price_emb = self.img_pooling(img_emb, item_emb_img), self.text_pooling(text_emb, item_emb_text), self.meta_pooling(price_emb, item_emb_price)

        if self.mode == 'NoText':
            item_emb = img_emb + price_emb
        elif self.mode == 'NoImage':
            item_emb = text_emb + price_emb
        elif self.mode == 'NoPrice':
            item_emb = img_emb + text_emb
        else:
            item_emb = img_emb + text_emb + price_emb

        if self.mode != 'NoFusion' and self.mode != 'NoText' and self.mode != 'NoImage':
            fusion = img_emb * text_emb
            item_emb = self.fusion_lam * fusion + item_emb

        return item_emb.squeeze(), img_emb, text_emb, image_zero_id
        
    def forward(self, item_seq, seq_lens):
        """
        item_emb_text (B, L, D)
        """

        if self.training_stage == 'pretrain':
            item_emb, _, _, _ = self.Modality2Item(item_seq, split=True)
            item_seq = item_seq[0]
        else:
            item_emb, _, _, _ = self.Modality2Item(item_seq)

        #position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.mode == 'transductive':
            input_emb = input_emb + self.item_embedding(item_seq)

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, seq_lens - 1)
         
        return output #(B, D)

    def full_seq_item_contrastive_task(self, seq_output, same_pos_id, pos_items_emb):

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        test_items_emb, _, _, _ = self.Modality2Item(self.indices.unsqueeze(1))
        if self.mode == 'transductive':
            test_items_emb = test_items_emb + self.item_embedding.weight
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        neg_logits = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.exp(neg_logits).sum(dim=1) + self.layer_norm_eps

        loss = -torch.log((pos_logits / neg_logits) + self.layer_norm_eps)
        return loss.mean()

    def batch_seq_item_contrastive_task(self, seq_output, same_pos_id, pos_items_emb):

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([self.mask_exist], dtype=neg_logits.dtype, device=same_pos_id.device), neg_logits) #mask
        neg_logits = torch.exp(neg_logits).sum(dim=1) + self.layer_norm_eps

        loss = -torch.log((pos_logits / neg_logits) + self.layer_norm_eps)
        return loss.mean()

    def batch_image_text_contrastive_task(self, image_emb, text_emb, image_zero_id):

        image_emb = F.normalize(image_emb, dim=-1)
        text_emb  = F.normalize(text_emb, dim=-1)

        pos_logits = (image_emb * text_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(image_emb, text_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.exp(neg_logits).sum(dim=1) + self.layer_norm_eps

        loss = -torch.log((pos_logits / neg_logits) + self.layer_norm_eps)
        #mask items without image
        loss[image_zero_id] = 0
        return loss.sum() / (~image_zero_id).sum()
        
    def full_sort_predict(self, interns, seq_lens):
        seq_output = self.forward(interns, seq_lens)
        test_items_emb, _, _, _ = self.Modality2Item(self.indices.unsqueeze(1))
        if self.mode == 'transductive':
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        rating_pred = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return rating_pred

    def calculate_loss(self, interns, label, seq_lens):
        """
        label (B)
        """
        if self.training_stage != 'pretrain':
            pos_item_emb, image_emb, text_emb, image_zero_id = self.Modality2Item(label.unsqueeze(1))
        else:
            pos_item_emb, image_emb, text_emb, image_zero_id = self.Modality2Item(label, split=True)
            label = label[0]

        same_pos_id = (label.unsqueeze(1) == label.unsqueeze(0)).fill_diagonal_(False)
        if self.mode == 'transductive':
            pos_item_emb += self.item_embedding(label)

        pos_item_emb = F.normalize(pos_item_emb, dim=1)
        seq_output = self.forward(interns, seq_lens)
        seq_output = F.normalize(seq_output, dim=1)

        if self.training_stage != 'pretrain':
            item_loss = self.full_seq_item_contrastive_task(seq_output, same_pos_id, pos_item_emb)
            loss = item_loss
        else:
            item_loss = self.batch_seq_item_contrastive_task(seq_output, same_pos_id, pos_item_emb)
            contrastive_loss = self.batch_image_text_contrastive_task(image_emb, text_emb, image_zero_id)
            loss = item_loss + 1e-3 * contrastive_loss

        return loss

    def configure_optimizers(self):
        optimizer = self.NAME2OPT[self.optim](self.parameters(), lr=1e-3)
        if self.training_stage != 'pretrain':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
        return [optimizer], [scheduler]

    def _shared_eval_step(self, batch, batch_idx):

        interns, label, seq_lens = batch
        rating_pred = self.full_sort_predict(interns, seq_lens)
        rating_pred[torch.arange(len(rating_pred)).unsqueeze(1), interns] = self.mask_exist
        pred_list = torch.argsort(rating_pred, dim=1)[:, -50:]

        return pred_list.detach().cpu(), label.detach().cpu()

    def _shared_eval_epoch_end(self, outputs, mode):

        results = {"r10":0, "r50":0, "n10":0, "n50":0}
        with torch.no_grad():
            for i, (pred_list, label) in enumerate(outputs):
                if i == 0:
                    total_pred_list = pred_list
                    total_label = label
                else:
                    total_pred_list = torch.cat((total_pred_list, pred_list), dim=0)
                    total_label     = torch.cat((total_label, label), dim=0)

            start = 0
            total_r10 = 0
            for dataset, end in self.dataset_split:
                label_dataset, pred_dataset = total_label[start:end], total_pred_list[start:end]
                results["r10"] = hit_at_k(label_dataset,  pred_dataset, 10, dataset=dataset, mode=mode)
                results["r50"] = hit_at_k(label_dataset,  pred_dataset, 50, dataset=dataset, mode=mode)
                results["n10"] = ndcg_at_k(label_dataset, pred_dataset, 10, dataset=dataset, mode=mode)
                results["n50"] = ndcg_at_k(label_dataset, pred_dataset, 50, dataset=dataset, mode=mode)
                start = end

                self.log(f'{dataset}_{mode}_r10', results["r10"], sync_dist=True)
                self.log(f'{dataset}_{mode}_r50', results["r50"], sync_dist=True)
                self.log(f'{dataset}_{mode}_n10', results["n10"], sync_dist=True)
                self.log(f'{dataset}_{mode}_n50', results["n50"], sync_dist=True)
                total_r10 += results["r10"].item()

            self.log(f'total_{mode}_r10', total_r10, sync_dist=True)

    def training_step(self, batch, batch_idx):
        interns, label, seq_lens = batch
        loss = self.calculate_loss(interns, label, seq_lens)
        self.log("train_loss", loss.item(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_list, label = self._shared_eval_step(batch, batch_idx)
        return pred_list, label

    def test_step(self, batch, batch_idx):
        pred_list, label = self._shared_eval_step(batch, batch_idx)
        return pred_list, label
        

    def validation_epoch_end(self, outputs):
        self._shared_eval_epoch_end(outputs, 'validation')
        return

    def test_epoch_end(self, outputs):
        self._shared_eval_epoch_end(outputs, 'test')
        return
