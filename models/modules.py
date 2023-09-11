import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
import pdb

"""
Modules used in the M2R layer
"""

class Whitening(nn.Module):
    """
    Whitening for projection
    """
    def __init__(self, input_size, output_size, dropout=0.0):
         super(Whitening, self).__init__()

         self.dropout = nn.Dropout(p=dropout)
         self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
         self.lin  = nn.Linear(input_size, output_size, bias=False)
         self.apply(self._init_weights)

    def _init_weights(self, module):
         if isinstance(module, nn.Linear):
             torch.nn.init.xavier_normal_(module.weight.data)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)

class Linear(nn.Module):
    """
    a wrapper for nn.Linear
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(Linear, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lin  = nn.Linear(input_size, output_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight.data)

    def forward(self, x):
        return self.lin(self.dropout(x))

"""
Modules used in the Modality Pooling layer
"""

proj2mol = {'Whitening' : Whitening, 'Linear' : Linear}
pool2mol = {'Mean' : torch.mean, 'Max' : torch.max}

class gate(nn.Module):
    """
    mixture of expert which is different from that in UniSRec since we use x_hat instead of x to calculate the gate
    """
    def __init__(self, config, in_dim, out_dim, n_exps):
        super(gate, self).__init__()

        self.noisy_gating = config['noise']

        self.input_size  = in_dim
        self.output_size = out_dim
        self.n_exps = n_exps

        #FIXME why zero initialization? why not use Linear?
        self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.n_exps), requires_grad=True)

    #FIXME assume the weight follows a gaussian distribution
    def gating(self, x, train, noise_eps=1e-2):
        mean = x @ self.w_gate
        if self.noisy_gating and train:
            std = F.softplus(x @ self.w_noise) + noise_eps
            #reparameterization trick
            logits = mean + (torch.mul(torch.randn_like(mean).to(x.device), std))
        else:
            #only use the mean for test
            logits = mean

        return F.softmax(logits, dim=-1)

    def forward(self, x, xo):
        #FIXME generate weights using x before the MoE as in UniSRec
        gates = self.gating(xo, self.training)
        gated_outputs  = torch.mul(gates.unsqueeze(-1), x)
        return gated_outputs.sum(-2)

class WFree(nn.Module):
    """
    A wrapper for the mean over multi experts
    """
    def __init__(self, config):
         super(WFree, self).__init__()

         pooling = pool2mol['Mean']

    def forward(self, x, xo=None):
        return pooling(x, dim=1)

"""
Modules for multi-modality fusion
"""

class SFusion(nn.Module):
    """
    sum operator for the fusion.
    Using this operator is equivalent to the original transformer
    """

    def __init__(self, attn_dropout):
        super(SFusion, self).__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, value, attentions, scores):
        scores = self.attn_dropout(scores)
        value  = torch.matmul(scores, value)
        value  = value.permute(0, 2, 1, 3).contiguous() #(B, L, H, D)
        B, L, H, D  = value.size()
        return value.view(B, L, H*D)

class DeepSet_layer(nn.Module):
    """
    DeepSet layer in https://arxiv.org/pdf/1703.06114.pdf
    """

    def __init__(self, in_dim, out_dim, pooling, lam=True, num_modality=3):
        super(DeepSet_layer, self).__init__()

        self.Gamma  = nn.Linear(in_dim, out_dim)
        self.lam = lam
        if self.lam:
            self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        self.pooling = pooling
        self.num_modality = num_modality

    def forward(self, x):

        if self.pooling == 'Mean':
           xm = x.mean(dim=-3, keepdim=True)
        elif self.pooling == 'Max':
           xm = x.max(dim=-3, keepdim=True)
        elif self.pooling == 'cross':
           xm1, xm2, xm3 = torch.chunk(x, self.num_modality, dim=-3) #(B, H, M, E, D)
           xm = torch.mul(torch.mul(value_m1, value_m2), value_m3).unsqueeze(-3) #(B, H, M, 1, E, D)

        if self.lam:
            return self.Gamma(x) - self.Lambda(xm)
        else:
            return self.Gamma(x-xm)
            

class DFusion(nn.Module):

    def __init__(self, attn_dropout, layers, dim, pooling, lam=True):
        super(DFusion, self).__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        sequantial = []

        for i in range(layers):
            sequential.append(DeepSet_layer(dim, dim, pooling, lam))
            if i != (layers - 1):
                #FIXME layernorm may not be neccessary
                sequential.append(nn.LayerNorm(dim))
                sequential.append(nn.ELU(inplace=True))

        self.DeepSet == nn.Sequential(*sequantial)

    def forward(self, value, attentions, scores):
        #value (B, H, L, D)
        #attentions (B, H, L, L)
        #scores (B, H, L, L)
        #FIXME the function is super complicated. Need to double check

        B, H, L, L = attentions.size()
        n_exps = int(L/self.num_modality)

        #split the attention matrix to multiple E x E sub matrices of certain modalities
        attentions_chunk = torch.cat(attentions.chunk(self.num_modality, dim=-1)).view(B, H, self.num_modality, self.num_modality, n_exps, n_exps)
        attentions_chunk = attentions_chunk.transpose(1,0) #(B, H, M, m, E, E)
        subscores = F.softmax(attentions_chunk, dim=-1)
        subscores = self.attn_dropout(subscores)

        #FIXME donot model the importance of modalities during fusion
        ##modality_scores = scores.view(B, H, L, self.num_modality, n_exps).sum(-1, keepdim=True) #(B, H, L, m, 1)
        ##modality_scores = modality_scores.view(B, H, self.num_modality, n_exps, self.num_modality, 1).tranpose(-2, -3)
        #(B, H, M, m, E, 1)

        #split the value matrix to M x E x D modality specific matrices
        value = torch.stack(value.chunk(self.num_modality, dim=-2), 2) #(B, H, m, E, D)
        value = value.unsqueeze(2) #(B, H, 1, m, E, D)
        value = torch.matmul(subscores, value) #(B, H, M, m, E, D)

        #FIXME donot model the importance of modalities during fusio
        #score on each modality decided by the score map
        ##value_s = torch.mul(modality_scores, value) #(B, H, M, m, E, D)
        ##value_s = value.sum(dim=-3, keepdim=True)

        value = self.DeepSet(value)

        value = value.view(B, H, L, -1)
        value = value.permute(0, 2, 1, 3).contiguous() #(B, L, H, D)
        return value.view(B, H, -1)

class FMHA(nn.Module):
    """
    Improve the flexibility of the original MHA
    The implementation is adapted from that in s3:
    https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/modules.py
    """
    def __init__(self, config):
         super(FMHA, self).__init__()
         assert config["hidden_size"] % config["n_heads_fusion"] == 0
        
         self.num_heads = config["n_heads_fusion"]
         self.attn_head_size = int(config["hidden_size"] / self.num_heads)

         self.query = nn.Linear(config["hidden_size"], config["hidden_size"])
         self.key   = nn.Linear(config["hidden_size"], config["hidden_size"])
         self.value = nn.Linear(config["hidden_size"], config["hidden_size"])

         self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])

         if config["tf_fusion"] == 'SFusion':
             self.Fusion = SFusion(config["attn_dropout_prob"])
         elif config["tf_fusion"] == 'Dfusion':
             self.Fusion = DFusion(config["attn_dropout_prob"], config["layers_ds"], self.attn_head_size, 
                 pooling=config["pooling_ds"])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    #we may not need attention mask in Fusion
    #FIXME no layernorm and input or output dropout inside the multi-head attention
    def forward(self, x, mask=None):
        query = self.query(x)
        key   = self.key(x)
        value = self.value(x)

        query = self.transpose_for_scores(query) #(B, H, L, D)
        key   = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attentions = torch.matmul(query, key.transpose(-1, -2)) #(B, H, L, L)
        attentions = attentions / math.sqrt(self.attn_head_size)

        scores = F.softmax(attentions, dim=-1)

        value = self.Fusion(value, attentions, scores)
        return self.dense(value)

class PointWiseFeedForward(nn.Module):
    """
    Same with that in SASRec
    """
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.dropout1(self.gelu(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
 
class Transformer_Fusion(nn.Module):
    """
    Multi-modality fusion with transformer like network
    """

    def __init__(self, config):
        super(Transformer_Fusion, self).__init__()

        self.input_dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.layers_fusion = config["n_layers_fusion"]

        #we do not need last layer norm since it's just for the Fusion

        for _ in range(self.layers_fusion):
            new_attn_layernorm = nn.LayerNorm(config["hidden_size"], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FMHA(config)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(config["hidden_size"], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config["hidden_size"], config["hidden_dropout_prob"])
            self.forward_layers.append(new_fwd_layer)

    def forward(self, x):
        """
        x (B, L, H*D)
        """

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](x)
            mha_outputs = self.attention_layers[i](Q)
            
            #residule
            x = Q + mha_outputs
            x = self.forward_layernorms[i](x)
            #residule is in the fwa
            x = self.forward_layers[i](x)

        return x
