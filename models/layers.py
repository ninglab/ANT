import torch
import torch.nn as nn
from recbole.model.layers import MultiHeadAttention
from models import modules
import math
import pdb

class FeedForward(nn.Module):

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
          
        return hidden_states

class TransformerLayer(nn.Module):

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        MoE,
        n_exps = 4
    ):

        super(TransformerLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(
             n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.MoE = MoE
        self.n_exps = n_exps

        if MoE:
            self.feed_forward = nn.ModuleList([FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps) for i in range(n_exps)])
            self.pooling = modules.gate({'noise':True}, hidden_size, hidden_size, n_exps)
        else:
            self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        if self.MoE:
            output = [self.feed_forward[i](attention_output) for i in range(self.n_exps)]
            output = torch.stack(output, dim=-2)
            output = self.pooling(output, attention_output)
            output = self.LayerNorm(output + attention_output)
        else:
            output = self.feed_forward(attention_output)
            output = self.LayerNorm(output + attention_output)

        return output        

class TransformerEncoder(nn.Module):

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        last_n_MoE=1,
        n_exps=4,
    ):

        super(TransformerEncoder, self).__init__()

        self.layer = nn.ModuleList()

        #MoE on the last two layers
        for i in range(n_layers):
            if i+last_n_MoE >= n_layers: 
                self.layer.append(TransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, MoE=True, n_exps=n_exps))
            else:
                self.layer.append(TransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, MoE=False, n_exps=0))

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
class NovaLayer(nn.Module):

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):

        super(NovaLayer, self).__init__()

        self.multi_head_attention = nn.MultiheadAttention(
             hidden_size, n_heads, attn_dropout_prob, batch_first=True
        )

        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.LayerNorm_key = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.LayerNorm_value = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.LayerNorm_ff = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, key, hidden_states, attention_mask):

        key = self.LayerNorm_key(key)
        hidden_states = self.LayerNorm_value(hidden_states)
        attention_output, _ = self.multi_head_attention(key, key, hidden_states, attn_mask=attention_mask)
        ##attention_output, _ = self.multi_head_attention(key, key, hidden_states)
        attention_output = attention_output + hidden_states

        attention_output = self.LayerNorm_ff(attention_output)
        output = self.feed_forward(attention_output)
        output = output + attention_output

        return output
    
class NovaEncoder(nn.Module):

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(NovaEncoder, self).__init__()

        self.layer = nn.ModuleList()

        for i in range(n_layers):
            self.layer.append(NovaLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps))

    def forward(self, key, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(key, hidden_states, attention_mask)
        return hidden_states
