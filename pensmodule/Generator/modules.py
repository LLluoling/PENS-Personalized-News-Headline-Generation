import torch
import numpy as np 
import pandas as pd 
import torch.nn as nn
import math
import random
from torch.autograd import Variable
import torch.nn.functional as F
import os
import time
from collections import deque
from torch.nn import utils as nn_utils


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context, user_embed=None):

        batch_size = output.size(0)
        hidden_size = output.size(2) #2*dim
        input_size = context.size(1)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        mix = torch.bmm(attn, context)

        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


class Attention2(nn.Module):
    def __init__(self, dim, mask=None):
        super(Attention2, self).__init__()
        self.dim = dim
        self.linear_out = nn.Linear(dim*3, dim)
        
        self.mask = mask

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context, user_embed):
        # context (bz, 500, 128)
        # output (bz, seq_len, 128)
        # user_embed (bz, 128)

        batch_size = output.size(0)
        seq_len = output.size(1)
        hidden_size = output.size(2) 
        input_size = context.size(1)
        attn1 = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn1.data.masked_fill_(self.mask, -float('inf'))
        attn1 = F.softmax(attn1.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        mix1 = torch.bmm(attn1, context) #(b, 500, 128)

        user_embed_expanded = user_embed.unsqueeze(1).expand(batch_size,seq_len, self.dim)
        attn2 = torch.bmm(user_embed_expanded, context.transpose(1, 2))
        if self.mask is not None:
            attn2.data.masked_fill_(self.mask, -float('inf'))
        attn2 = F.softmax(attn2.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        mix2 = torch.bmm(attn2, context) #(b, 500, 128)

        # user_embed (b, )

        combined = torch.cat((mix1, mix2, output), dim=2)
        output = F.tanh(self.linear_out(combined))
        # output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn1



class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True)  + 1e-8)
        
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # 300
        self.n_heads = n_heads # 20
        self.d_k = d_k # 20
        self.d_v = d_v # 20
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads) # 300, 400
        
        self._initialize_weights()
                
#         self.fc = nn.Linear(n_heads * d_v, d_model)
#         self.layer_norm = nn.LayerNorm(d_model)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                
    def forward(self, Q, K, V, attn_mask=None):
#       Q, K, V: [bz, seq_len, 300] -> W -> [bz, seq_len, 400]-> q_s: [bz, 20, seq_len, 20]
        max_len = Q.size(1)
        residual, batch_size = Q, Q.size(0)
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, max_len, max_len) #  [bz, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [bz, 20, seq_len, seq_len]
        
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask) # [bz, 20, seq_len, 20]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # [bz, seq_len, 400]
#         output = self.fc(context)
        return context #self.layer_norm(output + residual)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


'''
Value Netwrok
'''
class ValueNetwork(nn.Module):
    def __init__(self, args):
        super(ValueNetwork, self).__init__()
        self.args = args
        self.hidden_dim = args['hidden_dim'] * 2
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.fc2 = nn.Linear(self.hidden_dim//2, 1)


    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

'''
Replay Buffer
'''
class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self):
        # only sample from the half experineces with higher rewards
        rand = random.randint(0, len(self.buffer)-1)
        return self.buffer[rand]

    def clear(self):
        self.buffer.clear()
    
    def sort_buffer(self):
        self.buffer = sorted(self.buffer, key=lambda x: x[1])