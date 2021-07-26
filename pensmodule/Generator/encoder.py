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
from .modules import MultiHeadAttention, PositionalEmbedding

class LSTMEncoder(nn.Module):
    def __init__(self, args, embeddings):
        super(LSTMEncoder, self).__init__()
        self.args = args 
        self.embeddings = embeddings
        if args['rnn_type_enc'] == 'lstm':
            self.rnn = nn.LSTM(300, args['hidden_dim'], args['num_layers_enc'],
                 batch_first=True, dropout=0.2, bidirectional=True)
        else:
            self.rnn = nn.GRU(300, args['hidden_dim'], args['num_layers_enc'],
                 batch_first=True, dropout=0.2, bidirectional=True)

        if args['use_bridge']:
            self.total_hidden_dim = args['hidden_dim']*args['num_layers_enc']
            self._initialize_bridge(args['rnn_type_enc'])

    def forward(self,src):

        # 
        emb = self.embeddings(src)
        # memory_bank => output         (B, S, hidden_dim * #directions)
        # encoder_final => (hn, cn)     (#directions * #layers, B, hidden_dim)
        memory_bank, encoder_final = self.rnn(emb)
        if self.args['use_bridge']:
            encoder_final = self._bridge(encoder_final)
        return memory_bank, encoder_final


    def _initialize_bridge(self, rnn_type):
        number_of_states = 2 if rnn_type == "lstm" else 1
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    
    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)
        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout



class TransformerEncoder(nn.Module):
    def __init__(self, args, embedding_matrix):
        super(TransformerEncoder, self).__init__()
        assert embedding_matrix is not None
        self.args = args 
        self.embeddings = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], _weight=embedding_matrix.float())
        self.pos_encoder = PositionalEmbedding(300)
        self.attn_body = MultiHeadAttention(300, 20, 20, 20)
        self.drop_layer = nn.Dropout(p=0.2)

    def forward(self,src):

        body = self.drop_layer(self.embeddings(src))
        body = body + self.pos_encoder(body)
        memory_bank = self.drop_layer(self.attn_body(body, body, body))
        encoder_final = None
        return memory_bank, encoder_final

