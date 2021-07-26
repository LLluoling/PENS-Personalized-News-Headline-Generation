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
from .modules import Attention, Attention2

class Decoder_P(nn.Module):
    def __init__(self, args, embeddings, vocab_size, device, sos_id=1, eos_id=2):
        super(Decoder_P, self).__init__()
        self.args = args
        self.embeddings = embeddings
        self.device = device

        self.hidden_dim = args['hidden_dim'] * 2
        self.vocab_size = vocab_size
        self.max_length = 16
        self.dropout = nn.Dropout(0.2)
        self.eos_id = eos_id
        self.sos_id = sos_id

        if args['rnn_type_dec'] == 'lstm':
            self.rnn = nn.LSTM(300, self.hidden_dim, args['num_layers_dec'], batch_first=True)
        else:
            self.rnn = nn.GRU(300, self.hidden_dim, args['num_layers_dec'], batch_first=True)
        
        
        self.decoder_type = args['decoder_type']
        number_of_states = 2 if args['rnn_type_dec'] == "lstm" else 1

        if args['decoder_type'] == 1:
            self.attention = Attention(self.hidden_dim)
            self.transform = nn.ModuleList([nn.Linear(self.args['user_size'],
                                                self.hidden_dim,
                                                bias=True)
                                        for _ in range(number_of_states)])
                                        
        elif args['decoder_type'] == 2:
            self.attention = Attention2(self.hidden_dim)
            self.transform = nn.Linear(self.args['user_size'], self.hidden_dim)
        elif args['decoder_type'] == 3:
            self.attention = Attention(self.hidden_dim)
            self.transform = nn.Linear(self.args['user_size'], self.hidden_dim)
            self.transform2output = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        
        self.out = nn.Linear(self.hidden_dim, self.vocab_size)
        self.p_gen_linear = nn.Linear(self.hidden_dim, 1)

    def forward_step(self, tgt_input, hidden, encoder_outputs, user_embedding, src=None):
        # hidden is tuple or vec
        # src: (B, doc_len); [16, 500]
        bz, seq_len = tgt_input.size(0), tgt_input.size(1)
        embedded = self.embeddings(tgt_input)
        embedded = self.dropout(embedded) # B*S*hidden_dim

        output, hidden = self.rnn(embedded, hidden)
        user_temp = None

        if self.decoder_type == 1:
            output_expand = output
        elif self.decoder_type == 2:
            user_temp = self.transform(user_embedding)
            output_expand = output
        elif self.decoder_type == 3:
            user_embs = self.transform(user_embedding)
            user_embs = user_embs.unsqueeze(1).expand(output.shape[0],output.shape[1], self.hidden_dim)
            output_expand = self.transform2output(torch.cat((output,user_embs),dim=-1))

        attn = None
        if self.args['if_attention']:
            # output: (B, S, 2*hidden_dim)   attn: (batch, out_len, in_len)
            # 
            output, attn = self.attention(output_expand, encoder_outputs, user_temp)
            # output, attn = self.attention(output, encoder_outputs)

        # predicted (B, S, vocab_size)
        predicted_softmax = F.softmax(self.out(output), dim=2)
        # pointer gen
        p_gen = torch.sigmoid(self.p_gen_linear(output)) # (B, S, 1)
        vocab_dist = p_gen * predicted_softmax
        # extend src for all seqs
        doc_len = src.size()[1]
        src_extend = src.unsqueeze(1).expand(bz, seq_len, doc_len) # (B, S, doc_len)
        attn_ = (1 - p_gen) * attn
        logit = torch.log(vocab_dist.scatter_add(2, src_extend, attn_))
        return logit, hidden, attn, output

    # def forward(self, target, src, encoder_hidden=None, encoder_outputs=None, teacher_forcing_ratio=1, user_embedding=None):
    def forward(self, tgt_input=None, tgt_output=None, src=None, encoder_hidden=None, encoder_outputs=None, teacher_forcing_ratio=1, user_embedding=None):
        '''
        function: generate a sequence; 
        usage: seq2seq pretraining
        '''
        tgt_input, batch_size, max_length = self._validate_args(tgt_input, encoder_hidden, encoder_outputs,\
                                                             teacher_forcing_ratio)
        if len(user_embedding.shape)==1:
            user_embedding = user_embedding.expand(batch_size, -1)
        
        # hidden: (#layers, B, #directions * hidden_dim)  
        if self.decoder_type == 1:
            decoder_hidden_init = self._user_init_state(encoder_hidden, user_embedding)
        else:
            decoder_hidden_init = self._init_state(encoder_hidden)

        decoder_hidden = decoder_hidden_init
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)
        def decode(step, step_output, step_attn):
            '''
            1. get the index of the end of the sequence
            2. update the sequence length info
            3. return the index of the sequence end
            '''

            # step_output (B, hidden_dim) (B, vocab_size)
            
            decoder_outputs.append(step_output)
            # if self.use_attention:
            #     ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            
            # vocab index of topk words
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)
            
            eos_batches = symbols.data.eq(self.eos_id)# if the index is end of sequence
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        states = torch.zeros(batch_size, max_length, self.hidden_dim).to(self.device)
        if use_teacher_forcing:
            decoder_input = tgt_input
            decoder_output, decoder_hidden, attn, output = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, user_embedding, src)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = tgt_input[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn, output = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, user_embedding, src)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols # vocab
                states[:, di, :] = output.squeeze(1)

        return decoder_outputs, decoder_hidden_init, decoder_hidden, sequence_symbols, lengths, states

    def _init_state(self, encoder_hidden=None):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden


    def _user_init_state(self, encoder_hidden=None, user_hidden=None):
        if encoder_hidden is None or user_hidden is None:
            return None
        def bottle_hidden(linear, states):
            size = states.size()
            result = linear(states)
            return F.relu(result).unsqueeze(0)
        if isinstance(encoder_hidden, tuple):
            outs = tuple([bottle_hidden(layer, user_hidden)
                          for _, layer in enumerate(self.transform)])
        else:
            outs = bottle_hidden(self.transform[0], user_hidden)
        return outs

    
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, B, hidden_dim) -> (#layers, B, #directions * hidden_dim)
        """
        if self.args['bidirectional_enc']:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
    
    
    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if self.args['if_attention']:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.args['rnn_type_enc'] == 'lstm':
                    batch_size = encoder_hidden[0].size(1)
                else:
                    batch_size = encoder_hidden.size(1)

        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).to(self.device)
        max_length = self.max_length

        return inputs, batch_size, max_length

