from .encoder import *
import torch
import numpy as np 
import pandas as pd 
import torch.nn as nn
from torch.nn import utils as nn_utils
import rouge 
from torch.distributions import Categorical
from .decoder import *
from .decoder_pointer import *
class HeadlineGen(nn.Module):
    # def __init__(self, args, encoder, decoder, load_pretrained):
    def __init__(self, args, embedding_matrix, index2word, device, pointer_gen=False, sos_id=1, eos_id=2):
        super(HeadlineGen, self).__init__()
        assert embedding_matrix is not None
        self.args = args
        self.index2word = index2word
        self.device = device
        self.embeddings = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], _weight=torch.from_numpy(embedding_matrix).float())
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pointer_gen = pointer_gen
        if args['encoder_type'] == 'lstm':
            self.encoder = LSTMEncoder(args, self.embeddings)
        else:
            self.encoder = TransformerEncoder(args, self.embeddings)
        vocab_size = len(embedding_matrix)

        if self.pointer_gen:
            self.decoder = Decoder_P(args, self.embeddings, vocab_size, device, sos_id, eos_id)
        else:
            self.decoder = Decoder(args, self.embeddings, vocab_size, device, sos_id, eos_id)

        self.loss_fn = nn.NLLLoss()
        
        self.dropout = nn.Dropout(0.2)
        
    
    def forward(self, src, tgt_input=None, tgt_output=None, user_embedding=None, teacher_forcing_ratio=0):
        
        encoder_outputs, encoder_state = self.encoder(src)
        decoder_outputs, decoder_hidden_init, _, sequence_symbols, lengths, states = self.decoder(tgt_input, tgt_output, src, encoder_state, \
                                        encoder_outputs, teacher_forcing_ratio, user_embedding) 
        
        sequences = torch.cat(sequence_symbols,dim=-1)
        wd_strs, wd_indexes = self.predict(sequences, lengths)
        
        return encoder_outputs, decoder_outputs, decoder_hidden_init, sequences, lengths, wd_strs, wd_indexes, states

        
    def predict(self, sequences, lengths):

        wd_strs, wd_indexes = [], []
        for seq,length in zip(sequences, lengths):
            wd_index = [str(ind.item()) for ind in seq][:length]
            wd_str = [self.index2word[ind.item()] for ind in seq][:length]
            wd_index = [ind for ind in wd_index if ind !=3 ]
            wd_str = [ind for ind in wd_str if ind !='<eos>' ]
            wd_indexes.append(' '.join(wd_index))
            wd_strs.append(' '.join(wd_str))

        
        return wd_strs, wd_indexes
        
    
    def batchBLLLoss(self, src, tgt_input=None, tgt_output=None, user_embedding=None, teacher_forcing_ratio=1):
        # 
        encoder_outputs, encoder_state = self.encoder(src)
        decoder_outputs, _, _, _, _, _ = self.decoder(tgt_input, tgt_output, src, encoder_state, \
                                        encoder_outputs, teacher_forcing_ratio, user_embedding)
        
        loss = 0.
        for step, step_output in enumerate(decoder_outputs):
            loss += self.loss_fn(step_output,tgt_output[:,step])
        return loss

    

