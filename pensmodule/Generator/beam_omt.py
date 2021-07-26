""" Manage beam search info structure.
    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""
import torch
import numpy as np
import torch.nn.functional as F

class Beam():
    ''' Beam search '''

    def __init__(self, size, PAD_idx, SOS_idx, EOS_idx, device=False):

        self.size = size
        self.PAD_idx = PAD_idx
        self.SOS_idx = SOS_idx
        self.EOS_idx = EOS_idx
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), self.PAD_idx, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.SOS_idx

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        # word_prob: [5, 114493]
        num_words = word_prob.size(1) # 114493

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort
       
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        #prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.EOS_idx:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.SOS_idx] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k  = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


class Translator(object):
    ''' Load with trained model and handle the beam search '''
    def __init__(self, model, index2word, beam_size=5):
        
        self.model = model
        self.max_title_length = 16
        self.index2word = index2word
        self.vocab_size = len(index2word)
        self.beam_size = beam_size
        self.device = torch.device('cuda')


    def translate_batch(self, src, user_embedding):
        ''' 
        Translation work in one batch 
        user_embedding: [B, dim] 
        '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)
            
            return beamed_tensor
        
        def collect_active_hidden(decoder_hidden, inst_idx_to_position_map, active_inst_idx_list, n_bm):
            ''' Collect tensor parts associated to active instances. '''
            ht, ct = decoder_hidden
            # ht: (#layers, B*n_bm, #directions * hidden_dim)
            n_layers = ht.size()[0]
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            curr_active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)
            n_curr_active_inst = len(curr_active_inst_idx)
            n_prev_active_inst = len(inst_idx_to_position_map)
            dim = ht.size()[-1]

            ht = ht.view(n_layers, n_prev_active_inst, -1)
            ht = ht.index_select(1, curr_active_inst_idx)
            ht = ht.view(n_layers, n_curr_active_inst*n_bm, dim)

            ct = ct.view(n_layers, n_prev_active_inst, -1)
            ct = ct.index_select(1, curr_active_inst_idx)
            ct = ct.view(n_layers, n_curr_active_inst*n_bm, dim)
            
            decoder_hidden = (ht, ct)
            return decoder_hidden

        def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output, src_seq, decoder_hidden, inst_idx_to_position_map, n_bm, user_embedding, n_layer, n_inst):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq, user_embedding, n_inst, n_active_inst, decoder_hidden, n_layer):
                #dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = []
                user_emb_lst = []
                ht_lst = []
                ct_lst = []
                user_dim = user_embedding.size()[-1]
                ht, ct = decoder_hidden
                hid_dim = ht.size()[-1]
                user_embedding = user_embedding.view(n_inst, n_bm, user_dim)
                #ht = ht.view(n_layer, n_inst, n_bm, hid_dim)
                #ct = ct.view(n_layer, n_inst, n_bm, hid_dim)
                # decoder_hidden: (ht, ct); ht: (#layers, B*n_bm, #directions * hidden_dim)
                for i in range(len(inst_dec_beams)):
                    b = inst_dec_beams[i]
                    if not b.done:
                        dec_partial_seq.append(b.get_current_state())
                        user_emb_lst.append(user_embedding[i])
                        #ht_lst.append(ht[:,i,:,:])
                        #ct_lst.append(ct[:,i,:,:])

                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                user_emb = torch.stack(user_emb_lst).view(-1, user_dim)
                #ht = torch.stack(ht_lst, dim=1).view(n_layer, n_active_inst*n_bm, hid_dim) # [2, 80, 400]
                #ct = torch.stack(ct_lst, dim=1).view(n_layer, n_active_inst*n_bm, hid_dim)
                decoder_hidden = (ht, ct)
                return dec_partial_seq, user_emb, decoder_hidden
   
            def predict_word(dec_seq, src_seq, enc_output, n_active_inst, n_bm, user_embedding, decoder_hidden):
                
                #dec_output, attn_dist = self.model.decoder(self.model.embedding(dec_seq), enc_output, (mask_src,mask_trg))
                dec_input = dec_seq[:,-1].unsqueeze(1)
                decoder_output, decoder_hidden, step_attn, _ = self.model.decoder.forward_step(dec_input, decoder_hidden, enc_output, user_embedding, src_seq)
                step_output = decoder_output.squeeze(1) # [B*n_bm, vocab_size]
                word_prob = step_output.view(n_active_inst, n_bm, -1) # [B, n_bm, vocab_size]
                return word_prob, decoder_hidden

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq, user_embedding, decoder_hidden = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq, user_embedding, n_inst, n_active_inst, decoder_hidden, n_layer) # [B * n_bm, len_dec_seq] B为还未结束的batch数量
            word_prob, decoder_hidden = predict_word(dec_seq, src_seq, enc_output, n_active_inst, n_bm, user_embedding, decoder_hidden)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list, decoder_hidden

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #-- Encode
            # src: [B, S] 
            # src_enc: [B, S, n_direction * dim] (16, 500, 400) bidirectional 2*200
            # src_state:(ht, ct): [# layers* # directions, B, dim] (4, 16, 200)
            src_enc, src_state = self.model.encoder(src)
            
            decoder_hidden = self.model.decoder._init_state(src_state) # (#directions*#layers, B, hidden_dim) -> (#layers, B, #directions * hidden_dim)
            ht, ct = decoder_hidden # ht,ct: [2, 16, 400]
            n_layer = ht.size()[0]
            n_direction = src_state[0].size()[0]/n_layer
           
            #-- Repeat data for beam search
            n_bm = self.beam_size
            n_inst, len_s, d_h = src_enc.size()
           
            src_seq = src.repeat(1, n_bm).view(n_inst * n_bm, len_s) # [B, S] --> [B * n_bm, S]
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h) # [B * n_bm, S, dim]
            d_user = user_embedding.size()[1]
            user_embedding = user_embedding.repeat(1, n_bm).view(n_inst * n_bm, d_user) # [B, dim] --> [B * n_bm, dim]
            ht = ht.repeat(1, 1, n_bm).view(n_layer, n_inst*n_bm, d_h)
            ct = ct.repeat(1, 1, n_bm).view(n_layer, n_inst*n_bm, d_h)
            decoder_hidden = (ht, ct) # ht,ct: (#layers, B*n_bm, #directions * hidden_dim)

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, PAD_idx=0, SOS_idx=1,EOS_idx=2, device=self.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst)) # B
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, self.max_title_length + 1):
                active_inst_idx_list, decoder_hidden = beam_decode_step(inst_dec_beams, len_dec_seq, src_enc, src_seq, decoder_hidden, inst_idx_to_position_map, n_bm, user_embedding, n_layer, n_inst)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, new_inst_idx_to_position_map = collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)
                decoder_hidden = collect_active_hidden(decoder_hidden, inst_idx_to_position_map, active_inst_idx_list, n_bm)
                inst_idx_to_position_map = new_inst_idx_to_position_map

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

        return batch_hyp, batch_scores
