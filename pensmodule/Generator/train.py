import os
import torch
from torch.autograd import Variable
import numpy as np 
import pandas as pd 
import torch.nn as nn
import torch.optim as optim
import rouge 
from collections import OrderedDict
from torch.distributions import Categorical
from .model import HeadlineGen
import torch.nn.functional as F
from tqdm import tqdm
import torch
import re
import pickle
import math
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tensorboardX import SummaryWriter
from .modules import ValueNetwork, Memory
from .util import ROUGE_Score, ROUGE_Score_TOP3, discount_reward_update
from copy import deepcopy

class Trainer(nn.Module):

    def __init__(self, args, model, usermodel, device, mode=1, experiment_name=None):
        super(Trainer, self).__init__()
        self.model_args = args['model']
        self.train_args = args['train']
        self.model = model
        self.mode = mode
        self.usermodel = usermodel
        self.usermodel.eval()
        self.device = device

        self.experiment_path = os.path.join('../../runs/seq2seq', experiment_name)
        self.writer = SummaryWriter(
            self.experiment_path, comment=experiment_name)
        
        if self.train_args['warm_start']:
            assert self.train_args['tag'] is not None
            self.load_checkpoint(self.train_args['tag'])
    

    def _init_optimizer(self, Optimizer=torch.optim.Adam,
                Scheduler=torch.optim.lr_scheduler.StepLR,
                optimizer_params={}, scheduler_params={'step_size': 1000, 'gamma': 0.98}):
        # optimizer_params = self.train_args.get('optimizer_params', {})
        # scheduler_params = self.train_args.get('scheduler_params', )
        self.opt = Optimizer(list(self.model.parameters()),
                             **optimizer_params)
        self.scheduler = Scheduler(self.opt, **scheduler_params)
        self.step = 0

    
    def _init_evaluator_(self):
        self.lm_criterion = nn.CrossEntropyLoss()
        self.lm = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.lm.eval()

        self.lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        
        self.rouge_evaluator = rouge.Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])


    def _init_context_(self, body_path='../../data2/news_body.npy', vert_path='../../data2/news_vert.npy'):
        print('loading news body')
        # notice: news body is a string of word indexes
        self.news_body = torch.from_numpy(np.load(body_path)).to(self.device)
        self.news_vert = torch.from_numpy(np.load(vert_path)).to(self.device)

        news_body = self.news_body.tolist()
        news_body = [ i[:i.index(0)] if 0 in i else i for i in news_body ]
        self.news_body_indexes = np.array([ ' '.join(map(str, i)) for i in news_body])
        with open('../../data2/dict.pkl', 'rb') as f:
            _,_,word_dict = pickle.load(f)
        regex = ' {} | {} '.format(word_dict[','], word_dict['.'])
        self.news_body_splits = [re.split(regex, x) for x in self.news_body_indexes]

    
    def save_checkpoint(self, tag=None, path=None, mkdir=False, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            tag = "temp_{}".format(self.step)
        if path is None:
            path = os.path.join(self.experiment_path,
                                "checkpoint_{}.pth".format(tag))
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(OrderedDict([
            ('model', self.model),
            ('step', self.step)
        ]), path)
        print("Saved " + path)
        return path

    
    def load_checkpoint(self, tag=None, path=None, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is not None and path is None:
            path = os.path.join(self.experiment_path,
                                "checkpoint_{}.pth".format(tag))
        checkpoint = torch.load(path)

        self.model = checkpoint['model']
        self.step = int(checkpoint['step'])

        print('Loaded ' + path)


    def pretrain(self, train_iter, global_user_embed, max_step=5000):   
        self.model.train()
        pbar = tqdm(train_iter)        
        losses = []
        for i, batch in enumerate(pbar):
            src, tgt_input, tgt_output = batch[:3]
            src = torch.as_tensor(src, device=self.device).long()
            tgt_input = torch.as_tensor(tgt_input, device=self.device).long()
            tgt_output = torch.as_tensor(tgt_output, device=self.device).long()

            loss = self.model.batchBLLLoss(src, tgt_input, tgt_output, global_user_embed,
                                            self.train_args['teacher_forcing_ratio']).mean()
            loss.backward()
            if (i+1) % self.train_args['update_frequency'] == 0:
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()
            self.step += 1
            
            loss_item = loss.detach().cpu().numpy()
            losses.append(loss_item)

            self.writer.add_scalar('pretrain loss', loss_item, self.step)
            pbar.set_description("pretrain loss: {:.3f}".format(np.mean(losses[-200:])))
            if (i+1) % self.train_args['report_frequency'] == 0:
                print("step={:4d}, pretrain loss: {:.3f}".format(i+1, np.mean(losses)))

            if i == max_step:
                break

    
    def train(self, train_iter, train_option='a2c', tag='a2c', max_train_step=3000):
        self.model.train()
        pbar = tqdm(train_iter)        
        losses, losses2, rewards = [], [], []
        self.step = 1
        if train_option == 'monte':
            for i, batch in enumerate(pbar):
                news_ids, clicked_rep, src, tgt_input, tgt_output = batch
                bz = clicked_rep.shape[0]
                
                clicked_rep = torch.as_tensor(clicked_rep, device=self.device)
                src = torch.as_tensor(src, device=self.device).long()
                tgt_input = torch.as_tensor(tgt_input, device=self.device).long()
                tgt_output = torch.as_tensor(tgt_output, device=self.device).long()

                with torch.no_grad():
                    user_embeds = self.usermodel.user_encoder(clicked_rep)
                news_body_indexes = self.news_body_indexes[news_ids]
                news_body = self.news_body[news_ids]
                news_vert = self.news_vert[news_ids]
                encoder_outputs, decoder_outputs, decoder_hidden_init, sequences, lengths, wd_strs, wd_indexes, wd_str_lists, wd_index_lists, states = \
                                        self.model(src, tgt_input, tgt_output, user_embeds, 0)


                reward = self.get_value(encoder_outputs, decoder_hidden_init, sequences, lengths,\
                                        user_embeds, news_body_indexes, news_body, news_vert, wd_strs, wd_indexes, src)
                rewards.append(np.mean([r[lenth-1] for r,lenth in zip(reward,lengths)]))
                # import pdb; pdb.set_trace()
                loss = torch.zeros(bz).to(self.device)
                length_torch = torch.from_numpy(lengths).float().to(self.device)
                # 
                for step, step_output in enumerate(decoder_outputs):
                    for (b, length) in zip(range(bz),lengths): 
                        if step<length:
                            loss[b] += -step_output[b][sequences[b][step]]*reward[b][step] 
                # loss = loss/length_torch
                loss = torch.mean(loss)
                loss.backward()
                if (i+1) % self.train_args['update_frequency'] == 0:
                    self.opt.step()
                    self.opt.zero_grad()
                    self.scheduler.step()
                
                self.step += 1

                if (i+1) % self.train_args['save_frequency'] == 0:
                    self.save_checkpoint(tag='train_step_'+str(i+1))
                
                loss_item = loss.detach().cpu().numpy()
                losses.append(loss_item)

                self.writer.add_scalar('train loss', loss_item, self.step)
                pbar.set_description("train loss: {:.3f}, train reward: {:.3f}".format(np.mean(losses[-10:]),np.mean(rewards[-10:])))
                if (i+1) % self.train_args['report_frequency'] == 0:
                    print("step={:4d}, train loss: {:.3f}, train reward: {:.3f}".format(i+1, np.mean(losses[-100:]), np.mean(rewards[-100:])))

                if i == max_train_step:
                    break

            return rewards 

        elif train_option == 'a2c':
            value_net = ValueNetwork(self.model_args).to(self.device)
            value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)
            value_criterion = nn.MSELoss()
            for i, batch in enumerate(pbar):
                news_ids, clicked_rep, src, tgt_input, tgt_output = batch
                bz = clicked_rep.shape[0]
                
                clicked_rep = torch.as_tensor(clicked_rep, device=self.device)
                src = torch.as_tensor(src, device=self.device).long()
                tgt_input = torch.as_tensor(tgt_input, device=self.device).long()
                tgt_output = torch.as_tensor(tgt_output, device=self.device).long()

                with torch.no_grad():
                    user_embeds = self.usermodel.user_encoder(clicked_rep)
                news_body_indexes = self.news_body_indexes[news_ids]
                news_body_split = [self.news_body_splits[i] for i in news_ids]
                news_body = self.news_body[news_ids]
                news_vert = self.news_vert[news_ids]
                encoder_outputs, decoder_outputs, decoder_hidden_init, sequences, lengths, wd_strs, wd_indexes, wd_str_lists, wd_index_lists, states = \
                                        self.model(src, tgt_input, tgt_output, user_embeds, 0)

                all_rewards = self.get_reward(sequences, lengths, wd_strs, wd_indexes, wd_str_lists, wd_index_lists, news_body, news_vert, news_body_indexes, news_body_split, user_embeds) 
                r = np.mean(np.sum(all_rewards, 1))
                rewards.append(r)
                self.writer.add_scalar('reward', r, self.step)

                vs = value_net(states.detach()).squeeze(-1)

                qs = torch.Tensor(discount_reward_update(all_rewards, lengths)).to(self.device) #bz, seq_len, 1
                advantages = qs - vs.detach()

                self.model.zero_grad()
                actor_network_loss = torch.zeros(bz).to(self.device)
                for step, step_output in enumerate(decoder_outputs):
                    for b in range(bz):
                        if step < lengths[b]:
                            actor_network_loss[b] += -step_output[b][sequences[b][step]]*advantages[b][step] 
                actor_network_loss = actor_network_loss.mean()
                actor_network_loss.backward()

                self.opt.step()

                loss_item1 = actor_network_loss.detach().cpu().numpy()
                losses.append(loss_item1)
                self.writer.add_scalar('actor loss', loss_item1, self.step)

                # train value network
                value_optimizer.zero_grad()
                value_network_loss = torch.zeros(bz).to(self.device)
                for b in range(bz):
                    value_network_loss[b] = value_criterion(vs[b, :lengths[b]], qs[b, :lengths[b]])
                value_network_loss = value_network_loss.mean() 
                value_network_loss.backward()
                # torch.nn.utils.clip_grad_norm(value_net.parameters(),0.5)
                value_optimizer.step()

                if (i+1) % self.train_args['save_frequency'] == 0:
                    self.save_checkpoint(tag='train_'+tag+'_step_'+str(i+1))
                
                loss_item2 = value_network_loss.detach().cpu().numpy()
                losses2.append(loss_item2)
                self.writer.add_scalar('criticc loss', loss_item2, self.step)
                
                pbar.set_description("train actor loss: {:.3f}, train critic loss: {:.3f}, train reward: {:.3f}".format(np.mean(losses[-10:]),np.mean(losses2[-10:]),np.mean(rewards[-10:])))
                if (i+1) % self.train_args['report_frequency'] == 0:
                    print("step={:4d}, train actor loss: {:.3f}, train critic loss: {:.3f}, train reward: {:.3f}".format(i+1, np.mean(losses[-100:]), np.mean(losses2[-10:]), np.mean(rewards[-100:])))

                if i == max_train_step:
                    break

            return rewards

   
    def get_value(self, encoder_outputs, decoder_hidden_init, sequences, lengths,\
                        user_embeds, news_body_indexes, news_body, news_vert, \
                        wd_strs, wd_indexes, src=None, sos_id=1, eos_id=2):
        
        bz = sequences.shape[0]
        max_title_length = 16
        input_src = torch.LongTensor([sos_id] * bz).view(bz, 1).to(self.device)
        inputs = torch.cat([input_src, sequences[:,:-1]], dim=-1) # exclude last target from inputs
        rewards = np.zeros((bz, 16))
        sample_seqs = []
        '''
        example
            sequence = [I think so ... ]
            inputs = [sos I think so ...]
            At step 0:
            - reward[0] is the reward of 'I'
            - tgt_input should be [sos I]
        '''
        for seq_num in range(max_title_length-1):
            decoder_outputs, decoder_hidden_init2, _, _ = self.model.decoder.forward_step(inputs[:,:seq_num+1], \
                                                decoder_hidden_init, encoder_outputs, user_embeds, src)
            
            reward = np.zeros((bz, self.train_args['sample_num']))
            for i in range(self.train_args['sample_num']):
                # 
                sample_seq = []
                sample_lengths = np.array([max_title_length] * bz)
                sample_seq.append(sequences[:,:seq_num+1])
                
                decoder_input = inputs[:,seq_num+1].unsqueeze(1)
                decoder_hidden = decoder_hidden_init2
                # (bz,vocab_dim) -> (bz,1)
                # decoder_input = torch.multinomial(decoder_outputs[:,seq_num,:],1) 
                for di in range(seq_num+1, max_title_length):
                    decoder_output, decoder_hidden, _, _ = self.model.decoder.forward_step(decoder_input, \
                                                                                    decoder_hidden, encoder_outputs, user_embeds, src)
                    step_output = decoder_output.squeeze(1) # step_output (bz, hidden_dim)
                    decoder_input = torch.multinomial(torch.exp(step_output),1) # sampling (bz,1)
                    sample_seq.append(decoder_input)

                    eos_batches = decoder_input.data.eq(eos_id)
                    eos_batches = eos_batches.cpu().view(-1).numpy()
                    update_idx = ((sample_lengths > di) & eos_batches) != 0
                    sample_lengths[update_idx] = len(sample_seq)
                
                sample_seq = torch.cat(sample_seq,dim=1) # bz*16

                sample_wd_strs, sample_wd_indexes,  _, _= self.model.predict(sample_seq, sample_lengths)
                temp_rewards = self.get_single_reward(sample_seq, sample_lengths, sample_wd_strs, sample_wd_indexes, \
                                                        news_body, news_vert, news_body_indexes, user_embeds, sos_id=1, eos_id=2)

                reward[:,i] = temp_rewards
            rewards[:,seq_num] = np.mean(reward, axis=-1)
        temp_rewards = self.get_single_reward(sequences, lengths, wd_strs, wd_indexes, \
                                                news_body, news_vert, news_body_indexes, user_embeds, sos_id=1, eos_id=2)
        rewards[:,-1] = temp_rewards
        for ind,length in enumerate(lengths):
            rewards[ind,(length-1)] = rewards[ind,-1]
            rewards[ind,length:] = 0
        return rewards


    def get_reward(self, sequences, lengths, wd_strs, wd_indexes, wd_str_lists, wd_index_lists, news_body, news_vert, news_body_indexes, news_body_split, user_embeds, sos_id=1, eos_id=2):
        
        # sequences (bz, len)
        bz, len_ = sequences.shape
        cur_rewards = np.zeros((bz, len_))
        tmp_seq = deepcopy(sequences)
        tmp_seq[:,:] = eos_id
        for i in range(len_):
            tmp_seq[:, i] = sequences[:, i]
            tmp_wd_index = [' '.join(x[:i+1]) for x in wd_index_lists]
            tmp_wd_index_list = [x[:i+1] for x in wd_index_lists]

            cur_rewards[:, i] = self.get_cov_reward(tmp_wd_index, tmp_wd_index_list, news_body_indexes, news_body_split) * 2
        
        diff = np.diff(cur_rewards)
        rewards = np.concatenate([cur_rewards[:,:1], diff], axis=1)
        inds = deepcopy(lengths) - 1
        inds = inds.reshape(-1, 1)
        end_reward = self.get_per_reward(sequences, lengths, news_body, news_vert, user_embeds) + \
                        self.get_flu_reward2(sequences, wd_strs)
        end_reward = end_reward.reshape(-1, 1)
        rewards[np.arange(rewards.shape[0])[:, None], inds] += end_reward * 0.1
        
        return rewards

    def get_per_reward(self, seqs, sample_lengths, news_body, news_vert, user_embeds, sos_id=1, eos_id=2):
        seqs_new = deepcopy(seqs)
        for ind,temp_len in enumerate(sample_lengths):
            if seqs_new[ind, temp_len - 1] == eos_id:
                seqs_new[ind, temp_len - 1:] = 0
            else:
                seqs_new[ind, temp_len:] = 0

        with torch.no_grad():
            news_embeds = self.usermodel.news_encoder([seqs_new, news_vert, news_body]) # (bz, 64)
        personalized_r = torch.bmm(news_embeds.unsqueeze(1), user_embeds.unsqueeze(-1))# (bz, 1)
        personalized_r = personalized_r.squeeze(-1).squeeze(-1).detach().cpu().numpy()
        return np.tanh(personalized_r)

    def get_flu_reward(self, seqs, wd_strs):
        bz = seqs.shape[0]
        fluency_r = np.ones((bz)).astype(np.float32)
        fluency_r = fluency_r*0.5

        tokenize_input = self.lm_tokenizer(wd_strs, padding=True, truncation=True, return_tensors="pt")['input_ids'].to(self.device)
        logits = self.lm(tokenize_input, labels=tokenize_input)['logits']
        # logits =  torch.softmax(logits,-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokenize_input[..., 1:].contiguous()
        for i in range(len(logits)):
            temp = self.lm_criterion(shift_logits[i], shift_labels[i].long()).item()
            # fluency_r[i] = np.tanh(temp/10)
            fluency_r[i] = temp/10

        '''
        TOO SLOW:
        for ind, sen in enumerate(wd_strs):
            tokens = self.lm_tokenizer.encode(sen)#['input_ids']
            tokenize_input = torch.as_tensor(tokens).to(self.device)
            temp_loss = self.lm(tokenize_input, labels=tokenize_input)[0].item()
            if not np.isnan(temp_loss):
                fluency_r[ind] = np.tanh(temp_loss/10)
        '''
        return fluency_r

    def get_flu_reward2(self, seqs, wd_strs):
        bz = seqs.shape[0]
        fluency_r = np.ones((bz)).astype(np.float32)
        fluency_r = fluency_r*0.2

        tokenize_input = self.lm_tokenizer(wd_strs, padding=True, truncation=True, return_tensors="pt")['input_ids'].to(self.device)

        for ind,wd_str in enumerate(wd_strs):
            tokenize_input = self.lm_tokenizer(wd_str, return_tensors="pt")['input_ids'].to(self.device)
            with torch.no_grad():
                try:
                    loss = self.lm(tokenize_input, labels=tokenize_input).loss.item()
                except:
                    loss = 8.
                fluency_r[ind] = np.tanh(50/math.exp(loss)) + len(set(wd_str.split())) / len(wd_str.split())
        return fluency_r/2
    
    
    def get_cov_reward(self,  wd_indexes, wd_index_list, news_body_indexes, news_body_split):
        if self.mode == 1:
            cover_r = ROUGE_Score(self.rouge_evaluator, wd_indexes, news_body_indexes)
            cover_r = cover_r[0] + cover_r[3] + cover_r[5] + cover_r[6] + cover_r[8]
        elif self.mode == 2:
            cover_r = ROUGE_Score(self.rouge_evaluator, wd_indexes, news_body_indexes)
            cover_r = cover_r[0] + cover_r[3] + cover_r[6]
            penalize_r = np.array([ (1- float(len(wd_lis)) / len(set(wd_lis))) * 0.1 for wd_lis in wd_index_list])
            cover_r += penalize_r
        elif self.mode == 3:
            cover_r = ROUGE_Score(self.rouge_evaluator, wd_indexes, news_body_indexes)
            cover_r = cover_r[0] + cover_r[3] + cover_r[5] + cover_r[6] + cover_r[8]
            penalize_r = np.array([ (1- float(len(wd_lis)) / len(set(wd_lis))) * 0.1 for wd_lis in wd_index_list])
            cover_r += penalize_r
        elif self.mode == 4:
            cover_r = ROUGE_Score_TOP3(self.rouge_evaluator, wd_indexes, news_body_indexes, news_body_split)
            cover_r = cover_r[0] + cover_r[3] + cover_r[5] + cover_r[6] + cover_r[8]
        elif self.mode == 5:
            cover_r = ROUGE_Score_TOP3(self.rouge_evaluator, wd_indexes, news_body_indexes, news_body_split)
            cover_r = cover_r[0] + cover_r[3] + cover_r[5] + cover_r[6] + cover_r[8]
            penalize_r = np.array([ (1- float(len(wd_lis)) / len(set(wd_lis))) * 0.1 for wd_lis in wd_index_list])
            cover_r += penalize_r

        return cover_r


    def get_single_reward(self, seqs, sample_lengths, wd_strs, wd_indexes, news_body, news_vert, news_body_indexes, user_embeds, sos_id=1, eos_id=2):
        
        bz = seqs.shape[0]

        # personalized_r
        personalized_r = self.get_per_reward(seqs, sample_lengths, news_body, news_vert, user_embeds)

        # flency_r
        fluency_r = self.get_flu_reward2(seqs, wd_strs)

        # cover_r
        cover_r = self.get_cov_reward(wd_indexes, None, news_body_indexes)

        all_rewards = (personalized_r + fluency_r) *0.1 + 5*cover_r

        return all_rewards
