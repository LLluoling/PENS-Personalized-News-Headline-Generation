import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from .modules import MultiHeadAttention,AttentionPooling

class NAML(nn.Module):
    def __init__(self, embedding_matrix, category_dict):
        super(NAML, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding_matrix).type(torch.FloatTensor), requires_grad=True)
        self.vert_embed = nn.Embedding(len(category_dict)+1, 400, padding_idx=0)

        
        self.attn_title = MultiHeadAttention(300, 20, 20, 20)
        self.attn_body = MultiHeadAttention(300, 20, 20, 20)
        # self.attn_news = MultiHeadAttention(400, 20, 20, 20)

        self.title_attn_pool = AttentionPooling(400, 400)
        self.body_attn_pool = AttentionPooling(400,400)
        self.news_attn_pool = AttentionPooling(400,400)

        self.attn_pool_news = AttentionPooling(64, 64)

        self.drop_layer = nn.Dropout(p=0.2)
        self.fc = nn.Linear(400, 64)
        self.criterion = nn.CrossEntropyLoss()
        
    def news_encoder(self, news_feature):
        title, vert, body = news_feature[0], news_feature[1],news_feature[2]
        news_len = title.shape[-1]
        title = title.reshape(-1, news_len)
        title = self.drop_layer(self.embed(title))
        title = self.drop_layer(self.attn_title(title, title, title))
        title = self.title_attn_pool(title).reshape(-1, 1, 400)

        body_len = body.shape[-1]
        body = body.reshape(-1, body_len)
        body = self.drop_layer(self.embed(body))
        body = self.drop_layer(self.attn_body(body, body, body))
        body = self.body_attn_pool(body).reshape(-1, 1, 400)


        vert = self.drop_layer(self.vert_embed(vert.reshape(-1))).reshape(-1, 1, 400)

        news_vec = torch.cat((title, body, vert), 1)
        news_vec = self.news_attn_pool(news_vec)
        news_vec = self.fc(news_vec)
        return news_vec

        
    def user_encoder(self, x):

        x = self.attn_pool_news(x).reshape(-1, 64)
        return x

    def forward(self, user_feature, news_feature, label=None, compute_loss=True):

        bz = label.size(0)
        news_vecs = self.news_encoder(news_feature).reshape(bz, -1, 64)
        
        user_newsvecs = self.news_encoder(user_feature).reshape(bz, -1, 64)
        user_vec = self.user_encoder(user_newsvecs).unsqueeze(-1) # batch * 400 * 1
        score = torch.bmm(news_vecs, user_vec).squeeze(-1)
        if compute_loss:
            loss = self.criterion(score, label)
            return loss, score
        else:
            return score


class NRMS(nn.Module):
    def __init__(self, embedding_matrix):
        super(NRMS, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding_matrix).type(torch.FloatTensor), requires_grad=True)
        
        
        self.attn_word = MultiHeadAttention(300, 20, 20, 20)
        # self.attn_news = MultiHeadAttention(400, 20, 20, 20)

        self.attn_pool_word = AttentionPooling(400, 400)
        self.attn_pool_news = AttentionPooling(64, 64)

        self.drop_layer = nn.Dropout(p=0.2)
        self.fc = nn.Linear(400, 64)
        self.criterion = nn.CrossEntropyLoss()
        
    def news_encoder(self, news_feature):
        x = news_feature[0]
        news_len = x.shape[-1]
        x = x.reshape(-1, news_len)
        x = self.drop_layer(self.embed(x))
        x = self.drop_layer(self.attn_word(x, x, x))
        x = self.attn_pool_word(x).reshape(-1, 400)
        x = self.fc(x)
        return x

        
    def user_encoder(self, x):

        x = self.attn_pool_news(x).reshape(-1, 64)
        return x

    def forward(self, user_feature, news_feature, label=None, compute_loss=True):

        bz = label.size(0)
        news_vecs = self.news_encoder(news_feature).reshape(bz, -1, 64)
        
        user_newsvecs = self.news_encoder(user_feature).reshape(bz, -1, 64)
        user_vec = self.user_encoder(user_newsvecs).unsqueeze(-1) # batch * 400 * 1
        score = torch.bmm(news_vecs, user_vec).squeeze(-1)
        if compute_loss:
            loss = self.criterion(score, label)
            return loss, score
        else:
            return score


class EBNR(nn.Module):
    def __init__(self, embedding_matrix):
        super(EBNR, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding_matrix).type(torch.FloatTensor), requires_grad=True)
        
        
        self.gru_word = nn.GRU(300, 400, batch_first=True)
        self.gru_news = nn.GRU(64, 64, batch_first=True)

        self.attn_pool_word = AttentionPooling(400, 400)
        self.attn_pool_news = AttentionPooling(64, 64)

        self.drop_layer = nn.Dropout(p=0.2)
        self.fc = nn.Linear(400, 64)
        self.criterion = nn.CrossEntropyLoss()
        
    def news_encoder(self, news_feature):
        x = news_feature[0]
        news_len = x.shape[-1]
        x = x.reshape(-1, news_len)
        x = self.drop_layer(self.embed(x))
        x = self.gru_word(x)[0]
        x = self.attn_pool_word(x).reshape(-1, 400)
        x = self.fc(x)
        return x

        
    def user_encoder(self, x):

        x = self.gru_news(x)[0]
        x = self.attn_pool_news(x).reshape(-1, 64)
        return x

    def forward(self, user_feature, news_feature, label=None, compute_loss=True):

        bz = label.size(0)
        news_vecs = self.news_encoder(news_feature).reshape(bz, -1, 64)
        
        user_newsvecs = self.news_encoder(user_feature).reshape(bz, -1, 64)
        user_vec = self.user_encoder(user_newsvecs).unsqueeze(-1) # batch * 400 * 1
        score = torch.bmm(news_vecs, user_vec).squeeze(-1)
        if compute_loss:
            loss = self.criterion(score, label)
            return loss, score
        else:
            return score



