import numpy as np
import pandas as pd
import torch
import pickle
import re
import json
from torch.utils.data import DataLoader, Dataset
from collections import Counter

class news_dataset(Dataset):
    def __init__(self, news_title, news_vert, news_body):
        self.news_title = news_title
        self.news_vert = news_vert
        self.news_body = news_body
    
    def __getitem__(self, idx):
        return [self.news_title[idx], self.news_vert[idx], self.news_body[idx]]
    
    def __len__(self):
        return len(self.news_title)

def news_collate_fn(news_info):
    news_info = [torch.LongTensor(i) for i in zip(*news_info)]
    return news_info

class UserDataset(Dataset):
    def __init__(self, news_scoring,  Users):
        self.news_scoring = news_scoring
        self.Users = Users
    
    def __getitem__(self, idx):
        user = np.array(self.Users[idx])
        clicked_rep = self.news_scoring[user]
        return  clicked_rep
    
    def __len__(self):
        return len(self.Users)

class TrainDataset(Dataset):
    def __init__(self, TrainUsers, TrainSamples, news_title, news_vert, news_body):
        self.TrainUsers = TrainUsers
        self.TrainSamples = TrainSamples
        
        self.news_title = news_title
        self.news_vert = news_vert
        self.news_body = news_body

    def __getitem__(self, idx):
        sample = self.TrainSamples[idx]
        userid = sample[0]
        news = np.array(sample[1])
        user = self.TrainUsers[userid]
        label = np.array(sample[2])

        news_feature = [self.news_title[news], self.news_vert[news], self.news_body[news]]
        user_feature = [self.news_title[user], self.news_vert[user], self.news_body[user]]

        return news_feature, user_feature, label
    
    def __len__(self):
        return len(self.TrainSamples)

def collate_fn(arr):
    bz = len(arr)
    news_feature, user_feature, _ = zip(*arr)
    user_feature = [torch.LongTensor(i) for i in zip(*user_feature)]
    news_feature = [torch.LongTensor(i) for i in zip(*news_feature)]
    label = torch.zeros(bz, dtype=torch.long) # (bz)
    return user_feature, news_feature, label


