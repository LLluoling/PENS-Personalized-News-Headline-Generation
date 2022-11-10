import numpy as np
import pandas as pd
import torch
import pickle
import re
import json
from collections import Counter
from torch.utils.data import DataLoader, Dataset


class Seq2SeqDataset(Dataset):
    def __init__(self, sources, target_inputs, target_outputs):
        self.sources = np.array(sources)
        self.target_inputs = np.array(target_inputs)
        self.target_outputs = np.array(target_outputs)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.target_inputs[idx], self.target_outputs[idx]


class ImpressionDataset(Dataset):
    def __init__(self, news_scoring, sources, target_inputs, target_outputs, TrainUsers, TrainSamples):
        self.news_scoring = news_scoring
        self.sources = np.array(sources)
        self.target_inputs = np.array(target_inputs)
        self.target_outputs = np.array(target_outputs)
        self.TrainSamples = TrainSamples
        self.TrainUsers = TrainUsers

    def __len__(self):
        return len(self.TrainSamples)

    def __getitem__(self, idx):
        userid = self.TrainSamples[idx][0]
        candidate_news = self.TrainSamples[idx][1][0]
        
        clicked_news = np.array(self.TrainUsers[userid])
        clicked_rep = self.news_scoring[clicked_news]
        source = self.sources[candidate_news]
        target_input = self.target_inputs[candidate_news]
        target_output = self.target_outputs[candidate_news]
        
        return candidate_news, clicked_rep, source, target_input, target_output


class TestImpressionDataset(Dataset):
    def __init__(self, news_scoring, sources, TrainUsers, TrainSamples):
        self.news_scoring = news_scoring
        self.sources = np.array(sources)
        self.TrainSamples = TrainSamples
        self.TrainUsers = TrainUsers

    def __len__(self):
        return len(self.TrainSamples)

    def __getitem__(self, idx):
        userid = self.TrainSamples[idx][0]
        candidate_news = self.TrainSamples[idx][1]
        rewrite_titile = self.TrainSamples[idx][2]
        
        clicked_news = np.array(self.TrainUsers[userid])
        clicked_rep = self.news_scoring[clicked_news]
        source = self.sources[candidate_news]
        
        
        return candidate_news, clicked_rep, source, rewrite_titile



