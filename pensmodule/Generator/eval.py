from .util import *
import torch
import numpy as np 
import pandas as pd 
import torch.nn as nn
import rouge 
from tqdm import tqdm
import torch
import sys
import numpy as np
from tensorboardX import SummaryWriter
from .beam_omt import Translator

def predict(usermodel, model, test_iter, device, index2word, beam=True, beam_size=5, eos_id=2):
    usermodel.eval()
    model.eval()
    preds, ys = [], []
    pbar = tqdm(test_iter)
    if beam:
        t = Translator(model, index2word, beam_size=beam_size)      
    for i, batch in enumerate(pbar):
        news_ids, clicked_rep, src, rewrite_titile = batch
        clicked_rep = torch.as_tensor(clicked_rep, device=device)
        src = torch.as_tensor(src, device=device).long()
        with torch.no_grad():
            user_embeds = usermodel.user_encoder(clicked_rep)
            if beam:
                sent_b, _ = t.translate_batch(src, user_embeds)
                headlines = []
                
                for i in range(len(sent_b)):
                    new_words = []
                    for w in sent_b[i][0]:
                        if w==eos_id:
                            break
                        new_words.append(w)
                        if len(new_words)>2 and (new_words[-2]==w):
                            new_words.pop()
                    sent_beam_search = ' '.join([index2word[idx] for idx in new_words])
                    headlines.append(sent_beam_search)
                preds.extend(headlines)
            else:
                _, _, _, _, _, wd_strs, _, _ = \
                                        model(src, None, None, user_embeds, 0)
                preds.extend(wd_strs)
        ys.extend(rewrite_titile)
    modified_pred = []
    for pred in preds:
        if pred.strip() != '':
            modified_pred.append(pred)
        else:
            modified_pred.append('news')
    rouge_evaluator = rouge.Rouge(metrics=['rouge-1','rouge-2', 'rouge-l'])
    
    refs = [_.lower() for _ in ys]
    hyps = [_.lower() for _ in modified_pred]
    scores = rouge_evaluator.get_scores(hyps, refs)
    scores1 = np.array([score['rouge-1']['f'] for score in scores])
    scores2 = np.array([score['rouge-2']['f'] for score in scores])
    scoresf = np.array([score['rouge-l']['f'] for score in scores])
    return scores1, scores2, scoresf

def load_model_from_ckpt(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    if torch.cuda.device_count() > 1:
        print('multiple gpu training')
        model = nn.DataParallel(model)
    return model

