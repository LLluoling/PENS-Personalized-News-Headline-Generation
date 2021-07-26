import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def evaluate(user_scorings,news_scorings,Impressions):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 =[]
    
    CTR1 = []
    CTR10 = []
    
    g = 0
    for i in tqdm(range(len(Impressions))):
        docids = np.array(Impressions[i][1])
        labels = np.array(Impressions[i][2])
        if labels.mean() ==0 or labels.mean()==1:
            g +=1
            continue
        uv = user_scorings[i]
        
        nv = news_scorings[docids]
        score = np.dot(nv,uv)
        

        auc = roc_auc_score(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
        
        arg = score.argsort()[-5:]
        sorted_label = labels[arg]
        CTR10.append(sorted_label.mean())
        CTR1.append(sorted_label[-1])
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)
    CTR1 = np.array(CTR1)
    CTR10 = np.array(CTR10)
    
    AUC = AUC.mean()
    MRR = MRR.mean()
    nDCG5 = nDCG5.mean()
    nDCG10 = nDCG10.mean()
    CTR10 = CTR10.mean()
    CTR1 = CTR1.mean()
    print(g,g/len(Impressions))
    
    return AUC, MRR, nDCG5, nDCG10,CTR1,CTR10