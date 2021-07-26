import numpy as np


def ROUGE_Score(rouge_evaluator, news_headline, news_body):

    refs = [_.lower() for _ in news_body]
    hyps = [_.lower() for _ in news_headline]
    scores = rouge_evaluator.get_scores(hyps, refs)
    scores1 = np.array([score['rouge-1']['f'] for score in scores])
    scoresf = np.array([score['rouge-l']['f'] for score in scores])
    # scores = np.array([(score['rouge-1']['f'] + score['rouge-l']['f'])/2 for score in scores])
    # scores = np.array([np.tanh(score['rouge-l']['f']*500) for score in scores])
    return scores1, scoresf


# def discount_reward(r, gamma,final_r):
#     discounted_r = np.zeros_like(r)
#     running_add = final_r
#     for t in reversed(range(0, len(r))):
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#     return discounted_r


def discount_reward(r, gamma=0.98):
    seq_len = r.shape[1]
    for t in reversed(range(0, seq_len-1)):
        r[:,t] = r[:,t+1] * gamma + r[:,t]
    return r

