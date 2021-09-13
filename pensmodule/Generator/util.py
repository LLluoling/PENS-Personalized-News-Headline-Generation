import numpy as np


def ROUGE_Score(rouge_evaluator, news_headline, news_body):

    refs = [_.lower() for _ in news_body]
    hyps = [_.lower() for _ in news_headline]
    scores = rouge_evaluator.get_scores(hyps, refs)
    scores1_f = np.array([score['rouge-1']['f'] for score in scores])
    scores1_r = np.array([score['rouge-1']['r'] for score in scores])
    scores1_p = np.array([score['rouge-1']['p'] for score in scores])

    scores2_f = np.array([score['rouge-2']['f'] for score in scores])
    scores2_r = np.array([score['rouge-2']['r'] for score in scores])
    scores2_p = np.array([score['rouge-2']['p'] for score in scores])

    scoresl_f = np.array([score['rouge-l']['f'] for score in scores])
    scoresl_r = np.array([score['rouge-l']['r'] for score in scores])
    scoresl_p = np.array([score['rouge-l']['p'] for score in scores])

    # size: (9, batch_size)
    return np.array([scores1_f, scores1_r, scores1_p, scores2_f, scores2_r, scores2_p, scoresl_f, scoresl_r, scoresl_p])
    
    # return scores1_f + scores2_f + scoresl_f
    # return scores1_f + scores2_f + scoresl_f + scores2_p + scoresl_p


def ROUGE_Score_TOP3(rouge_evaluator, news_headline, news_body, news_body_split):
    scores = np.zeros((9, len(news_headline)))
    for ind, (headline, body_split) in enumerate(zip(news_headline, news_body_split)):
        single_score = ROUGE_Score(rouge_evaluator, [headline]* len(body_split), body_split)
        single_score = np.average(np.sort(single_score)[:,-3:], 1)
        scores[:, ind] = single_score
    return scores


def discount_reward(r, gamma=0.98):
    seq_len = r.shape[1]
    for t in reversed(range(0, seq_len-1)):
        r[:, t] = r[:, t+1] * gamma + r[:, t]
    return r


def discount_reward_update(rewards, lengths, GAMMA=0.98):
    bz = rewards.shape[1]
    Qvals = np.zeros_like(rewards)
    for b in range(len(rewards)):
        Qvals[b][lengths[b] - 1] = rewards[b][lengths[b] - 1]
        for l in reversed(range(1, lengths[b])):
            Qvals[b][l-1]  = rewards[b][l-1] + GAMMA * Qvals[b][l]
    return Qvals


