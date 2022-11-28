import pandas as pd
import numpy as np
import pickle
import nltk.stem as ns
import pdb
from tqdm import trange

ALPHA = 0.66
LAMBDA = 1


def tokenize(sent):
    sent = sent.lower()
    sent = sent.strip('.')
    lemmatizer = ns.WordNetLemmatizer()
    sent = sent.replace("'s", '')
    sent = sent.split(' ')
    for ith, word in enumerate(sent):
        word_n = lemmatizer.lemmatize(word, pos='n')
        word_v = lemmatizer.lemmatize(word, pos='v')
        
        if word_n != word_v:
            if word_n == word:
                sent[ith] = word_v
            else:
                sent[ith] = word_n
    return sent


def cs_word(w_cause, w_effect, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA):
    
    M = 62675002
    
    try:
        p_w_cause = float(sum(causes[w_cause].values())) / M
    except KeyError:
        p_w_cause = 0
        
    try:
        p_w_effect = float(sum(effects[w_effect].values())) / M
    except KeyError:
        p_w_effect = 0
    
    try:
        p_join = float(causes[w_cause][w_effect]) / M
    except KeyError:
        p_join = 0
        
    # print(p_w_cause, p_w_effect, p_join)
    
    if p_join != 0:
        cs_nes = p_join / p_w_cause ** ALPHA / p_w_effect
        cs_surf = p_join / p_w_cause / p_w_effect ** ALPHA 
        cs = cs_nes ** LAMBDA * cs_surf ** (1 - LAMBDA)
    else:
        cs = float(2) / len(causes)
    return cs
    
    
def cs_sent(s_cause, s_effect, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA):
    
    cs = 0
    num_zero = 0
    for w_cause in s_cause:
        for w_effect in s_effect:
            cs_tmp = cs_word(w_cause, w_effect, effects, causes, ALPHA=ALPHA, LAMBDA=LAMBDA)
            cs = cs + cs_tmp
            if cs_tmp == 0:
                num_zero = num_zero + 1        
    cs = cs / (len(s_cause) + len(s_effect))
    
    return cs
    
    
def inf(data):
    L = data.shape[0]

    premise = data['cause'].tolist()
    hypothesis = data['effect'].tolist()
    truth = data['explanation'].tolist()

    pred = list()
    reference_rnn = list()
    reference_gpt2 = list()
    reference_mt = list()

    # Causal_Strength(Cause, Effect)
    cs_1 = list()

    # Causal_Strength(Cause + Truth, Effect)
    cs_2 = list()

    # Causal_Strength(Cause, Truth + Effect)
    cs_3 = list()

    for ith in trange(L):

        premise_tmp = tokenize(premise[ith])
        hypothesis_tmp = tokenize(hypothesis[ith])
        premise_truth_tmp = tokenize(premise[ith] + truth[ith])
        truth_hypothesis_tmp = tokenize(truth[ith] + hypothesis[ith])

        cs_tmp_1 = cs_sent(premise_tmp, hypothesis_tmp, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA)
        cs_tmp_2 = cs_sent(premise_truth_tmp, hypothesis_tmp, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA)
        cs_tmp_3 = cs_sent(premise_tmp, truth_hypothesis_tmp, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA)

        cs_1.append(cs_tmp_1)
        cs_2.append(cs_tmp_2)
        cs_3.append(cs_tmp_3)
        
    res = data.copy()
    res['Cause:Effect'] = cs_1
    res['Cause+Truth:Effect'] = cs_2
    res['Cause:Truth+Effect'] = cs_3
    
    res.to_csv('CEQ.csv')
    return res


if __name__ == '__main__':
    data = pd.read_csv('./data/eval.csv')

    f = open("causes.pkl", 'rb')
    causes = pickle.load(f)
    f.close()

    f = open("effects.pkl", 'rb')
    effects = pickle.load(f)
    f.close()

    res = inf(data)
