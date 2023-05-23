import numpy as np

def RPP(ranking1, ranking2, p=None):
    if p is None:
        m = min(len(ranking1), len(ranking2))
        p = np.ones(m)/m
    return (p * np.sign(ranking2[:len(p)] - ranking1[:len(p)])).sum()

def DCGPP(ranking1, ranking2):
    m = min(len(ranking1), len(ranking2))
    p = 1 / np.log2(np.arange(1, m+1) + 1)
    p /= p.sum()
    return RPP(ranking1, ranking2, p)

def INVPP(ranking1, ranking2):
    m = min(len(ranking1), len(ranking2))
    p = 1 / np.arange(1, m+1)
    p /= p.sum()
    return RPP(ranking1, ranking2, p)

def ISL(ranking):
    return ranking[0]

def delta_ISL(ranking1, ranking2):
    return ISL(ranking1) - ISL(ranking2)

def TSL(ranking):
    return ranking[-1]

def delta_TSL(ranking1, ranking2):
    return TSL(ranking1) - TSL(ranking2)

def ASL(ranking):
    return np.mean(ranking)

def delta_ASL(ranking1, ranking2):
    return ASL(ranking1) - ASL(ranking2)

def DCG(relevant, k=None):
    if k is None: k=len(relevant)
    return (relevant[:k]/np.log2(np.arange(1, k+1) + 1)).sum()

def delta_DCG(relevant1, relevant2, k=None):
    return DCG(relevant1, k) - DCG(relevant2, k)

def NDCG(relevant, k=None):
    if k is None: k = len(relevant)
    idcg = DCG(np.sort(relevant)[::-1])
    return 0 if idcg == 0 else DCG(relevant)/idcg

def delta_NDCG(relevant1, relevant2, k=None):
    return NDCG(relevant1, k) - NDCG(relevant2, k)

def P_at_k(relevant, k):
    return relevant[:min(k, len(relevant))].sum() / k

def AP(ranking):
    return np.mean(1/ranking)

def delta_AP(ranking1, ranking2):
    return AP(ranking1) - AP(ranking2)

def RR(ranking):
    return 1. / ranking[0]
    
def delta_RR(ranking1, ranking2):
    return RR(ranking1) - RR(ranking2)

def OI(rankings1, rankings2):
    N = min(len(rankings1), len(rankings2))
    p = np.zeros(N)
    clicks_A = 0
    clicks_B = 0
    clicks = 0
    
    for i in range(N):
        if rankings1[i] == rankings2[i]:
            p[i] = 0.5
        elif rankings1[i] < rankings2[i]:
            p[i] = 1
        else:
            p[i] = 0
        
        if np.random.uniform() < p[i]:
            clicks_A += 1
            clicks += 1
        else:
            clicks_B += 1
            clicks += 1
            
    return (clicks_A - clicks_B) / clicks

def RBP(i, gamma):
    return gamma**(i-1)

def DCG2(i):
    return 1 / np.log(i + 1)
