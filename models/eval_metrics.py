import math
import numpy as np
import pdb
import os
import pdb

def hit_at_k(actual, predicted, topk, dataset=None, mode=None):

    act_set = actual
    pred_set = predicted[:, -topk:]
    hit = (pred_set == act_set.unsqueeze(1)).sum(-1)
    output = hit.clone()
    #avoid repeat items
    hit = (hit != 0).sum()

    return hit / float(len(actual))

def ndcg_at_k(actual, predicted, topk, dataset=None, mode=None):
    res = 0
    k = 1
    output = []

    for user_id in range(len(actual)):
        idcg = idcg_k(k)
        dcg_k = sum([(int(predicted[user_id][-j-1] == actual[user_id]) / math.log(j+2, 2)) for j in range(topk)])
        res += dcg_k / idcg
        output.append(dcg_k / idcg)

    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
