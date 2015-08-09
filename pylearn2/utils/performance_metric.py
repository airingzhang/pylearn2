'''
Created on Aug 9, 2015

@author: ningzhang

This file defines some evaluation scores often used in our experiments.
'''

import numpy as np

def ScoreOneLabel(self, preds, targets):
    """
    Computes Average precision and precision at 50 for one label.
    
    This code is copied from Nitish's deepnet.
    actually it can be easily extend to deal with the whole matrix.
    
    Here we retain the structure temporally for simplicity .
    We may absorb the ComputeSocre into this method in the future.
    """
    
    targets_sorted = targets[(-preds.T).argsort().flatten(),:]
    cumsum = targets_sorted.cumsum()
    prec = cumsum / np.arange(1.0, 1 + targets.shape[0])
    total_pos = float(sum(targets))
    if total_pos == 0:
        total_pos = 1e-10
    ap = np.dot(prec, targets_sorted) / total_pos
    prec50 = prec[50]
    return ap, prec50

def ComputeScore(self, preds, targets):
    
    """
    Computes Average precision and precision at 50.
    """
    assert preds.shape == targets.shape
    numdims = preds.shape[1]
    ap = 0
    prec = 0
    ap_list = []
    prec_list = []
    for i in range(numdims):
        this_ap, this_prec = self.ScoreOneLabel(preds[:,i], targets[:,i])
        ap_list.append(this_ap)
        prec_list.append(this_prec)
        ap += this_ap
        prec += this_prec
    ap /= numdims
    prec /= numdims
    return ap, prec, ap_list, prec_list