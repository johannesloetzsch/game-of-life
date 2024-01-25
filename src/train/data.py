#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from model import step
import numpy as np
import itertools
from operator import itemgetter

def borderless(frames):
    return [f[1:-1,1:-1] for f in frames]

## N is required for training
def pretraining_step(frame0_, frame1_):
    frame0, frame1 = borderless([frame0_, frame1_])
    (N_, T_, A_) = step(frame0_, round=False)
    N, T, A = borderless([N_, T_, A_])
    return frame0, frame1, N, T, A

def groupby(l, keyfn):
    return dict([[key, [g[1] for g in group]]
                 for key, group
                 in itertools.groupby(sorted(l, key=keyfn), keyfn)])

def training_data(N, frame0, frame1):
    Xall = N.flatten()
    Yall = ((frame1-frame0)).flatten()
    zipped = list(zip(Xall, Yall))
    z_all = groupby(zipped, itemgetter(0))
    #print(dict([[k,len(v)] for [k,v] in z_all.items()]))
    def norm(Y):
        #minmax = np.min(Y) if np.average(Y)<0 else np.max(Y)
        minmax = np.average(Y)
        return np.clip(minmax, -1, 1)
    z = [[k,norm(v)] for [k,v] in z_all.items()]
    z = dict(z)
    X, Y = np.array(list(z.keys())), np.array(list(z.values()))
    #print(X)
    #print(Y)
    return X, Y
