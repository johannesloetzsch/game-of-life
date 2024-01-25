#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from functions import convolve2d#, gaussian
from kernel import K3
from train.fit import target

## Kernel

K = K3

## Model

clip = lambda A: np.clip(A, 0, 1)
#clip = lambda A, m=20: sigmoid(m*(A-.5))

def step(A, p=[-0.32, 2.82, 0.14], overshoot=0.95, offset=0, round=False, dt=1.0, rand=0.6):
    N = convolve2d(A, K, mode='same', boundary='wrap')
    #N = convolve2d(A, K, mode='same', boundary='fill', fillvalue=0)
    T = target(A, N, p, overshoot=overshoot) + offset
    if round is True:
        #round = np.round
        round = lambda f: np.round(np.tanh(f))
    if round:
        T = round(T)
    A_new = dt*T + (1-dt)*A
    return (N, T, A_new)