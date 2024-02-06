#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from functions import convolve2d#, gaussian
import kernel
from train.fit import target

## Kernel

K_default = kernel.Moore3

## Model

def step(A, p=[0, 2.5, 0.1], overshoot=0, offset=-0, round=False, dt=1, rand=0.6, K=None):
    if K is None:
        K = K_default
        
    N = convolve2d(A, K, mode='same', boundary='wrap')
    #N = convolve2d(A, K, mode='same', boundary='fill', fillvalue=0)
    T = target(A, N, p, overshoot=overshoot) + offset
    if round is True:
        #round = np.round
        round = lambda f: np.round(np.tanh(f))
    if round:
        T = round(T)
    #A_new = np.clip(A + dt*(T-A), -1, 1)
    A_new = dt*T + (1-dt)*A
    return (N, T, A_new)
