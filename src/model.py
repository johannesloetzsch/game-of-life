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

def step(A, dt=1.0, rand=0.6, p=[-0.32, 2.82, 0.14]):
    N = convolve2d(A, K, mode='same', boundary='wrap')
    #N = convolve2d(A, K, mode='same', boundary='fill', fillvalue=0)
    #G = growth(N)
    #T = target(A, N, 2.75, 2, 0.5)
    T = target(A, N, p)
    #A = clip(A + dt*G + rand*np.random.rand(A.shape[0], A.shape[1])-0.5*rand)
    A = dt*np.round(T+0.03) + (1-dt)*A
    A = np.round((A+1)/2)*2-1
    return (N, T, A)