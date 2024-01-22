#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from functions import convolve2d, gaussian
from kernel import K11_discrete

## Kernel

K = K11_discrete

## Model

clip = lambda A: np.clip(A, 0, 1)
#clip = lambda A, m=20: sigmoid(m*(A-.5))

def growth(N):
    u = 0.35
    v = 5
    w = 0.2
    G_unlimited = -1+2*v*gaussian((N-u)/w)
    #return G_unlimited
    return np.clip(G_unlimited, -1, 1)
    #return -1*2*sigmoid(30*G_unlimited)

def step(A, dt=0.02, rand=0.6):
    N = convolve2d(A, K, mode='same', boundary='wrap')
    #N = convolve2d(A, K, mode='same', boundary='fill', fillvalue=0)
    G = growth(N)
    A = clip(A + dt*G + rand*np.random.rand(A.shape[0], A.shape[1])-0.5*rand)
    return (A, N, G)