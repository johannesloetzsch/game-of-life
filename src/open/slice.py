#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def top(frame0, i, grow_dimension=0, other_dimension=1, minimum=-1):
   assert(grow_dimension == 0)
   cut = frame0[0:i][:]
   empty = minimum + np.zeros([frame0.shape[grow_dimension]-i, frame0.shape[other_dimension]])
   return np.concatenate([cut, empty])

def slice(frame0, grow_dimension=0, other_dimension=1):
    frames = [top(frame0, i) for i in range(frame0.shape[grow_dimension]-1)]
    return np.array(frames)