#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

def init(M, x=100, y=100):
    M = np.array(M)
    y2 = (y-M.shape[1])/2
    x2 = (x-M.shape[0])/2
    return np.insert( np.insert(M, math.floor(y2)*[0]+math.ceil(y2)*[M.shape[1]], 0, axis=1),
                                   math.floor(x2)*[0]+math.ceil(x2)*[M.shape[0]], 0, axis=0)

glider = np.array([
[0, 1, 0],
[0, 0, 1],
[1, 1, 1]])

sidecar = [
[0, 1, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 1, 0],
[1, 0, 0, 0, 0, 0, 1, 0],
[1, 1, 1, 1, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 1],
[0, 1, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 1],
[0, 1, 1, 1, 1, 1, 1, 0]]

loafer = [
[0, 1, 1, 0, 0, 1, 0, 1, 1],
[1, 0, 0, 1, 0, 0, 1, 1, 0],
[0, 1, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 1, 1, 1],
[0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 1]]

demo = np.concatenate([
    np.concatenate([init(glider, 50, 5), init([[]], 50, 25), init(sidecar, 50, 15), init([[]], 50, 5), init(np.rot90(glider), 50, 50)], axis=1),
    np.concatenate([init(loafer, 50, 25), init(loafer, 50, 50), init(glider.T, 50, 25)], axis=1)])