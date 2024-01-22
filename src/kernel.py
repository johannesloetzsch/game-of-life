#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from functions import gaussian

def normalized_array(A):
    return np.array(A)/np.sum(A)

K3 = normalized_array([
[1, 1, 1],
[1, 0, 1],
[1, 1, 1]])

K11 = normalized_array([
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
[1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
[1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]])

K11_discrete = normalized_array(gaussian([[math.sqrt((X-5)**2+(Y-5)**2)-4 for X in range(11)] for Y in range(11)])
                                #gaussian([[math.sqrt((X-5)**2+(Y-5)**2)-1 for X in range(11)] for Y in range(11)])
                               )
