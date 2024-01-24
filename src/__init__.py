#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import resource
resource.setrlimit(resource.RLIMIT_AS, (20_000_000_000, 30_000_000_000))

import numpy as np
import train.fit
import simulation

def reload():
    import importlib
    for module in [simulation]:
        importlib.reload(module)

if __name__ == "__main__":
    random_faktor = 1e-1
    X = np.array([0, 1, 2, 2.9, 3, 3.1, 4, 5, 6, 7, 8])/8*2-1
    Y = [-1, -1, 0, 0.95, 1, 0.95, -1, -1, -1, -1, -1] + random_faktor*np.random.rand(len(X))-random_faktor/2
    
    popt, pcov = train.fit.fit_plot(X, Y)
    
    A = simulation.simulate(p=popt)