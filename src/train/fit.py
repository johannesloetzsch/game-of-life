#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as o
import scipy
gaussian = scipy.stats.norm.pdf

def growth(N, u, v, w):
    return -1+2*v*gaussian((N-u)/w)

def target(A, N, p, overshoot=0.5):
    [u, v, w] = p
    target_raw = A + (1+overshoot)*growth(N, u, v, w)
    #target_raw = 2*growth(N, u, v, w)
    return np.clip(target_raw, -1, 1)

def plot_training_data(X, Y):
    plt.scatter(X, Y)
    plt.show()

def fit(X, Y, p0=[2.75, 2, 0.5]):
    popt, pcov = o.curve_fit(growth, X, Y, p0=p0)
    return popt, pcov

#popt, pcov = fit(X, Y)

def fit_plot(X_data, Y_data, p0=None):
    popt, pcov = fit(X_data, Y_data, p0)
    print(popt)
    
    X = np.linspace(min(X_data), max(X_data), 200)
    Y = growth(X, *popt)
    T1 = target(1, X, popt)
    T05 = target(0.5, X, popt)
    T0 = target(0, X, popt)
    Tn05 = target(-0.5, X, popt)
    Tn1 = target(-1, X, popt)
    
    fig, ax = plt.subplots()
    ax.scatter(X_data, Y_data, label="training")
    ax.plot(X, Y, color="black", label="G", linestyle="dotted")
    ax.plot(X, T1, color="green", label="T 1", linestyle="dotted")
    ax.plot(X, T05, color="green", label="T .5")
    ax.plot(X, T0, color="yellow", label="T 0")
    ax.plot(X, Tn05, color="red", label="T -.5")
    ax.plot(X, Tn1, color="red", label="T -1", linestyle="dotted")
    ax.legend()
    ax.set(ylim=(-1.1, 1.5))
    plt.show()
    
    print("cond(pcov) =", np.linalg.cond(pcov))
    plt.imshow(np.log10(np.clip(np.abs(pcov), 1e-12, None)))
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
    return popt, pcov

if __name__ == "__main__":
    random_faktor = 1e-1
    X = np.array([0, 1, 2, 2.9, 3, 3.1, 4, 5, 6, 7, 8])/8*2-1
    Y = [-1, -1, 0, 0.95, 1, 0.95, -1, -1, -1, -1, -1] + random_faktor*np.random.rand(len(X))-random_faktor/2
    
    popt, pcov = fit_plot(X, Y)