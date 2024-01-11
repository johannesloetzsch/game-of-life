# -*- coding: utf-8 -*-
"""
Model for growth/target function
"""

from functools import partial
import numpy as np
import scipy
import math


## Kernel

def normalized_array(A):
    return A
    #return np.array(A)/np.sum(A)

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
K=K3

## Model

convolve2d = scipy.signal.convolve2d
gaussian = scipy.stats.norm.pdf
sigmoid = scipy.special.expit
clip = lambda A: np.clip(A, 0, 1)  # sigmoid

def growth(N):
    u = 2.65
    v = 4.5
    w = 0.37
    G_unlimited = -1+2*v*gaussian((N-u)/w)
    return np.clip(G_unlimited, -1, 1)

dt = 0.1
t_formater = "{:." + str(math.ceil(-math.log10(dt))) + "f}"
t_format = lambda t: t_formater.format(t)
def step(A):
    N = convolve2d(A, K, mode='same', boundary='wrap')
    G = growth(N)
    A = clip(A+dt*G)
    return (A, G, N)



np.random.seed(0)
A = 0.75*np.random.rand(100, 100)
A_frames = []
G_frames = []
N_frames = []
frames = 100
for f in range(frames):
    (A_new, G, N) = step(A)
    A_frames.append(A)
    G_frames.append(G)
    N_frames.append(N)
    A = A_new


import matplotlib.animation
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

def animate(f, dt=dt):
    t = f*dt
    plt.cla()
    ax1.set_title("$A("+t_format(t)+") = \int G(t) dt$")
    ax1.axis('off')
    ax1.imshow(A_frames[f])
    ax2.set_title("$N(t) = conv(K,A(t))$")
    ax2.axis('off')
    ax2.imshow(N_frames[f])
    ax3.set_title("$G(t) = \\frac{dA}{dt} = g(N(t))$")
    ax3.axis('off')
    ax3.imshow(G_frames[f])
    if __name__ == "__main__":
        plt.pause(0.01)

def plot(frames=frames, dt=dt):
    plt.rcParams["animation.html"] = "html5"
    plt.ioff()
    f = partial(animate, dt=dt)
    return matplotlib.animation.FuncAnimation(fig, f, frames=frames, interval=1000)

#plt.show()
#plot()