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
    return G_unlimited
    #return np.clip(G_unlimited, -1, 1)

def step(A, dt=0.02):
    N = convolve2d(A, K, mode='same', boundary='wrap')
    G = growth(N)
    A = clip(A+dt*G)
    return (A, G, N)

## Simulation

def dtf(state_i):
    return 0.0005

states = 10000
print("simulate {} states…".format(states))
print("t_max={}".format(int(states*dtf(states))))
np.random.seed(0)
A = np.random.rand(100, 100)
A_states = []
G_states = []
N_states = []
for i in range(states):
    dt = dtf(i)
    (A_new, G, N) = step(A, dt)
    A_states.append(A)
    G_states.append(G)
    N_states.append(N)
    A = A_new
print("✔ simulation finished")

## Visualization

import matplotlib.animation
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

def animate(frame, frames, states, dtf, fps, a, b):
    #state = int(frame/frames*states)
    state = int(a*frame**2 + b*frame)
    dt = dtf(state)
    t = state*dt
    t_formater = "{:." + str(math.ceil(-math.log10(dt))) + "f}"
    t_format = lambda t: t_formater.format(t)
    ax1.set_title("$A("+t_format(t)+") = \int G(t) dt$")
    ax1.axis('off')
    ax1.imshow(A_states[state])
    ax2.set_title("$N(t) = conv(K,A(t))$")
    ax2.axis('off')
    ax2.imshow(N_states[state])
    ax3.set_title("$G(t) = \\frac{dA}{dt} = g(N(t))$")
    ax3.axis('off')
    ax3.imshow(G_states[state])
    #plt.colorbar(img3)
    if __name__ == "__main__":
        plt.pause(1/fps)

def plot(frames=100, fps=5, lin=0, dtf=dtf):
    print("render {} frames with {} fps…".format(frames, fps))
    b = lin
    a = ((states-1) - b*(frames-1)) / (frames-1)**2
    print("lin={} => a={}, b={}".format(lin, a, b))
    plt.rcParams["animation.html"] = "html5"
    plt.ioff()
    plt.cla()
    f = partial(animate, dtf=dtf, frames=frames, states=states, fps=fps, a=a, b=b)
    return matplotlib.animation.FuncAnimation(fig, f, frames=frames, interval=1000/fps)

#plt.show()
#plot()
