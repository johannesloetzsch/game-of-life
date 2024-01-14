# -*- coding: utf-8 -*-
"""
Model for growth/target function
"""

import resource
resource.setrlimit(resource.RLIMIT_AS, (20_000_000_000, 30_000_000_000))

from functools import partial
import numpy as np
import scipy
import math

convolve2d = scipy.signal.convolve2d
gaussian = scipy.stats.norm.pdf
sigmoid = scipy.special.expit

## Kernel

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
K11_discrete = normalized_array(gaussian([[math.sqrt((X-5)**2+(Y-5)**2)-4 for X in range(11)] for Y in range(11)]) + gaussian([[math.sqrt((X-5)**2+(Y-5)**2)-1 for X in range(11)] for Y in range(11)]))
K = K11_discrete

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

## Model

clip = lambda A: np.clip(A, 0, 1)
#clip = lambda A, m=20: sigmoid(m*(A-.5))

def growth(N):
    u = 0.209
    v = 2
    w = 0.05
    G_unlimited = -1+2*v*gaussian((N-u)/w)
    #return G_unlimited
    return np.clip(G_unlimited, -1, 1)
    #return -1*2*sigmoid(30*G_unlimited)

def step(A, dt=0.02, rand=0):
    N = convolve2d(A, K, mode='same', boundary='wrap')
    #N = convolve2d(A, K, mode='same', boundary='fill', fillvalue=0)
    G = growth(N)
    A = clip(A + dt*G + rand*np.random.rand(A.shape[0], A.shape[1])-0.5*rand)
    return (A, G, N)

## Simulation

def dtf(state_i):
    return .5

states = 100
print("simulate {} states…".format(states))
print("t_max={}".format(int(states*dtf(states))))
np.random.seed(0)
#A = demo
A = np.random.rand(600, 800)
#A = init(K3) + 0.5*np.random.rand(100, 100)
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
    #A = np.ndarray.round(A_new)
print("✔ simulation finished")

## Visualization

import matplotlib.animation
import matplotlib.pyplot as plt

def animate(frame, frames, states, dtf, fps, a, b, axs):
    (ax1, ax2, ax3) = axs
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

def plot(frames=100, fps=2, lin=1, dtf=dtf):
    fig, axs = plt.subplots(1, 3)
    print("render {} frames with {} fps…".format(frames, fps))
    f = max(1, frames-1)
    b = lin * (states-1)/f
    a = ((states-1) - b*f) / f**2
    print("lin={} => a={:.2f}, b={:.2}".format(lin, a, b))
    plt.rcParams["animation.html"] = "html5"
    plt.ioff()
    plt.cla()
    f = partial(animate, dtf=dtf, frames=frames, states=states, fps=fps, a=a, b=b, axs=axs)
    return matplotlib.animation.FuncAnimation(fig, f, frames=frames, interval=1000/fps)

#plt.show()
#plot()

def plotly(X):
    import plotly.express as px
    from plotly.offline import plot
    fig = px.imshow(X)
    fig.update_xaxes(side="top")
    fig.show()
    plot(fig, auto_open=True)
    
def plotlib(X):
    fig, ax = plt.subplots()
    ax.imshow(X)
    plt.show()
    
if __name__ == "__main__":
    frames = 6
    #frame_indices = range(0, frames)
    frame_indices = range(0, states, int(states/frames))
    F = [np.concatenate([A_states[i],N_states[i],G_states[i]]) for i in frame_indices]
    img = np.concatenate(F, axis=1)
    #plotly(img)
    plotlib(img)
