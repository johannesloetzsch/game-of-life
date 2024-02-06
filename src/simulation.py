#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from model import step
import time
from visualization.plotly import imshow

def dtf(state_i):
    return 1.0

def record_steps_quadratic(lin, frames, steps):
    f = max(1, frames-1)
    b = lin * (steps-1)/f
    a = ((steps-1) - b*f) / f**2
    return [int(a*i**2 + b*i) for i in range(frames)]

def callback(N, T, A, step, time_start, record_steps, show_img=True):
    if step in record_steps:
        #img = G
        img = np.concatenate([N, T, A], axis=1)
        if show_img:
            #imshow(G)
            imshow(img)
            #imshow(np.array([A, N, G]), animation_frame=0)
        time_simulation = time.perf_counter() - time_start
        print("step: {:3d}, fps: {:3.1f}".format(step, step/time_simulation))
        return img

def simulate(A=None, steps=10, callback=callback, p=[-0.32, 2.82, 0.14], overshoot=0.95, offset=0, round=True, dt=None, const=None, K=None):
    if A is None:
        np.random.seed(0)
        #A = np.random.rand(100, 100)*2-1
        #from figures.conway import demo
        #A = demo*2-1
        from figures.conway import init, glider
        A = init(glider, 10, 10)*2-1

    frames = []
    record_steps = record_steps_quadratic(lin=1, frames=10, steps=steps)
    time_start = time.perf_counter()

    for i in range(steps):
        if dt is None:
            dt = dtf(i)
        (N, T, A) = step(A, p, overshoot=overshoot, offset=offset, round=round, dt=dt, K=K)
        if not const is None:
            A = np.concatenate([const[:1], A[1:]])
        img = callback(N, T, A, i+1, time_start, record_steps)
        if not img is None:
            frames.append(img)

    imshow(np.array(frames), animation_frame=0)
    return A