#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from model import step
import time
from visualization.plotly import imshow

def dtf(state_i):
    return .1

def record_steps_quadratic(lin, frames, steps):
    f = max(1, frames-1)
    b = lin * (steps-1)/f
    a = ((steps-1) - b*f) / f**2
    return [int(a*i**2 + b*i) for i in range(frames)]

def callback(A, N, G, step, time_start, record_steps, show_img=True):
    if step in record_steps:
        img = G
        #img = np.concatenate([A, N, G], axis=1)
        if show_img:
            imshow(G)
            #imshow(img)
            #imshow(np.array([A, N, G]), animation_frame=0)
        time_simulation = time.perf_counter() - time_start
        print("step: {:3d}, fps: {:3.1f}".format(step, step/time_simulation))
        return img

def simulate(A=None, steps=1000, callback=callback):
    if A is None:
        np.random.seed(0)
        A = np.random.rand(100, 100)

    frames = []
    record_steps = record_steps_quadratic(lin=0, frames=10, steps=steps)
    time_start = time.perf_counter()

    for i in range(steps):
        dt = dtf(i)
        (A, N, G) = step(A, dt)
        img = callback(A, N, G, i+1, time_start, record_steps)
        if not img is None:
            frames.append(img)

    imshow(np.array(frames), animation_frame=0)
    return A