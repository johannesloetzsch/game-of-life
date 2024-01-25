#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import plotly.express as px
from plotly.offline import plot

colors = [["black", "brown", "blue", "green", "lightgreen"], "Aggrnyl_r", "algae", "Emrld", "Viridis", "Rainbow_r"]

def imshow(X, auto_open=False, **kwargs):
    fig = px.imshow(X, color_continuous_scale=colors[0], zmin=-1, zmax=1, **kwargs)
    fig.update_xaxes(side="top")
    fig.show()
    plot(fig, auto_open=auto_open)