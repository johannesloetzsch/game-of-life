#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import plotly.express as px
from plotly.offline import plot

colors = [["red", "yellow", "lightblue", "lightgreen", "green"], "Aggrnyl_r", "algae", "Emrld", "Viridis", "Rainbow_r"]

def imshow(X, **kwargs):
    fig = px.imshow(X, color_continuous_scale=colors[0], zmin=-1, zmax=1, **kwargs)
    fig.update_xaxes(side="top")
    fig.show()
    plot(fig, auto_open=True)