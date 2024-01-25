#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageSequence, ImageOps
from urllib.request import urlopen
import numpy as np

def convert(frame, scale):
    frame = ImageOps.scale(frame, scale, Image.Resampling.NEAREST)
    frame = ImageOps.invert(frame.convert("RGB")).convert("L")
    return frame

def normalize(frame, factor=2, offset=1):
    return frame/255*factor - offset

def denormalize(frame, factor=2, offset=1):
    return ((frame+offset)/factor*255).astype(np.uint8)

def image_open(url, scale=1):
    pic = Image.open(urlopen(url))
    frames = np.array([normalize(np.array(convert(frame, scale)))
                       for frame in ImageSequence.Iterator(pic)])
    return frames