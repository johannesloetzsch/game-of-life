#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import resource
resource.setrlimit(resource.RLIMIT_AS, (20_000_000_000, 30_000_000_000))

import simulation

def reload():
    import importlib
    for module in [simulation]:
        importlib.reload(module)

if __name__ == "__main__":
    A = simulation.simulate()