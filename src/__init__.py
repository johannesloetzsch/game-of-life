import model

def reload():
    import importlib
    for module in [model]:
        importlib.reload(module)