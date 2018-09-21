import numpy as np

def make_batch(x):
    b = []
    for i in range(len(x[0])):
        z = [el[i] for el in x]
        b.append(np.asarray(z))
    return b