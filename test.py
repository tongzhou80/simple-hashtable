import numpy as np
from simple_hashtable import SimpleDict

np.random.seed(0)
d = SimpleDict(size=20003, val_dtype=np.int32)
N = 10000
keys = np.random.randint(10000, size=N)
vals = np.random.randint(10000, size=N)
for k, v in zip(keys, vals):
    d[k] = v
    if k % 3 == 0:
        d.delete(k)
    
d1 = {}
for k, v in zip(keys, vals):
    d1[k] = v
    if k % 3 == 0:
        del d1[k]

for k, v in zip(keys, vals):
    if d.get(k, None) != d1.get(k, None):
        print(f'key: {k}, val: {v}, d: {d[k]}, d1: {d1[k]}')
    assert d.get(k, None) == d1.get(k, None)