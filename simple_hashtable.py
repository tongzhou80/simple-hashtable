import numpy as np
from numba import njit

#@njit
def _init(val_dtype, size=8):
    # status 0: empty, 1: occupied, 2: deleted
    bucket_type = np.dtype([('status', np.ubyte), ('key', np.int32), ('val', val_dtype)])
    table = np.empty(size, dtype=bucket_type)
    table['status'] = 0
    return table


@njit
def _probe(key, table):
    h = key
    p = 0
    search = True
    while search:
        i = (h + p) % len(table)
        # Found an empty slot
        if table[i]['status'] == 0:
            search = False
        # Slot is occupied => stop search if key matches
        elif table[i]['status'] == 1 and table[i]['key'] == key:
            search = False
        # Skip deleted slots
        else:
            pass
        
        # Probing strategy
        p += 1
    return i


@njit
def _put(key, val, table):
    i = _probe(key, table)
    table[i]['status'] = 1
    table[i]['key'] = key
    table[i]['val'] = val


@njit
def _get(key, table, default=None):
    i = _probe(key, table)
    if table[i]['status'] == 1:
        return table[i]['val']
    else:
        return default
    

@njit
def _has(key, table):
    i = _probe(key, table)
    return table[i]['status'] == 1


@njit
def _delete(key, table):
    i = _probe(key, table)
    table[i]['status'] = 2


class SimpleDict:
    def __init__(self, val_dtype, size=8):
        self._val_dtype = val_dtype
        self._table = _init(val_dtype, size)

    def put(self, key, val):
        _put(key, val, self._table)

    def get(self, key, default=None):
        return _get(key, self._table, default)

    def has(self, key):
        return _has(key, self._table)

    def delete(self, key):
        _delete(key, self._table)

    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, val):
        self.put(key, val)

    def __contains__(self, key):
        return self.has(key)
    
    def __len__(self):
        return sum(self._table['status'] == 1)
    

