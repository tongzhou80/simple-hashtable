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
def _put(key, val, table, usage_info):
    i = _probe(key, table)
    if table[i]['status'] != 1:
        usage_info[0] += 1
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
def _delete(key, table, usage_info):
    i = _probe(key, table)
    if table[i]['status'] == 1:
        usage_info[0] -= 1
        usage_info[1] += 1
    table[i]['status'] = 2


@njit
def _rehash(old_table, new_table, old_usage_info, new_usage_info):
    for i in range(len(old_table)):
        if old_table[i]['status'] == 1:
            _put(old_table[i]['key'], old_table[i]['val'], new_table, new_usage_info)


class SimpleDict:
    def __init__(self, val_dtype, size=8):
        self.val_dtype = val_dtype
        self.table = _init(val_dtype, size)
        # (num_of_occupied_buckets, num_of_deleted_buckets)
        self.usage_info = [0, 0]

    def put(self, key, val):
        _put(key, val, self.table, self.usage_info)
        if (sum(self.usage_info) / len(self.table)) > 0.7:
            self.rehash()

    def get(self, key, default=None):
        return _get(key, self.table, default)

    def has(self, key):
        return _has(key, self.table)

    def delete(self, key):
        _delete(key, self.table, self.usage_info)

    def rehash(self):
        new_table = _init(self.val_dtype, len(self.table) * 4)
        new_usage_info = [0, 0]
        _rehash(self.table, new_table, self.usage_info, new_usage_info)
        self.table = new_table
        self.usage_info = new_usage_info

    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, val):
        self.put(key, val)

    def __contains__(self, key):
        return self.has(key)
    
    def __len__(self):
        return self.usage_info[0]
    

