# -*- coding: utf-8 -*-
from contextlib import contextmanager
from threading  import Lock
import etcd3
import hnswlib
import numpy as np
import os

class RWLock(object):
    """ RWLock class; this is meant to allow an object to be read from by
        multiple threads, but only written to by a single thread at a time. See:
        https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock
        Usage:
            from rwlock import RWLock
            my_obj_rwlock = RWLock()
            # When reading from my_obj:
            with my_obj_rwlock.r_locked():
                do_read_only_things_with(my_obj)
            # When writing to my_obj:
            with my_obj_rwlock.w_locked():
                mutate(my_obj)
    """

    def __init__(self):

        self.w_lock = Lock()
        self.num_r_lock = Lock()
        self.num_r = 0

    # ___________________________________________________________________
    # Reading methods.

    def r_acquire(self):
        self.num_r_lock.acquire()
        self.num_r += 1
        if self.num_r == 1:
            self.w_lock.acquire()
        self.num_r_lock.release()

    def r_release(self):
        assert self.num_r > 0
        self.num_r_lock.acquire()
        self.num_r -= 1
        if self.num_r == 0:
            self.w_lock.release()
        self.num_r_lock.release()

    @contextmanager
    def r_locked(self):
        """ This method is designed to be used via the `with` statement. """
        try:
            self.r_acquire()
            yield
        finally:
            self.r_release()

    # ___________________________________________________________________
    # Writing methods.

    def w_acquire(self):
        self.w_lock.acquire()

    def w_release(self):
        self.w_lock.release()

    @contextmanager
    def w_locked(self):
        """ This method is designed to be used via the `with` statement. """
        try:
            self.w_acquire()
            yield
        finally:
            self.w_release()

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')
    
def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

INITIAL_ELEMENTS = 1
INDEX_PERSIST_FILE = "index.bin"

class Poi(object):
    def __init__(self, etcdConf, dim, ef, m):
        self.dim = dim
        self.lock = lock.RWLock()
        self.database = etcd3.client(**etcdConf)
        
        maxele_raw = self.database.get('max_element')[0]
        if maxele_raw:
            self.max_elements = int_from_bytes(maxele_raw)
        else:
            self.max_elements = INITIAL_ELEMENTS
            self.database.put('max_element', int_to_bytes(INITIAL_ELEMENTS))
        
        self.index = hnswlib.Index(space='l2', dim=dim)
        if os.path.exists(INDEX_PERSIST_FILE):
            self.loadIndex()
        else:
            self.index.init_index(max_elements=self.max_elements, ef_construction=ef, M=m)
            self.persistIndex()
    
    def changeMaxElems(self, e):
        self.max_elements = e
    
    def flushIndex(self):
        self.persistIndex()
        self.index = hnswlib.Index(space='l2', dim=self.dim)
        self.loadIndex()
    
    def persistIndex(self):
        try:
            self.lock.w_acquire()
            self.index.save_index(INDEX_PERSIST_FILE)
        finally:
            self.lock.w_release()
    
    def loadIndex(self):
        try:
            self.lock.w_acquire()
            self.index.load_index(INDEX_PERSIST_FILE, max_elements = self.max_elements)
        finally:
            self.lock.w_release()
    
    def insert(self, vid, vector):
        self.database.put(int_to_bytes(vid), vector.tobytes())
        try:
            self.lock.w_acquire()
            self.index.add_items([ vector ], [ vid ])
        except RuntimeError as e:
            if "exceeds" in str(e):
                print(e)
                self.max_elements *= 2
                self.database.put('max_element', int_to_bytes(self.max_elements))
                self.flushIndex()
                self.index.add_items([ vector ], [ vid ])
            else:
                raise e
        finally:
            self.lock.w_release()
    
    def get(self, vid):
        res = self.database.get(int_to_bytes(vid))[0]
        if res:
            return np.frombuffer(res, dtype="float32")
        else:
            return None
        
    def query(self, vector, k):
        try:
            self.lock.r_acquire()
            res = self.index.knn_query(vector, k)
            return [(res[0][0][i], res[1][0][i]) for i in range(k)]
        finally:
            self.lock.r_release()

    def queryById(self, vid, k):
        vector = self.get(vid)
        if vector is None:
            return []
        return self.query(vector, k)
        
    def rebuildIndex(self):
        pass


if __name__ == "__main__":
    poi = Poi({
            "host": 'localhost',
            "port": 2379
        }, 768, 100, 16)
    #poi.insert(6, np.float32(np.random.random(768)))
    poi.persistIndex()
    print(poi.queryById(3, 4))