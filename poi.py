# -*- coding: utf-8 -*-
import etcd3
import hnswlib
import numpy as np
import os

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')
    
def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

INITIAL_ELEMENTS = 10000
INDEX_PERSIST_FILE = "index.bin"

class Poi(object):
    def __init__(self, etcdConf, dim, ef, m):
        self.dim = dim
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
        self.index.set_ef(ef)
    
    def changeMaxElems(self, e):
        self.max_elements = e
    
    def flushIndex(self):
        self.persistIndex()
        self.index = hnswlib.Index(space='l2', dim=self.dim)
        self.loadIndex()
    
    def persistIndex(self):
        self.index.save_index(INDEX_PERSIST_FILE)
    
    def loadIndex(self):
        self.index.load_index(INDEX_PERSIST_FILE, max_elements = self.max_elements)
    
    def insert(self, vid, vector):
        self.database.put(int_to_bytes(vid), vector.tobytes())
        try:
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
    
    def get(self, vid):
        res = self.database.get(int_to_bytes(vid))[0]
        if res:
            return np.frombuffer(res, dtype="float32")
        else:
            return None
        
    def query(self, vector, k):
        res = self.index.knn_query(vector, k)
        return [(res[0][0][i], res[1][0][i]) for i in range(k)]

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