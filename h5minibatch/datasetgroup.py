from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import h5py

class DataSetGroup(object):
    def __init__(self, name, dsets, dtype=np.float32):
        assert isinstance(name, str)
        assert isinstance(dsets,list)
        self.name=name
        self.dsets = copy.deepcopy(dsets)
        self.dtype = dtype
        self._validated = False

    def __str__(self):
        return "dset_group_%s" % self.name

    def __repr__(self):
        return "DataSetGroup(name=%s, dsets=%r, dtype=%r)" % (self.name, self.dsets, self.dtype)
    
    def validate(self, h5in):
        if isinstance(h5in,str):
            h5in = h5py.File(h5in,'r')
        if self._validated: return
        for dset in self.dsets:
            assert dset in h5in, "dset=%s not a datset in h5 file: %s" % (name, h5in.filename)
            h5_dset = h5in[dset]
            assert hasattr(h5_dset, 'shape'), "dset=%s in h5 file: %s is not an array" % (dset, h5in.filename)
            assert len(h5_dset.shape)==1, "name=%s in h5 file %s not an array of scalar" % (name, h5in.filename)
        self._validated = True
        return True

    def read_sample(self, h5, row):
        res = np.empty(shape=(len(self.dsets),), dtype=self.dtype)
        for idx,dset in enumerate(self.dsets):
            res[idx]=h5[dset][row]
        return res
            
    def __len__(self):
        return len(self.dsets)
