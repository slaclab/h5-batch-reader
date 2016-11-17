from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

########## imports ##########
# python system
import os
import sys
import uuid
import unittest

#3rd party
import numpy as np
import h5py

# this package
from h5batchreader import DataSetGroup

class TestDatasetGroup(unittest.TestCase):
    def setUp(self):
        self.FNAME = 'test_datasetGroup_' + str(uuid.uuid4()) + '.h5'
        h5=h5py.File(self.FNAME,'w')
        h5['A'] = np.zeros(10)
        h5['B'] = np.ones(10, dtype=np.int)
        h5['C'] = 2*np.ones(10)
        h5['D'] = -1*np.ones(10)
        h5.close()

    def tearDown(self):
        os.unlink(self.FNAME)
        
    def test_datasetGroup(self):
        fvecnames=['A','B','C']
        dgroup = DataSetGroup(name='bld', dsets=fvecnames)
        print(dgroup)
        print("%r" % dgroup)
        dgroup.validate(self.FNAME)
        print("number of datasets in group: %d" % len(dgroup))
        h5 = h5py.File(self.FNAME,'r')
        sample = dgroup.read_sample(h5,2)
        self.assertEqual(sample[0],0)
        self.assertEqual(sample[1],1)
        self.assertEqual(sample[2],2)
        

if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0],'-v'])
    
