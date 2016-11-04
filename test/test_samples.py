from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import uuid
import numpy as np
import h5py

from h5minibatch import Samples

def maskOne(N,exclude):
    mask = np.ones(N,np.int8)
    for idx in exclude:
        mask[idx]=0
    return mask

def maskNegOne(N,exclude):
    mask = np.ones(N,np.int8)
    for idx in exclude:
        mask[idx]=-1
    return mask

class TestSamples( unittest.TestCase ) :
    def setUp(self):
        self.h5filenames = ['unitest_' + uuid.uuid4().hex + '.h5' for k in range(3)]
        self.N=30        
        maskA=maskOne(N=self.N, exclude=[3,10])
        maskB=maskOne(N=self.N, exclude=[7,20,22])
        negoneA = maskNegOne(N=self.N, exclude=[22,25])
        negoneB = maskNegOne(N=self.N, exclude=[9])
        self.maskedRows = [3,7,9,10,20,22,25]
        labels = [[k%4 for k in range(self.N)],
                  [k%3 for k in range(self.N)]]
        for fname in self.h5filenames:
            h5 = h5py.File(fname,'w')
            h5['maskA']=maskA
            h5['maskB']=maskB
            h5['negoneA']=negoneA
            h5['negoneB']=negoneB
            h5['labelA']=labels[0]
            h5['labelB']=labels[1]
            h5.close()

    def tearDown(self):
        for fname in self.h5filenames:
            os.unlink(fname)

    def testSamples(self):
        samples = Samples(self.h5files, ['labelA', 'labelB'],
                          ['maskA', 'maskB'],
                          ['negoneA', 'negoneB'])
        h5s = self.h5files.deepcopy()
        h5s.sort()
        self.assertEqual(len(h5s), len(samples.h5files))
        for xx,yy in zip(h5s, samples.h5files):
            self.assertEqual(xx,yy)



if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])

