from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import uuid
import numpy as np
import h5py

from h5minibatch import utils

class TestMakeMask( unittest.TestCase ) :
    def setUp(self):
        self.h5filename = 'unitest_' + uuid.uuid4().hex + '.h5'
        h5 = h5py.File(self.h5filename,'w')
        self.N=100

        maskA=np.ones(self.N, np.int)
        maskB=np.ones(self.N, np.int)
        self.excludeA=[3,20]
        self.excludeB=[25,58]
        for mask, exclude, nm in zip([maskA,maskB],
                                     [self.excludeA, self.excludeB],
                                     ['maskA','maskB']):
            for idx in exclude:
                mask[idx]=0
            h5[nm] = mask

        negoneC=np.ones(self.N, np.int)
        negoneD=np.ones(self.N, np.int)
        self.excludeC=[13,70]
        self.excludeD=[15,84]
        for negone, exclude, nm in zip([negoneC,negoneD],
                                       [self.excludeC, self.excludeD],
                                       ['negoneC','negoneD']):
            for idx in exclude:
                negone[idx]=-1
            h5[nm] = negone
        h5.close()

    def testMakeMask(self):
        h5 = h5py.File(self.h5filename, 'r')
        mask = utils.makeMask(h5, exclude_if_negone_mask_datasets=['negoneC'])
        self.assertEqual(len(mask),self.N)
        self.assertEqual(mask.dtype, bool, msg='mask dtyp=%s' % mask.dtype)
        for idx in range(self.N):
            if idx in self.excludeC:
                self.assertEqual(mask[idx],False)
            else:
                self.assertEqual(mask[idx],True)

        mask = utils.makeMask(h5, exclude_if_negone_mask_datasets=['negoneC', 'negoneD'])
        for idx in range(self.N):
            if idx in self.excludeC or idx in self.excludeD:
                self.assertEqual(mask[idx],False)
            else:
                self.assertEqual(mask[idx],True)
        

        mask = utils.makeMask(h5, include_if_one_mask_datasets=['maskA'])
        for idx in range(self.N):
            if idx in self.excludeA:
                self.assertEqual(mask[idx],False)
            else:
                self.assertEqual(mask[idx],True)

        mask = utils.makeMask(h5, include_if_one_mask_datasets=['maskA', 'maskB'])
        for idx in range(self.N):
            if (idx in self.excludeA) or (idx in self.excludeB):
                self.assertEqual(mask[idx],False)
            else:
                self.assertEqual(mask[idx],True)

        mask = utils.makeMask(h5, exclude_if_negone_mask_datasets=['negoneC', 'negoneD'],
                              include_if_one_mask_datasets=['maskA', 'maskB'])
        for idx in range(self.N):
            if idx in self.excludeA or idx in self.excludeB or idx in self.excludeC or idx in self.excludeD:
                self.assertEqual(mask[idx],False)
            else:
                self.assertEqual(mask[idx],True)


    def tearDown(self):
        os.unlink(self.h5filename)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])

