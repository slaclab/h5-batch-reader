from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import uuid
import unittest
import numpy as np
import h5py
import h5minibatch

class TestBatchIter( unittest.TestCase):
    def setUp(self):
        self.NROWS=30
        self.NFILES=10
        self.IMGROWS=6
        self.IMGCOLS=4
        self.h5files = ['test_batchiter_' + uuid.uuid4().hex + '.h5' for kk in range(self.NFILES)]
        for h5file in self.h5files:
            h5 = h5py.File(h5file,'w')
            h5['bldA'] = np.random.rand(self.NROWS)
            h5['bldB'] = np.random.rand(self.NROWS)
            h5['label'] = np.array([kk%4 for kk in range(self.NROWS)], np.int)
            mask = np.ones(self.NROWS, np.int8)
            mask[2]=0
            mask[9]=0
            mask[15]=0
            mask[25]=0
            h5['msk'] = mask
            h5['img'] = (100*np.random.rand(self.NROWS, self.IMGROWS, self.IMGCOLS)).astype(np.int16)
            h5.close()
        sortedFiles = [fname for fname in self.h5files]
        sortedFiles.sort()
        self.h5filesSorted = sortedFiles

    def testIter(self):

        def testBatch(self, batchDict):
            fvecs = batchDict['fvecs']
            self.assertEqual(fvecs.shape, (12, 2))
            for idx in range(12):
                bA,bB = fvecs[idx,:]
                bLab = batchDict['labels'][idx]
                ff,rr = batchDict['filesRows'][idx]
                h5fname = self.h5filesSorted[ff]
                h5 = h5py.File(h5fname,'r')
                fA=h5['bldA'][rr]
                fB=h5['bldB'][rr]
                fLab=h5['label'][rr]
                self.assertEqual(h5['msk'][rr],1)
                self.assertAlmostEqual(bA, fA, places=1)
                self.assertAlmostEqual(bB, fB, places=1)
                self.assertEqual(bLab, fLab)
                bImg = batchDict['datasets']['img'][idx]
                fImg = h5['img'][rr]
                imgDiff=np.sum(np.abs(bImg.astype(np.float) - fImg.astype(np.float)))
                self.assertAlmostEqual(imgDiff, 0.0, places=2)

        h5mini = h5minibatch.H5MiniBatchReader(h5files=self.h5files,
                                               include_if_one_mask_datasets=['msk'],
                                               verbose=True)
        h5mini.prepareClassification(label_dataset = 'label',
                                     datasets_for_feature_vector = ['bldA','bldB'],
                                     feature_image_datasets = ['img'],
                                     minibatch_size = 12,
                                     validation_size_percent=0.10,                
                                     test_size_percent=0.02) # test will only have

        testIter = h5mini.batchIterator(partition='test', epochs=1)
        self.assertEqual(testIter.samples.totalSamples,12)
        testBatchDicts = [batchDict for batchDict in testIter]
        self.assertEqual(len(testBatchDicts),1)
        batchDict = testBatchDicts[0]        
        testBatch(self, batchDict)

        validIter = h5mini.batchIterator(partition='validation', epochs=1)
        self.assertEqual(validIter.samples.totalSamples,24)
        validBatchDicts = [batchDict for batchDict in validIter]
        self.assertEqual(len(validBatchDicts),2)
        [testBatch(self, batchDict) for batchDict in validBatchDicts]

        curEpoch = 0
        samples = []
        for idx, batchDict in enumerate(h5mini.batchIterator(partition='train', epochs=2)):
            testBatch(self, batchDict)
            if batchDict['epoch']==1 and curEpoch==0:
                epoch0_samples = np.concatenate(samples)
                samples = []
            samples.append(batchDict['filesRows'])
            curEpoch = batchDict['epoch']
        epoch1_samples = np.concatenate(samples)
        epoch0_set = set([(fr[0],fr[1]) for fr in epoch0_samples])
        epoch1_set = set([(fr[0],fr[1]) for fr in epoch1_samples])
        self.assertEqual(epoch0_set, epoch1_set)
        self.assertEqual(len(epoch0_set), 12*18)
        self.assertEqual(idx, 2*18-1)
        
    def tearDown(self) :
        for h5filename in self.h5files:
            os.unlink(h5filename)

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])
