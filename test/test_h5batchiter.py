from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import uuid
import numpy as np
import h5py
import tempfile

from h5minibatch import H5BatchReader, DataSetGroup

class TestReadEpochs( unittest.TestCase ) :

    def setUp(self) :
        self.numH5Files = 3
        self.samplesPerFile = 100
        self.h5filenames = []
        self.dsets = ['A',      'B',      'C',        'D',       'E',        'F']
        self.dtypes = [np.int8, np.int16, np.float32, np.float32, np.float64, np.int32]
        self.shapes = [(3,3),   (3,),     (),         (),         (),        ()]
        
        for idx in range(self.numH5Files):
            h5filename = 'unitest_' + uuid.uuid4().hex + ('_%3.3d' % idx) + '.h5'
            self.h5filenames.append(h5filename)
            h5 = h5py.File(h5filename,'w')
            for dset, dtype, shape in zip(self.dsets, self.dtypes, self.shapes):
                full_shape=(self.samplesPerFile,)+shape
                data = np.random.randint(0,100,size=full_shape)
                h5[dset] = data.astype(dtype)
            h5.close()

        self.truth = {}
        self.h5filenames.sort()

        for idx, fname in enumerate(self.h5filenames):
            h5 = h5py.File(fname,'r')
            for row in range(self.samplesPerFile):
                self.truth[(idx,row)] = {}
                for dset in self.dsets:
                   self.truth[(idx,row)][dset]=h5[dset][row]
        

    def test_split(self):
        batchReader = H5BatchReader(h5files=self.h5filenames,
                                    dsets=['A','C'],
                                    dset_groups=[DataSetGroup(name='vec', dsets=['D','F'])],
                                    verbose=True)
        batchReader.split(60,30,10)
        train_iter = batchReader.train_iter(batchsize=19, epochs=2)
        validation_iter = batchReader.validation_iter(batchsize=31, epochs=2)
        test_iter = batchReader.test_iter(batchsize=17, epochs=2)

        trainSamples = []
        validationSamples = []
        testSamples = []

        for batch in train_iter:
            trainSamples.extend(batch['filesRows'])
        for batch in test_iter:
            testSamples.extend(batch['filesRows'])
        for batch in test_iter:
            testSamples.extend(batch['filesRows'])

        numTrainSamples =  len(np.unique(trainSamples))
        numValidationSamples = len(np.unique(validationSamples))
        numTestSamples = len(np.unique(testSamples))

        allSamples = trainSamples
        allSamples.extend(validationSamples)
        allSamples.extend(testSamples)
        
        numAll = len(np.unique(allSamples))

        self.assertEqual(numAll, numTrainSamples+numValidationSamples+numTestSamples)
        
    def test_readThreeEpochs(self):
        batchReader = H5BatchReader(h5files=self.h5filenames,
                                    dsets=['A','C'],
                                    dset_groups=[DataSetGroup(name='vec', dsets=['D','F'])],
                                    verbose=True)

        train_iter = batchReader.train_iter(batchsize=10, epochs=3)

        files = np.zeros((900,), np.int)
        rows = np.zeros((900,), np.int)

        idx = 0
        for batch in train_iter:
            assert idx < 900
            sampleFiles = batch['filesRows']['file']
            sampleRows = batch['filesRows']['row']
            files[idx:(idx+10)] = sampleFiles
            rows[idx:(idx+10)] = sampleRows
            idx += 10
            self.assertEqual(batch['dset_groups']['vec'].shape,(10,2))
            for batchRow,sampleFile, sampleRow in zip(range(10), sampleFiles, sampleRows):
                for dset in ['A','C']:
                    diff = np.sum(batch['dsets'][dset][batchRow] - self.truth[(sampleFile,sampleRow)][dset])
                    self.assertAlmostEqual(diff, 0.0)
                readD = batch['dset_groups']['vec'][batchRow][0]
                readF = batch['dset_groups']['vec'][batchRow][1]
                self.assertAlmostEqual(readD, self.truth[(sampleFile,sampleRow)]['D'])
                self.assertAlmostEqual(readF, self.truth[(sampleFile,sampleRow)]['F'])

        for filenum in range(3):
            self.assertEqual(np.sum(files==filenum), 300)
        for row in range(100):
            self.assertEqual(np.sum(rows==row), 9)
        self.assertEqual(idx,900)

    def tearDown(self) :
        for h5filename in self.h5filenames:
            os.unlink(h5filename)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], '-v'])

